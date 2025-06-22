# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model head modules."""

import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_
import torch.nn.functional as F

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, make_anchors

from .block import DFL, Proto, BNContrastiveHead, ContrastiveHead, BNContrastiveHeadMLP, ContrastiveHeadMLP
from .conv import Conv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, DecouplingDeformableTransformerDecoderLayer, DecouplingDFLDeformableTransformerDecoderLayer, TextDeformableTransformerDecoder, locationDeformableTransformerDecoder, DecouplingTextDeformableTransformerDecoder, DecouplingDFLTextDeformableTransformerDecoder
from ..extra_modules.VManba.vmamba import VSSBlock
from .utils import bias_init_with_prob, linear_init_

__all__ = 'Detect', 'Segment', 'Pose', 'Classify', 'RTDETRDecoder', 'DualRTDETRDecoder', 'ManbaDecoder', 'ManbaWorldDecoder', 'locationManbaDecoder', 'DecouplingManbaWorldDecoder', 'DecouplingDFLManbaWorldDecoder'


class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class Segment(Detect):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class Pose(Detect):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3].sigmoid_()  # inplace sigmoid
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLOv8 classification head with specified input and output channels, kernel size, stride,
        padding, and groups.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """
    export = False  # export mode

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hd=256,  # hidden dim
            nq=300,  # num queries
            ndp=4,  # num decoder points
            nh=8,  # num head
            ndl=6,  # num decoder layers
            d_ffn=1024,  # dim of feedforward
            eval_idx=-1,
            dropout=0.,
            act=nn.ReLU(),
            # Training args
            nd=100,  # num denoising
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
            使用给定参数初始化 RTDETRDecoder 模块。

         参数：
             nc (int)：类数。 默认值为 80。
             ch（元组）：主干特征图中的通道。 默认值为 (512, 1024, 2048)。
             hd (int)：隐藏层的维度。 默认值为 256。
             nq (int)：查询点的数量。 默认值为 300。
             ndp (int)：解码器点数。 默认值为 4。
             nh (int)：多头注意力中的头数。 默认值为 8。
             ndl (int)：解码器层数。 默认值为 6。
             d_ffn (int)：前馈网络的维度。 默认值为 1024。
             dropout (float): 辍学率。 默认值为 0。
             act (nn.Module)：激活函数。 默认为 nn.ReLU。
             eval_idx(int)：评价指标。 默认值为-1。
             nd (int): 去噪次数。 默认值为 100。
             label_noise_ratio (float)：标签噪声比。 默认值为 0.5。
             box_noise_scale (float)：盒子噪声尺度。 默认值为 1.0。
             learnt_init_query (bool)：是否学习初始查询嵌入。 默认值为 False。
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc + 1, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group
        #(Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Dict]]): 

        # Input projection and embedding
        # feats:[b, 20x20 + 40x40 + 80x80, c] shapes: [nl, 2]
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training 
        #The modified class embeddings, bounding boxes, attention mask and meta information for denoising. 
        # 三维张量[bs, num_dn, dn_cls_embed], [bs, num_dn, 4], [num_dn + num_queries, num_dn + num_queries], {'dn_pos_idx','dn_num_group': num_group,'dn_num_split': [num_dn, num_queries]}
        dn_embed, dn_bbox, attn_mask, dn_meta = \
            get_cdn_group(batch,
                          self.nc,
                          self.num_queries,
                          self.denoising_class_embed.weight,
                          self.num_denoising,
                          self.label_noise_ratio,
                          self.box_noise_scale,
                          self.training)
        # embed = [bs, num_dn + nq, dn_cls_embed = hd（256）] refer_bbox: (bs, num_queries + num_dn, 4) enc_bboxes: (bs, num_queries, 4) enc_scores: (bs, num_queries, nc)
        embed, refer_bbox, enc_bboxes, enc_scores = \
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        # dec_bboxes：[6, bs, num_dn + nq, 4] dec_scores: [6, bs, num_dn + nq, nc]
        dec_bboxes, dec_scores = self.decoder(embed,
                                              refer_bbox,
                                              feats,
                                              shapes,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              attn_mask=attn_mask)
        # dec_bboxes：[6, bs, num_dn + nq, 4] dec_scores: [6, bs, num_dn + nq, nc] enc_bboxes: (bs, num_queries, 4) enc_scores: (bs, num_queries, nc)
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc) 删除第一维度
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        # 遍历了 shapes 中的每一个特征图形状 (h, w)，并使用 enumerate() 函数获取了索引 i 和形状信息 (h, w)
        for i, (h, w) in enumerate(shapes):
            # torch.arange() 函数生成了行和列的坐标 sy 和 sx，然后使用 torch.meshgrid() 创建了网格 grid_y 和 grid_x，这个网格表示了锚框可能出现的位置。
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2) 结果张量表示一个坐标网格，其中每个元素对应于一个锚框的中心点的 (x, y) 坐标。
            
            # 将网格坐标归一化到 [0, 1] 的范围内，这里加上了 0.5 是为了将坐标放在格子的中心
            valid_WH = torch.tensor([h, w], dtype=dtype, device=device)# (2)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)

            # 计算了锚框的宽度和高度，根据参数 grid_size 和当前迭代次数 i，锚框的尺寸按照指数级递增
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float('inf'))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        # self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch) 创建了三个1x1卷积，对通道维度进行调整到256
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)] #将三个输出维度的张量都拿出来进行投影，（投影也就是将通道维度都调整到256）
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c] 将[b, h*w, c]存入feats中
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2] 这里指出shapes的形状为[nl, 2] 
            shapes.append([h, w])

        # [b, h*w, c] 将三个nl输出的feats h*w合并 得到 [b, 20x20 + 40x40 + 80x80, c] 
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = len(feats)
        # Prepare input for decoder
        # anchors: (1, 20x20 + 40x40 + 80x80, 4) 其大小由特征输出尺度决定 valid_mask：(1, 20x20 + 40x40 + 80x80, 1) 用来判断anchor是否有效
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        
        # self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256 (1, 20x20 + 40x40 + 80x80, 1) * [b, 20x20 + 40x40 + 80x80, c（256）] 
        
        # self.enc_score_head = nn.Linear(hd, nc)
        enc_outputs_scores = self.enc_score_head(features)  # 经过一个线性层 bs, h*w, 256 -> (bs, h*w, nc) 至信度

        # Query selection
        # (bs, num_queries) 根据查询数获取前num_queries个enc_outputs_scores idx
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256) 利用张量三维索引得到topk_ind对应的特征
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1) 
        # (bs, num_queries, 4) 利用张量三维索引得到topk_ind对应的anchors
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        # self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)
        # 希望经过一个全连接层能得到bbox，将256转为4，（这里可以换成DFL） def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors #为什么直接把值加上去？好吧，无法理解的设计，自己设计了一个anchor再设计了一个bbox输出再加起来
        enc_bboxes = refer_bbox.sigmoid()

        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1) # (bs, num_queries + num_dn, 4)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1) # (bs, num_queries, nc)
        
        # if learnt_init_query: self.tgt_embed = nn.Embedding(nq, hd) （这里只是要一个维度为(nq, hd)的编码权重）
        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features # （bs, nq, hd）
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1) # [bs, num_dn + nq, dn_cls_embed = hd（256）] 

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init_` would cause NaN when training with custom datasets.
        # linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)

################### DualRTDETRDecoder ########################
class DualRTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """
    export = False  # export mode

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hd=256,  # hidden dim
            nq=300,  # num queries
            ndp=4,  # num decoder points
            nh=8,  # num head
            ndl=6,  # num decoder layers
            d_ffn=1024,  # dim of feedforward
            eval_idx=-1,
            dropout=0.,
            act=nn.ReLU(),
            # Training args
            nd=100,  # num denoising
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
            使用给定参数初始化 RTDETRDecoder 模块。

         参数：
             nc (int)：类数。 默认值为 80。
             ch（元组）：主干特征图中的通道。 默认值为 (512, 1024, 2048)。
             hd (int)：隐藏层的维度。 默认值为 256。
             nq (int)：查询点的数量。 默认值为 300。
             ndp (int)：解码器点数。 默认值为 4。
             nh (int)：多头注意力中的头数。 默认值为 8。
             ndl (int)：解码器层数。 默认值为 6。
             d_ffn (int)：前馈网络的维度。 默认值为 1024。
             dropout (float): 辍学率。 默认值为 0。
             act (nn.Module)：激活函数。 默认为 nn.ReLU。
             eval_idx(int)：评价指标。 默认值为-1。
             nd (int): 去噪次数。 默认值为 100。
             label_noise_ratio (float)：标签噪声比。 默认值为 0.5。
             box_noise_scale (float)：盒子噪声尺度。 默认值为 1.0。
             learnt_init_query (bool)：是否学习初始查询嵌入。 默认值为 False。
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc + 1, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd)) # 编码器就是一个线性层
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group
        #(Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Dict]]): 
        # 这里都改为两份
        # Input projection and embedding
        # feats:[b, (20x20 + 40x40 + 80x80) + (20x20 + 40x40 + 80x80), c] shapes: [nl, 2]
        feats1, shapes1, feats2, shapes2  = self._get_encoder_input(x)

        # Prepare denoising training 
        #The modified class embeddings, bounding boxes, attention mask and meta information for denoising. 
        # 三维张量[bs, num_dn, dn_cls_embed], [bs, num_dn, 4], [num_dn + num_queries, num_dn + num_queries], {'dn_pos_idx','dn_num_group': num_group,'dn_num_split': [num_dn, num_queries]}
        dn_embed, dn_bbox, attn_mask, dn_meta = \
            get_cdn_group(batch,
                          self.nc,
                          self.num_queries,
                          self.denoising_class_embed.weight,
                          self.num_denoising,
                          self.label_noise_ratio,
                          self.box_noise_scale,
                          self.training)
        # embed = [bs, num_dn + nq, dn_cls_embed = hd（256）] refer_bbox: (bs, num_queries + num_dn, 4) enc_bboxes: (bs, num_queries, 4) enc_scores: (bs, num_queries, nc)
        embed1, refer_bbox1, enc_bboxes1, enc_scores1 = \
            self._get_decoder_input(feats1, shapes1, dn_embed, dn_bbox)
        embed2, refer_bbox2, enc_bboxes2, enc_scores2 = \
            self._get_decoder_input(feats2, shapes2, dn_embed, dn_bbox)

        # Decoder
        # dec_bboxes：[6, bs, num_dn + nq, 4] dec_scores: [6, bs, num_dn + nq, nc]
        dec_bboxes1, dec_scores1 = self.decoder(embed1,
                                              refer_bbox1,
                                              feats1,
                                              shapes1,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              attn_mask=attn_mask)
        dec_bboxes2, dec_scores2 = self.decoder(embed2,
                                              refer_bbox2,
                                              feats2,
                                              shapes2,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              attn_mask=attn_mask)
        # dec_bboxes：[6, bs, num_dn + nq, 4] dec_scores: [6, bs, num_dn + nq, nc] enc_bboxes: (bs, num_queries, 4) enc_scores: (bs, num_queries, nc)
        x1 = dec_bboxes1, dec_scores1, enc_bboxes1, enc_scores1, dn_meta
        x2 = dec_bboxes2, dec_scores2, enc_bboxes2, enc_scores2, dn_meta
        if self.training: 
            return [x1, x2]
        # (bs, 300, 4+nc) 删除第一维度
        y = [torch.cat((dec_bboxes1.squeeze(0), dec_scores1.squeeze(0).sigmoid()), -1), torch.cat((dec_bboxes2.squeeze(0), dec_scores2.squeeze(0).sigmoid()), -1)]
        return y if self.export else (y, [x1, x2])

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        # 遍历了 shapes 中的每一个特征图形状 (h, w)，并使用 enumerate() 函数获取了索引 i 和形状信息 (h, w)
        for i, (h, w) in enumerate(shapes):
            # torch.arange() 函数生成了行和列的坐标 sy 和 sx，然后使用 torch.meshgrid() 创建了网格 grid_y 和 grid_x，这个网格表示了锚框可能出现的位置。
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2) 结果张量表示一个坐标网格，其中每个元素对应于一个锚框的中心点的 (x, y) 坐标。
            
            # 将网格坐标归一化到 [0, 1] 的范围内，这里加上了 0.5 是为了将坐标放在格子的中心
            valid_WH = torch.tensor([h, w], dtype=dtype, device=device)# (2)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)

            # 计算了锚框的宽度和高度，根据参数 grid_size 和当前迭代次数 i，锚框的尺寸按照指数级递增
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float('inf'))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        # self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch) 创建了三个1x1卷积，对通道维度进行调整到256
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)] #将三个输出维度的张量都拿出来进行投影，（投影也就是将通道维度都调整到256）
        # Get encoder inputs
        feats1 = []
        shapes1 = []
        feats2 = []
        shapes2 = []
        for i, feat in enumerate(x):
            h, w = feat.shape[2:]

            # 分组处理特征
            if i < len(x) // 2:  # 前三个特征为一组，后三个特征为另一组
                feats1.append(feat.flatten(2).permute(0, 2, 1))
                shapes1.append([h, w])
            else:
                feats2.append(feat.flatten(2).permute(0, 2, 1))
                shapes2.append([h, w])

        # 合并分组后的特征
        feats1 = torch.cat(feats1, 1) if feats1 else None
        feats2 = torch.cat(feats2, 1) if feats2 else None

        return feats1, shapes1, feats2, shapes2


    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = len(feats)
        # Prepare input for decoder
        # anchors: (1, 20x20 + 40x40 + 80x80, 4) 其大小由特征输出尺度决定 valid_mask：(1, 20x20 + 40x40 + 80x80, 1) 用来判断anchor是否有效
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        
        # self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd)) 将一个线性层作为编码器，作为中间层
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256 (1, 20x20 + 40x40 + 80x80, 1) * [b, 20x20 + 40x40 + 80x80, c（256）] 
        
        # self.enc_score_head = nn.Linear(hd, nc) 特征经过两个线性层输出encoder置信度分数
        enc_outputs_scores = self.enc_score_head(features)  # 经过一个线性层 bs, h*w, 256 -> (bs, h*w, nc) 至信度

        # Query selection
        # (bs, num_queries) 根据查询数获取前num_queries个enc_outputs_scores idx 
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256) 利用张量三维索引得到topk_ind对应的特征 也就是根据编码器的置信度结果选择最好的特征和对应的anchor
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1) 
        # (bs, num_queries, 4) 利用张量三维索引得到topk_ind对应的anchors 也就是根据编码器的置信度结果选择最好的特征和对应的anchor
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content 设计思想：我觉得是因为作者认为模型学习的内容是相对于anchor的偏移量，所以预测结果由anchor + self.enc_bbox_head
        # self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)
        # 希望经过一个全连接层能得到bbox，将256转为4，（这里可以换成DFL） def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors #将根据置信度最优的前300个特征，经过编码器头MLP用于生成模型的预测框
        enc_bboxes = refer_bbox.sigmoid() # 经过sigmoid函数控制，得到最后的编码器预测框

        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1) # (bs, num_queries + num_dn, 4)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1) # (bs, num_queries, nc)
        
        # if learnt_init_query: self.tgt_embed = nn.Embedding(nq, hd) （这里只是要一个维度为(nq, hd)的固定编码权重）
        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features # （bs, nq, hd）
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1) # [bs, num_dn + nq, dn_cls_embed = hd（256）] 

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init_` would cause NaN when training with custom datasets.
        # linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)
################### end of DualRTDETRDecoder #####################
            
#################### ManbaDecoder ############################
class ManbaDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """
    export = False  # export mode

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hd=256,  # hidden dim
            nq=300,  # num queries
            ndp=4,  # num decoder points
            nh=8,  # num head
            ndl=6,  # num decoder layers
            d_ffn=1024,  # dim of feedforward
            eval_idx=-1,
            dropout=0.,
            act=nn.ReLU(),
            # Training args
            nd=100,  # num denoising
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False,
            # ========================
            dims=[128, 256, 512], 
            drop_path=[0.1, 0.1, 0.1],
            # =========================
            ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
            使用给定参数初始化 RTDETRDecoder 模块。

         参数：
             nc (int)：类数。 默认值为 80。
             ch（元组）：主干特征图中的通道。 默认值为 (512, 1024, 2048)。
             hd (int)：隐藏层的维度。 默认值为 256。
             nq (int)：查询点的数量。 默认值为 300。
             ndp (int)：解码器点数。 默认值为 4。
             nh (int)：多头注意力中的头数。 默认值为 8。
             ndl (int)：解码器层数。 默认值为 6。
             d_ffn (int)：前馈网络的维度。 默认值为 1024。
             dropout (float): 辍学率。 默认值为 0。
             act (nn.Module)：激活函数。 默认为 nn.ReLU。
             eval_idx(int)：评价指标。 默认值为-1。
             nd (int): 去噪次数。 默认值为 100。
             label_noise_ratio (float)：标签噪声比。 默认值为 0.5。
             box_noise_scale (float)：盒子噪声尺度。 默认值为 1.0。
             learnt_init_query (bool)：是否学习初始查询嵌入。 默认值为 False。
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)
        
        # Manba VSSBlocks
        self.VSSBlocks = nn.ModuleList()
        self.num_Blocks = len(dims)
        for i_Block in range(self.num_Blocks):
            self.VSSBlocks.append(VSSBlock(
                hidden_dim=dims[i_Block],
                drop_path=drop_path[i_Block],
            ))

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc + 1, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        #self.enc_output = VSSBlock(hidden_dim=hd, drop_path=0.1)
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group
        #(Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Dict]]): 

        x = [self.VSSBlocks[i](feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) for i, feat in enumerate(x)]

        # Input projection and embedding
        # feats:[b, 20x20 + 40x40 + 80x80, 256] shapes: [nl, 2]
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training 
        #The modified class embeddings, bounding boxes, attention mask and meta information for denoising. 
        # 三维张量[bs, num_dn, dn_cls_embed], [bs, num_dn, 4], [num_dn + num_queries, num_dn + num_queries], {'dn_pos_idx','dn_num_group': num_group,'dn_num_split': [num_dn, num_queries]}
        dn_embed, dn_bbox, attn_mask, dn_meta = \
            get_cdn_group(batch,
                          self.nc,
                          self.num_queries,
                          self.denoising_class_embed.weight,
                          self.num_denoising,
                          self.label_noise_ratio,
                          self.box_noise_scale,
                          self.training)
        # embed = [bs, num_dn + nq, dn_cls_embed = hd（256）] refer_bbox: (bs, num_queries + num_dn, 4) enc_bboxes: (bs, num_queries, 4) enc_scores: (bs, num_queries, nc)
        embed, refer_bbox, enc_bboxes, enc_scores = \
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        # dec_bboxes：[6, bs, num_dn + nq, 4] dec_scores: [6, bs, num_dn + nq, nc]
        dec_bboxes, dec_scores = self.decoder(embed,
                                              refer_bbox,
                                              feats,
                                              shapes,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              attn_mask=attn_mask)
        # dec_bboxes：[6, bs, num_dn + nq, 4] dec_scores: [6, bs, num_dn + nq, nc] enc_bboxes: (bs, num_queries, 4) enc_scores: (bs, num_queries, nc)
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc) 删除第一维度
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        # 遍历了 shapes 中的每一个特征图形状 (h, w)，并使用 enumerate() 函数获取了索引 i 和形状信息 (h, w)
        for i, (h, w) in enumerate(shapes):
            # torch.arange() 函数生成了行和列的坐标 sy 和 sx，然后使用 torch.meshgrid() 创建了网格 grid_y 和 grid_x，这个网格表示了锚框可能出现的位置。
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2) 结果张量表示一个坐标网格，其中每个元素对应于一个锚框的中心点的 (x, y) 坐标。
            
            # 将网格坐标归一化到 [0, 1] 的范围内，这里加上了 0.5 是为了将坐标放在格子的中心
            valid_WH = torch.tensor([h, w], dtype=dtype, device=device)# (2)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)

            # 计算了锚框的宽度和高度，根据参数 grid_size 和当前迭代次数 i，锚框的尺寸按照指数级递增
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float('inf'))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        # self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch) 创建了三个1x1卷积，对通道维度进行调整到256
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)] #将三个输出维度的张量都拿出来进行投影，（投影也就是将通道维度都调整到256）
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c] 将[b, h*w, c]存入feats中
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2] 这里指出shapes的形状为[nl, 2] 
            shapes.append([h, w])

        # [b, h*w, c] 将三个nl输出的feats h*w合并 得到 [b, 20x20 + 40x40 + 80x80, c] 
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = len(feats)
        # Prepare input for decoder
        # anchors: (1, 20x20 + 40x40 + 80x80, 4) 其大小由特征输出尺度决定 valid_mask：(1, 20x20 + 40x40 + 80x80, 1) 用来判断anchor是否有效
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        
        # self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd)) 改为manba
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256 (1, 20x20 + 40x40 + 80x80, 1) * [b, 20x20 + 40x40 + 80x80, c（256）] 
        
        # self.enc_score_head = nn.Linear(hd, nc)
        enc_outputs_scores = self.enc_score_head(features)  # 经过一个线性层 bs, h*w, 256 -> (bs, h*w, nc) 至信度

        # Query selection
        # (bs, num_queries) 根据查询数获取前num_queries个enc_outputs_scores idx
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256) 利用张量三维索引得到topk_ind对应的特征
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1) 
        # (bs, num_queries, 4) 利用张量三维索引得到topk_ind对应的anchors
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        # self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)
        # 希望经过一个全连接层能得到bbox，将256转为4，（这里可以换成DFL） def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors #为什么直接把值加上去？好吧，无法理解的设计，自己设计了一个anchor再设计了一个bbox输出再加起来
        enc_bboxes = refer_bbox.sigmoid()

        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1) # (bs, num_queries + num_dn, 4)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1) # (bs, num_queries, nc)
        
        # if learnt_init_query: self.tgt_embed = nn.Embedding(nq, hd) （这里只是要一个维度为(nq, hd)的编码权重）
        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features # （bs, nq, hd）
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1) # [bs, num_dn + nq, dn_cls_embed = hd（256）] 

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init_` would cause NaN when training with custom datasets.
        # linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)
#################### end of ManbaDecoder ############################

################### ManbaWorldDecoder ########################
class ManbaWorldDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """
    export = False  # export mode

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hd=512,  # hidden dim
            nq=300,  # num queries
            ndp=4,  # num decoder points
            nh=8,  # num head
            ndl=6,  # num decoder layers
            d_ffn=1024,  # dim of feedforward
            eval_idx=-1,
            dropout=0.,
            act=nn.ReLU(),
            # Training args
            nd=100,  # num denoising
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False,
            # ========================
            dims=[128, 256, 512], 
            drop_path=[0.1, 0.1, 0.1],
            # =========================
            embed=512,
            with_bn=False
            ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
            使用给定参数初始化 RTDETRDecoder 模块。

         参数：
             nc (int)：类数。 默认值为 80。
             ch（元组）：主干特征图中的通道。 默认值为 (512, 1024, 2048)。
             hd (int)：隐藏层的维度。 默认值为 256。
             nq (int)：查询点的数量。 默认值为 300。
             ndp (int)：解码器点数。 默认值为 4。
             nh (int)：多头注意力中的头数。 默认值为 8。
             ndl (int)：解码器层数。 默认值为 6。
             d_ffn (int)：前馈网络的维度。 默认值为 1024。
             dropout (float): 辍学率。 默认值为 0。
             act (nn.Module)：激活函数。 默认为 nn.ReLU。
             eval_idx(int)：评价指标。 默认值为-1。
             nd (int): 去噪次数。 默认值为 100。
             label_noise_ratio (float)：标签噪声比。 默认值为 0.5。
             box_noise_scale (float)：盒子噪声尺度。 默认值为 1.0。
             learnt_init_query (bool)：是否学习初始查询嵌入。 默认值为 False。
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)
        
        # Manba VSSBlocks
        self.VSSBlocks = nn.ModuleList()
        self.num_Blocks = len(dims)
        for i_Block in range(self.num_Blocks):
            self.VSSBlocks.append(VSSBlock(
                hidden_dim=dims[i_Block],
                drop_path=drop_path[i_Block],
            ))

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = TextDeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc + 1, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        #self.enc_output = VSSBlock(hidden_dim=hd, drop_path=0.1)
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        #self.enc_score_head = BNContrastiveHeadMLP(embed) if with_bn else ContrastiveHeadMLP() 
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([BNContrastiveHeadMLP(embed) if with_bn else ContrastiveHeadMLP() for _ in range(ndl)])
        #self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, text, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group
        #(Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Dict]]): 
        x = [self.VSSBlocks[i](feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) for i, feat in enumerate(x)]
        
        # Input projection and embedding
        # feats:[b, 20x20 + 40x40 + 80x80, 256] shapes: [nl, 2] cls:[b, 20x20 + 40x40 + 80x80, nc] 
        # 将分类和定位解耦开来，定位用trasnfromer 分类用CNN
        feats, shapes= self._get_encoder_input(x)

        # Prepare denoising training 
        #The modified class embeddings, bounding boxes, attention mask and meta information for denoising. 
        # 三维张量[bs, num_dn, dn_cls_embed], [bs, num_dn, 4], [num_dn + num_queries, num_dn + num_queries], {'dn_pos_idx','dn_num_group': num_group,'dn_num_split': [num_dn, num_queries]}
        # 创建用于 denoising（去噪）过程的查询组 将decoder的输出进行评估分为正样本和负样本后，负样本作为带噪声再次训练 也可以选择已经训练好的模型作为教师产生负样本一起训练
        dn_embed, dn_bbox, attn_mask, dn_meta = \
            get_cdn_group(batch,
                          self.nc,
                          self.num_queries,
                          self.denoising_class_embed.weight,
                          self.num_denoising,
                          self.label_noise_ratio,
                          self.box_noise_scale,
                          self.training)
        # embed = [bs, num_dn + nq, dn_cls_embed = hd（256）] refer_bbox: (bs, num_queries + num_dn, 4) enc_bboxes: (bs, num_queries, 4) enc_scores: (bs, num_queries, nc)
        embed, refer_bbox, enc_bboxes, enc_scores = \
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder 输入真实目标和噪声查询的组合
        # dec_bboxes：[6, bs, num_dn + nq, 4] dec_scores: [6, bs, num_dn + nq, nc]
        dec_bboxes, dec_scores = self.decoder(embed,
                                              refer_bbox,
                                              feats,
                                              shapes,
                                              text,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              attn_mask=attn_mask)
        # dec_bboxes：[6, bs, num_dn + nq, 4] dec_scores: [6, bs, num_dn + nq, nc] enc_bboxes: (bs, num_queries, 4) enc_scores: (bs, num_queries, nc)
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc) 删除第一维度 训练时用enc 推理时用dec
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        # 遍历了 shapes 中的每一个特征图形状 (h, w)，并使用 enumerate() 函数获取了索引 i 和形状信息 (h, w)
        for i, (h, w) in enumerate(shapes):
            # torch.arange() 函数生成了行和列的坐标 sy 和 sx，然后使用 torch.meshgrid() 创建了网格 grid_y 和 grid_x，这个网格表示了锚框可能出现的位置。
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2) 结果张量表示一个坐标网格，其中每个元素对应于一个锚框的中心点的 (x, y) 坐标。
            
            # 将网格坐标归一化到 [0, 1] 的范围内，这里加上了 0.5 是为了将坐标放在格子的中心
            valid_WH = torch.tensor([h, w], dtype=dtype, device=device)# (2)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)

            # 计算了锚框的宽度和高度，根据参数 grid_size 和当前迭代次数 i，锚框的尺寸按照指数级递增
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float('inf'))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        # self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch) 创建了三个1x1卷积，对通道维度进行调整到256
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)] #将三个输出维度的张量都拿出来进行投影，（投影也就是将通道维度都调整到256）
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c] 将[b, h*w, c]存入feats中
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2] 这里指出shapes的形状为[nl, 2] 
            shapes.append([h, w])

        # [b, h*w, c] 将三个nl输出的feats h*w合并 得到 [b, 20x20 + 40x40 + 80x80, c] 
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None, text=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = len(feats)
        # Prepare input for decoder
        # anchors: (1, 20x20 + 40x40 + 80x80, 4) 其大小由特征输出尺度决定 valid_mask：(1, 20x20 + 40x40 + 80x80, 1) 用来判断anchor是否有效
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        
        # self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd)) 改为manba
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256 (1, 20x20 + 40x40 + 80x80, 1) * [b, 20x20 + 40x40 + 80x80, c（256）] 
        
        # self.enc_score_head = nn.Linear(hd, nc)
        # self.enc_score_head = BNContrastiveHead(embed) if with_bn else ContrastiveHead()
        enc_outputs_scores = self.enc_score_head(features) # 经过一个线性层 bs, h*w, 256 -> (bs, h*w, nc) 至信度

        # Query selection
        # (bs, num_queries) 根据查询数获取前num_queries个enc_outputs_scores idx
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256) 利用张量三维索引得到topk_ind对应的特征
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1) 
        # (bs, num_queries, 4) 利用张量三维索引得到topk_ind对应的anchors
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        # self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)
        # 希望经过一个全连接层能得到bbox，将256转为4，（这里可以换成DFL） def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors #为什么直接把值加上去？(傻子，因为模型输出偏移量)好吧，无法理解的设计，自己设计了一个anchor再设计了一个bbox输出再加起来
        enc_bboxes = refer_bbox.sigmoid()

        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1) # (bs, num_queries + num_dn, 4)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1) # (bs, num_queries, nc)
        
        # if learnt_init_query: self.tgt_embed = nn.Embedding(nq, hd) （这里只是要一个维度为(nq, hd)的编码权重）这个不重要，实际是top_k_features
        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features # （bs, nq, hd）
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1) # [bs, num_dn + nq, dn_cls_embed = hd（256）] 

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init_` would cause NaN when training with custom datasets.
        # linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for reg_ in self.dec_bbox_head:
            # linear_init_(cls_)
            #constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)
################### end of ManbaWorldDecoder #################


################### ManbaWorldDecoder ########################
class locationManbaDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """
    export = False  # export mode

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hd=512,  # hidden dim
            nq=300,  # num queries
            ndp=4,  # num decoder points
            nh=8,  # num head
            ndl=6,  # num decoder layers
            d_ffn=1024,  # dim of feedforward
            eval_idx=-1,
            dropout=0.,
            act=nn.ReLU(),
            # Training args
            nd=100,  # num denoising
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False,
            # ========================
            dims=[128, 256, 512], 
            drop_path=[0.1, 0.1, 0.1],
            # =========================
            embed=512,
            with_bn=False
            ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
            使用给定参数初始化 RTDETRDecoder 模块。

         参数：
             nc (int)：类数。 默认值为 80。
             ch（元组）：主干特征图中的通道。 默认值为 (512, 1024, 2048)。
             hd (int)：隐藏层的维度。 默认值为 256。
             nq (int)：查询点的数量。 默认值为 300。
             ndp (int)：解码器点数。 默认值为 4。
             nh (int)：多头注意力中的头数。 默认值为 8。
             ndl (int)：解码器层数。 默认值为 6。
             d_ffn (int)：前馈网络的维度。 默认值为 1024。
             dropout (float): 辍学率。 默认值为 0。
             act (nn.Module)：激活函数。 默认为 nn.ReLU。
             eval_idx(int)：评价指标。 默认值为-1。
             nd (int): 去噪次数。 默认值为 100。
             label_noise_ratio (float)：标签噪声比。 默认值为 0.5。
             box_noise_scale (float)：盒子噪声尺度。 默认值为 1.0。
             learnt_init_query (bool)：是否学习初始查询嵌入。 默认值为 False。
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Convolutional layer classification head
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(hd, c3, 1), Conv(c3, c3, 1), nn.Conv2d(c3, embed, 1)) for _ in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

        # enbeding
        #self.embeding = MLP(4, 2 * hd, hd, num_layers=2)

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)
        
        # Manba VSSBlocks
        self.VSSBlocks = nn.ModuleList()
        self.num_Blocks = len(dims)
        for i_Block in range(self.num_Blocks):
            self.VSSBlocks.append(VSSBlock(
                hidden_dim=dims[i_Block],
                drop_path=drop_path[i_Block],
            ))

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = locationDeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        # self.denoising_class_embed = nn.Embedding(nc + 1, nc)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        #self.enc_output = VSSBlock(hidden_dim=hd, drop_path=0.1)
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.query_selector = nn.Linear(hd, 1)
        #self.enc_score_head = BNContrastiveHeadMLP(embed) if with_bn else ContrastiveHeadMLP() 
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        #self.dec_score_head = nn.ModuleList([BNContrastiveHeadMLP(embed) if with_bn else ContrastiveHeadMLP() for _ in range(ndl)])
        #self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, text, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group_withoutcls
        # Convolutional layer classification head
        # self.cls = [b, nc, h, w]
        # cls = []
        # for i in range(self.nl):
        #     cls.append(self.cv4[i](self.cv3[i](x[i]), text))
        # # pred_scores [b, nc, 20x20 + 40x40 + 80x80]
        # pred_scores = torch.cat([xi.view(cls[0].shape[0], self.nc, -1) for xi in cls], 2)
        # # 根据分类分数总和结果评估q数量
        # pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        # # [b, 20x20 + 40x40 + 80x80, nc]
        # # 得到前q个匹配
        # #topk_ind = torch.topk(pred_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # #batch_ind = torch.arange(end=len(pred_scores), dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)
        # #dec_scores = pred_scores[batch_ind, topk_ind].view(len(pred_scores), self.num_queries, -1) # (bs, num_queries, nc)
        # #dec_scores, _ = torch.topk(pred_scores, self.num_queries, dim=1)


        #(Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Dict]]): 
        x = [self.VSSBlocks[i](feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) for i, feat in enumerate(x)]
        # [b, 512, 20, 20],[b, 512, 40, 40],[b, 512, 80, 80]
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)] #将三个输出维度的张量都拿出来进行投影，（投影也就是将通道维度都调整到256）
        cls = []
        for i in range(self.nl):
            # [b, nc, 20, 20],[b, nc, 40, 40],[b, nc, 80, 80]
            cls.append(self.cv4[i](self.cv3[i](x[i]), text))
        # pred_scores [b, 20x20 + 40x40 + 80x80, nc]
        pred_scores = torch.cat([xi.flatten(2).permute(0, 2, 1) for xi in cls], 1)
        
        # 根据分类分数总和结果评估q数量
        #pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        
        # Input projection and embedding
        # feats:[b, 20x20 + 40x40 + 80x80, 256] shapes: [nl, 2] cls:[b, 20x20 + 40x40 + 80x80, nc] 
        # 将分类和定位解耦开来，定位用trasnfromer 分类用CNN
        feats, shapes= self._get_encoder_input(x)

        # Prepare denoising training 
        #The modified class embeddings, bounding boxes, attention mask and meta information for denoising. 
        # 三维张量[bs, num_dn, nc], [bs, num_dn, 4], [num_dn + num_queries, num_dn + num_queries], {'dn_pos_idx','dn_num_group': num_group,'dn_num_split': [num_dn, num_queries]}
        # 创建用于 denoising（去噪）过程的查询组 将decoder的输出进行评估分为正样本和负样本后，负样本作为带噪声再次训练 也可以选择已经训练好的模型作为教师产生负样本一起训练
        dn_bbox, attn_mask, dn_meta = \
            get_cdn_group_withoutcls(batch,
                          self.num_queries,
                          self.num_denoising,
                          self.box_noise_scale,
                          self.training)
        # embed = [bs, num_dn + nq, dn_cls_embed = hd（256）] refer_bbox: (bs, num_queries + num_dn, 4) enc_bboxes: (bs, num_queries, 4) enc_scores: (bs, num_queries, nc)

        dn_embed = None
        embed, refer_bbox, enc_bboxes, dec_scores = \
            self._get_decoder_input(feats, shapes, dn_bbox, dn_embed, pred_scores)

        # Decoder 输入真实目标和噪声查询的组合
        # dec_bboxes：[6, bs, num_dn + nq, 4] dec_scores: [6, bs, num_dn + nq, nc]
        if self.training and self.num_denoising > 0:
            num_dn = refer_bbox.shape[1]-self.num_queries
            padded_embed = torch.zeros((embed.shape[0], num_dn, embed.shape[2]), dtype=embed.dtype, device=embed.device)
            embed = torch.cat([padded_embed, embed], 1)
        dec_bboxes = self.decoder(embed,
                                refer_bbox,
                                feats,
                                shapes,
                                self.dec_bbox_head,
                                self.query_pos_head,
                                attn_mask=attn_mask)
        # dec_bboxes：[6, bs, num_dn + nq, 4] dec_scores: [6, bs, num_dn + nq, nc] enc_bboxes: (bs, num_queries, 4) enc_scores: (bs, num_queries, nc)
        x = dec_bboxes, dec_scores, enc_bboxes, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc) 删除第一维度 训练时用enc 推理时用dec
        # 这里还得改预测的流程
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        # 遍历了 shapes 中的每一个特征图形状 (h, w)，并使用 enumerate() 函数获取了索引 i 和形状信息 (h, w)
        for i, (h, w) in enumerate(shapes):
            # torch.arange() 函数生成了行和列的坐标 sy 和 sx，然后使用 torch.meshgrid() 创建了网格 grid_y 和 grid_x，这个网格表示了锚框可能出现的位置。
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2) 结果张量表示一个坐标网格，其中每个元素对应于一个锚框的中心点的 (x, y) 坐标。
            
            # 将网格坐标归一化到 [0, 1] 的范围内，这里加上了 0.5 是为了将坐标放在格子的中心
            valid_WH = torch.tensor([h, w], dtype=dtype, device=device)# (2)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)

            # 计算了锚框的宽度和高度，根据参数 grid_size 和当前迭代次数 i，锚框的尺寸按照指数级递增
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float('inf'))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        # self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch) 创建了三个1x1卷积，对通道维度进行调整到256
        #x = [self.input_proj[i](feat) for i, feat in enumerate(x)] #将三个输出维度的张量都拿出来进行投影，（投影也就是将通道维度都调整到256）
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c] 将[b, h*w, c]存入feats中
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2] 这里指出shapes的形状为[nl, 2] 
            shapes.append([h, w])

        # [b, h*w, c] 将三个nl输出的feats h*w合并 得到 [b, 20x20 + 40x40 + 80x80, c] 
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_bbox=None, dn_embed=None, pred_scores=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = len(feats)
        # Prepare input for decoder
        # anchors: (1, 20x20 + 40x40 + 80x80, 4) 其大小由特征输出尺度决定 valid_mask：(1, 20x20 + 40x40 + 80x80, 1) 用来判断anchor是否有效
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        
        # self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd)) 改为manba
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256 (1, 20x20 + 40x40 + 80x80, 1) * [b, 20x20 + 40x40 + 80x80, c（256）] 
        
        # self.enc_score_head = nn.Linear(hd, nc)
        # self.enc_score_head = BNContrastiveHead(embed) if with_bn else ContrastiveHead()
        # 用一个FFN？ Query Selector
        # query_selector = self.query_selector(features) # 经过一个线性层 bs, h*w, 256 -> (bs, h*w, nc) 至信度

        # Query selection
        # (bs, num_queries) 根据查询数获取前num_queries个enc_outputs_scores idx
        topk_ind = torch.topk(pred_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256) 利用张量三维索引得到topk_ind对应的特征
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1) 
        # (bs, num_queries, 4) 利用张量三维索引得到topk_ind对应的anchors
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        # self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)
        # 希望经过一个全连接层能得到bbox，将256转为4，（这里可以换成DFL） def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors #为什么直接把值加上去？(傻子，因为模型输出偏移量)好吧，无法理解的设计，自己设计了一个anchor再设计了一个bbox输出再加起来
        enc_bboxes = refer_bbox.sigmoid()

        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1) # (bs, num_queries + num_dn, 4)
        dec_scores = pred_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1) # (bs, num_queries, nc)
        
        # if learnt_init_query: self.tgt_embed = nn.Embedding(nq, hd) （这里只是要一个维度为(nq, hd)的编码权重）这个不重要，实际是top_k_features
        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features # （bs, nq, hd）
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1) # [bs, num_dn + nq, dn_cls_embed = hd（256）] 

        return embeddings, refer_bbox, enc_bboxes, dec_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        #bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init_` would cause NaN when training with custom datasets.
        # linear_init_(self.enc_score_head)
        #constant_(self.query_selector.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        #constant_(self.query_selector.layers[-1].weight, 0.)
        #constant_(self.query_selector.layers[-1].bias, 0.)
        for reg_ in self.dec_bbox_head:
            # linear_init_(cls_)
            #constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        #linear_init_(self.query_selector)
        #xavier_uniform_(self.query_selector.weight)
        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        #xavier_uniform_(self.embeding.layers[0].weight)
        #xavier_uniform_(self.embeding.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)
################### end of ManbaWorldDecoder #################


################### DecouplingManbaWorldDecoder ########################
class DecouplingManbaWorldDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """
    export = False  # export mode

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hd=512,  # hidden dim
            nq=300,  # num queries
            ndp=4,  # num decoder points
            nh=8,  # num head
            ndl=6,  # num decoder layers
            d_ffn=1024,  # dim of feedforward
            eval_idx=-1,
            dropout=0.,
            act=nn.ReLU(),
            # Training args
            nd=100,  # num denoising
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False,
            # ========================
            dims=[128, 256, 512], 
            drop_path=[0.1, 0.1, 0.1],
            # =========================
            embed=512,
            with_bn=False
            ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
            使用给定参数初始化 RTDETRDecoder 模块。

         参数：
             nc (int)：类数。 默认值为 80。
             ch（元组）：主干特征图中的通道。 默认值为 (512, 1024, 2048)。
             hd (int)：隐藏层的维度。 默认值为 256。
             nq (int)：查询点的数量。 默认值为 300。
             ndp (int)：解码器点数。 默认值为 4。
             nh (int)：多头注意力中的头数。 默认值为 8。
             ndl (int)：解码器层数。 默认值为 6。
             d_ffn (int)：前馈网络的维度。 默认值为 1024。
             dropout (float): 辍学率。 默认值为 0。
             act (nn.Module)：激活函数。 默认为 nn.ReLU。
             eval_idx(int)：评价指标。 默认值为-1。
             nd (int): 去噪次数。 默认值为 100。
             label_noise_ratio (float)：标签噪声比。 默认值为 0.5。
             box_noise_scale (float)：盒子噪声尺度。 默认值为 1.0。
             learnt_init_query (bool)：是否学习初始查询嵌入。 默认值为 False。
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)
        
        # Manba VSSBlocks
        self.VSSBlocks = nn.ModuleList()
        self.num_Blocks = len(dims)
        for i_Block in range(self.num_Blocks):
            self.VSSBlocks.append(VSSBlock(
                hidden_dim=dims[i_Block],
                drop_path=drop_path[i_Block],
            ))

        # Transformer module
        decoder_layer = DecouplingDeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DecouplingTextDeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc + 1, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        #self.enc_output = VSSBlock(hidden_dim=hd, drop_path=0.1)
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        #self.enc_score_head = BNContrastiveHeadMLP(embed) if with_bn else ContrastiveHeadMLP() 
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([BNContrastiveHeadMLP(embed) if with_bn else ContrastiveHeadMLP() for _ in range(ndl)])
        #self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        # # offsets
        # self.num_points = 4
        # self.offsets = nn.Linear(hd, len(ch)*2*4)

        self._reset_parameters()

    def forward(self, x, text, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group
        #(Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Dict]]): 
        x = [self.VSSBlocks[i](feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) for i, feat in enumerate(x)]
        
        # Input projection and embedding
        # feats:[b, 20x20 + 40x40 + 80x80, 256] shapes: [nl, 2] cls:[b, 20x20 + 40x40 + 80x80, nc] 
        # 将分类和定位解耦开来，定位用trasnfromer 分类用CNN
        feats, shapes= self._get_encoder_input(x)

        # Prepare denoising training 
        #The modified class embeddings, bounding boxes, attention mask and meta information for denoising. 
        # 三维张量[bs, num_dn, dn_cls_embed], [bs, num_dn, 4], [num_dn + num_queries, num_dn + num_queries], {'dn_pos_idx','dn_num_group': num_group,'dn_num_split': [num_dn, num_queries]}
        # 创建用于 denoising（去噪）过程的查询组 将decoder的输出进行评估分为正样本和负样本后，负样本作为带噪声再次训练 也可以选择已经训练好的模型作为教师产生负样本一起训练
        dn_embed, dn_bbox, attn_mask, dn_meta = \
            get_cdn_group(batch,
                          self.nc,
                          self.num_queries,
                          self.denoising_class_embed.weight,
                          self.num_denoising,
                          self.label_noise_ratio,
                          self.box_noise_scale,
                          self.training)
        # embed = [bs, num_dn + nq, dn_cls_embed = hd（256）] refer_bbox: (bs, num_queries + num_dn, 4) enc_bboxes: (bs, num_queries, 4) enc_scores: (bs, num_queries, nc)
        embed, refer_bbox, enc_bboxes, enc_scores = \
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder 输入真实目标和噪声查询的组合
        # dec_bboxes：[6, bs, num_dn + nq, 4] dec_scores: [6, bs, num_dn + nq, nc]
        dec_bboxes, dec_scores = self.decoder(embed,
                                              refer_bbox,
                                              feats,
                                              shapes,
                                              text,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              dn_meta,
                                              attn_mask=attn_mask)
        # dec_bboxes：[6, bs, num_dn + nq, 4] dec_scores: [6, bs, num_dn + nq, nc] enc_bboxes: (bs, num_queries, 4) enc_scores: (bs, num_queries, nc)
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc) 删除第一维度 训练时用enc 推理时用dec
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        # 遍历了 shapes 中的每一个特征图形状 (h, w)，并使用 enumerate() 函数获取了索引 i 和形状信息 (h, w)
        for i, (h, w) in enumerate(shapes):
            # torch.arange() 函数生成了行和列的坐标 sy 和 sx，然后使用 torch.meshgrid() 创建了网格 grid_y 和 grid_x，这个网格表示了锚框可能出现的位置。
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2) 结果张量表示一个坐标网格，其中每个元素对应于一个锚框的中心点的 (x, y) 坐标。
            
            # 将网格坐标归一化到 [0, 1] 的范围内，这里加上了 0.5 是为了将坐标放在格子的中心
            valid_WH = torch.tensor([h, w], dtype=dtype, device=device)# (2)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)

            # 计算了锚框的宽度和高度，根据参数 grid_size 和当前迭代次数 i，锚框的尺寸按照指数级递增
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float('inf'))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        # self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch) 创建了三个1x1卷积，对通道维度进行调整到256
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)] #将三个输出维度的张量都拿出来进行投影，（投影也就是将通道维度都调整到256）
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c] 将[b, h*w, c]存入feats中
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2] 这里指出shapes的形状为[nl, 2] 
            shapes.append([h, w])

        # [b, h*w, c] 将三个nl输出的feats h*w合并 得到 [b, 20x20 + 40x40 + 80x80, c] 
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None, text=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = len(feats)
        #hd = feats.shape[-1]
        # Prepare input for decoder
        # anchors: (1, 20x20 + 40x40 + 80x80, 4) 其大小由特征输出尺度决定 valid_mask：(1, 20x20 + 40x40 + 80x80, 1) 用来判断anchor是否有效
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        
        # self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd)) 改为manba
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256 (1, 20x20 + 40x40 + 80x80, 1) * [b, 20x20 + 40x40 + 80x80, c（256）] 
        
        # self.enc_score_head = nn.Linear(hd, nc)
        # self.enc_score_head = BNContrastiveHead(embed) if with_bn else ContrastiveHead()
        enc_outputs_scores = self.enc_score_head(features) # 经过一个线性层 bs, h*w, 256 -> (bs, h*w, nc) 至信度

        # Query selection
        # (bs, num_queries) 根据查询数获取前num_queries个enc_outputs_scores idx
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256) 利用张量三维索引得到topk_ind对应的特征
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1) 
        # (bs, num_queries, 4) 利用张量三维索引得到topk_ind对应的anchors
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        # self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)
        # 希望经过一个全连接层能得到bbox，将256转为4，（这里可以换成DFL） def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors #为什么直接把值加上去？(傻子，因为模型输出偏移量)好吧，无法理解的设计，自己设计了一个anchor再设计了一个bbox输出再加起来
        enc_bboxes = refer_bbox.sigmoid()

        # # 可变形特征采样
        # sampling_offsets = self.offsets(top_k_features).view(bs, self.num_queries, self.nl*self.num_points, 2)
        # add = sampling_offsets / self.num_points * enc_bboxes[:, :, None, 2:] * 0.5
        # sampling_locations = enc_bboxes[:, :, None, :2] + add 
        # # 按 value_spatial_shapes 的大小，将 value 按层拆分为不同的分辨率特征
        # value_list = feats.split([H_ * W_ for H_, W_ in shapes], dim=1)
        # sampling_grids = 2 * sampling_locations - 1
        # # [b,nq,num_point, 2]
        # grids_list = torch.split(sampling_grids, [2, 4, 6], dim=2)
        # sampling_value_list = []
        # for level, (H_, W_) in enumerate(shapes):
        #     # bs, H_*W_, hd ->
        #     # bs, hd, H_*W_ ->
        #     # bs, hd, H_, W_
        #     value_l_ = (value_list[level].transpose(1, 2).reshape(bs, hd, H_, W_))
        #     # bs, num_queries, num_points, 2 ->
        #     sampling_grid_l_ = grids_list[level]
        #     # bs*num_heads, embed_dims, num_queries, num_points
        #     # 函数解析：输入特征图，形状为[N,C,H,W]
        #     # 采样网格，形状为[N,Hout,Wout,2]
        #     # 输出为，[N,C,Hout,Wout]
        #     sampling_value_l_ = F.grid_sample(value_l_,
        #                                     sampling_grid_l_,
        #                                     mode='bilinear',
        #                                     padding_mode='zeros',
        #                                     align_corners=False)
        #     sampling_value_list.append(sampling_value_l_)
        # #(bs, hd, num_queries, num_points)
        # # num_points求和
        # embeddings_box = torch.cat(sampling_value_list, dim=-1).sum(-1).view(bs, hd, self.num_queries).transpose(1, 2).contiguous()

        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1) # (bs, num_queries + num_dn, 4)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1) # (bs, num_queries, nc)
        
        # if learnt_init_query: self.tgt_embed = nn.Embedding(nq, hd) （这里只是要一个维度为(nq, hd)的编码权重）这个不重要，实际是top_k_features
        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features # （bs, nq, hd）
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1) # [bs, num_dn + nq, dn_cls_embed = hd（256）] 

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init_` would cause NaN when training with custom datasets.
        # linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for reg_ in self.dec_bbox_head:
            # linear_init_(cls_)
            #constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)
################### end of DecouplingManbaWorldDecoder #################


################### DecouplingDFLManbaWorldDecoder ########################
class DecouplingDFLManbaWorldDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """
    export = False  # export mode

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hd=512,  # hidden dim
            nq=300,  # num queries
            ndp=4,  # num decoder points
            nh=8,  # num head
            ndl=6,  # num decoder layers
            d_ffn=1024,  # dim of feedforward
            eval_idx=-1,
            dropout=0.,
            act=nn.ReLU(),
            # Training args
            nd=100,  # num denoising
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False,
            # ========================
            dims=[128, 256, 512], 
            drop_path=[0.1, 0.1, 0.1],
            # =========================
            embed=512,
            with_bn=False
            ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
            使用给定参数初始化 RTDETRDecoder 模块。

         参数：
             nc (int)：类数。 默认值为 80。
             ch（元组）：主干特征图中的通道。 默认值为 (512, 1024, 2048)。
             hd (int)：隐藏层的维度。 默认值为 256。
             nq (int)：查询点的数量。 默认值为 300。
             ndp (int)：解码器点数。 默认值为 4。
             nh (int)：多头注意力中的头数。 默认值为 8。
             ndl (int)：解码器层数。 默认值为 6。
             d_ffn (int)：前馈网络的维度。 默认值为 1024。
             dropout (float): 辍学率。 默认值为 0。
             act (nn.Module)：激活函数。 默认为 nn.ReLU。
             eval_idx(int)：评价指标。 默认值为-1。
             nd (int): 去噪次数。 默认值为 100。
             label_noise_ratio (float)：标签噪声比。 默认值为 0.5。
             box_noise_scale (float)：盒子噪声尺度。 默认值为 1.0。
             learnt_init_query (bool)：是否学习初始查询嵌入。 默认值为 False。
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)
        
        # Manba VSSBlocks
        self.VSSBlocks = nn.ModuleList()
        self.num_Blocks = len(dims)
        for i_Block in range(self.num_Blocks):
            self.VSSBlocks.append(VSSBlock(
                hidden_dim=dims[i_Block],
                drop_path=drop_path[i_Block],
            ))

        # Transformer module
        decoder_layer = DecouplingDFLDeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DecouplingDFLTextDeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc + 1, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        #self.enc_output = VSSBlock(hidden_dim=hd, drop_path=0.1)
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        #self.enc_score_head = BNContrastiveHeadMLP(embed) if with_bn else ContrastiveHeadMLP() 
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([BNContrastiveHeadMLP(embed) if with_bn else ContrastiveHeadMLP() for _ in range(ndl)])
        #self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, text, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group
        #(Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Dict]]): 
        x = [self.VSSBlocks[i](feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) for i, feat in enumerate(x)]
        
        # Input projection and embedding
        # feats:[b, 20x20 + 40x40 + 80x80, 256] shapes: [nl, 2] cls:[b, 20x20 + 40x40 + 80x80, nc] 
        # 将分类和定位解耦开来，定位用trasnfromer 分类用CNN
        feats, shapes= self._get_encoder_input(x)

        # Prepare denoising training 
        #The modified class embeddings, bounding boxes, attention mask and meta information for denoising. 
        # 三维张量[bs, num_dn, dn_cls_embed], [bs, num_dn, 4], [num_dn + num_queries, num_dn + num_queries], {'dn_pos_idx','dn_num_group': num_group,'dn_num_split': [num_dn, num_queries]}
        # 创建用于 denoising（去噪）过程的查询组 将decoder的输出进行评估分为正样本和负样本后，负样本作为带噪声再次训练 也可以选择已经训练好的模型作为教师产生负样本一起训练
        dn_embed, dn_bbox, attn_mask, dn_meta = \
            get_cdn_group(batch,
                          self.nc,
                          self.num_queries,
                          self.denoising_class_embed.weight,
                          self.num_denoising,
                          self.label_noise_ratio,
                          self.box_noise_scale,
                          self.training)
        # embed = [bs, num_dn + nq, dn_cls_embed = hd（256）] refer_bbox: (bs, num_queries + num_dn, 4) enc_bboxes: (bs, num_queries, 4) enc_scores: (bs, num_queries, nc)
        embed, refer_bbox, enc_bboxes, enc_scores = \
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder 输入真实目标和噪声查询的组合
        # dec_bboxes：[6, bs, num_dn + nq, 4] dec_scores: [6, bs, num_dn + nq, nc]
        dec_bboxes, dec_scores = self.decoder(embed,
                                              refer_bbox,
                                              feats,
                                              shapes,
                                              text,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              dn_meta,
                                              attn_mask=attn_mask)
        # dec_bboxes：[6, bs, num_dn + nq, 4] dec_scores: [6, bs, num_dn + nq, nc] enc_bboxes: (bs, num_queries, 4) enc_scores: (bs, num_queries, nc)
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc) 删除第一维度 训练时用enc 推理时用dec
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        # 遍历了 shapes 中的每一个特征图形状 (h, w)，并使用 enumerate() 函数获取了索引 i 和形状信息 (h, w)
        for i, (h, w) in enumerate(shapes):
            # torch.arange() 函数生成了行和列的坐标 sy 和 sx，然后使用 torch.meshgrid() 创建了网格 grid_y 和 grid_x，这个网格表示了锚框可能出现的位置。
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2) 结果张量表示一个坐标网格，其中每个元素对应于一个锚框的中心点的 (x, y) 坐标。
            
            # 将网格坐标归一化到 [0, 1] 的范围内，这里加上了 0.5 是为了将坐标放在格子的中心
            valid_WH = torch.tensor([h, w], dtype=dtype, device=device)# (2)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)

            # 计算了锚框的宽度和高度，根据参数 grid_size 和当前迭代次数 i，锚框的尺寸按照指数级递增
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float('inf'))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        # self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch) 创建了三个1x1卷积，对通道维度进行调整到256
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)] #将三个输出维度的张量都拿出来进行投影，（投影也就是将通道维度都调整到256）
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c] 将[b, h*w, c]存入feats中
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2] 这里指出shapes的形状为[nl, 2] 
            shapes.append([h, w])

        # [b, h*w, c] 将三个nl输出的feats h*w合并 得到 [b, 20x20 + 40x40 + 80x80, c] 
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None, text=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = len(feats)
        # Prepare input for decoder
        # anchors: (1, 20x20 + 40x40 + 80x80, 4) 其大小由特征输出尺度决定 valid_mask：(1, 20x20 + 40x40 + 80x80, 1) 用来判断anchor是否有效
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        
        # self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd)) 改为manba
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256 (1, 20x20 + 40x40 + 80x80, 1) * [b, 20x20 + 40x40 + 80x80, c（256）] 
        
        # self.enc_score_head = nn.Linear(hd, nc)
        # self.enc_score_head = BNContrastiveHead(embed) if with_bn else ContrastiveHead()
        enc_outputs_scores = self.enc_score_head(features) # 经过一个线性层 bs, h*w, 256 -> (bs, h*w, nc) 至信度

        # Query selection
        # (bs, num_queries) 根据查询数获取前num_queries个enc_outputs_scores idx
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256) 利用张量三维索引得到topk_ind对应的特征
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1) 
        # (bs, num_queries, 4) 利用张量三维索引得到topk_ind对应的anchors
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        # self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)
        # 希望经过一个全连接层能得到bbox，将256转为4，（这里可以换成DFL） def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors #为什么直接把值加上去？(傻子，因为模型输出偏移量)好吧，无法理解的设计，自己设计了一个anchor再设计了一个bbox输出再加起来
        enc_bboxes = refer_bbox.sigmoid()


        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1) # (bs, num_queries + num_dn, 4)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1) # (bs, num_queries, nc)
        
        # if learnt_init_query: self.tgt_embed = nn.Embedding(nq, hd) （这里只是要一个维度为(nq, hd)的编码权重）这个不重要，实际是top_k_features
        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features # （bs, nq, hd）
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1) # [bs, num_dn + nq, dn_cls_embed = hd（256）] 

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init_` would cause NaN when training with custom datasets.
        # linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for reg_ in self.dec_bbox_head:
            # linear_init_(cls_)
            #constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)
################### end of DecouplingManbaWorldDecoder #################