# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh

#匈牙利匹配位置
class HungarianMatcher(nn.Module):
    """
    A module implementing the HungarianMatcher, which is a differentiable module to solve the assignment problem in an
    end-to-end fashion.

    HungarianMatcher performs optimal assignment over the predicted and ground truth bounding boxes using a cost
    function that considers classification scores, bounding box coordinates, and optionally, mask predictions.

    Attributes:
        cost_gain (dict): Dictionary of cost coefficients: 'class', 'bbox', 'giou', 'mask', and 'dice'.
        use_fl (bool): Indicates whether to use Focal Loss for the classification cost calculation.
        with_mask (bool): Indicates whether the model makes mask predictions.
        num_sample_points (int): The number of sample points used in mask cost calculation.
        alpha (float): The alpha factor in Focal Loss calculation.
        gamma (float): The gamma factor in Focal Loss calculation.

    Methods:
        forward(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=None, gt_mask=None): Computes the
            assignment between predictions and ground truths for a batch.
        _cost_mask(bs, num_gts, masks=None, gt_mask=None): Computes the mask cost and dice cost if masks are predicted.
    """

    def __init__(self, cost_gain=None, use_fl=True, with_mask=False, num_sample_points=12544, alpha=0.25, gamma=2.0):
        """Initializes HungarianMatcher with cost coefficients, Focal Loss, mask prediction, sample points, and alpha
        gamma factors.
        """
        super().__init__()
        if cost_gain is None:
            cost_gain = {'class': 1, 'bbox': 5, 'giou': 2, 'mask': 1, 'dice': 1}
        self.cost_gain = cost_gain
        self.use_fl = use_fl
        self.with_mask = with_mask
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=None, gt_mask=None):
        """
        Forward pass for HungarianMatcher. This function computes costs based on prediction and ground truth
        (classification cost, L1 cost between boxes and GIoU cost between boxes) and finds the optimal matching between
        predictions and ground truth based on these costs.

        Args:
            pred_bboxes (Tensor): Predicted bounding boxes with shape [batch_size, num_queries, 4].
            pred_scores (Tensor): Predicted scores with shape [batch_size, num_queries, num_classes].
            gt_cls (torch.Tensor): Ground truth classes with shape [num_gts, ].
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape [num_gts, 4].
            gt_groups (List[int]): List of length equal to batch size, containing the number of ground truths for
                each image.
            masks (Tensor, optional): Predicted masks with shape [batch_size, num_queries, height, width].
                Defaults to None.
            gt_mask (List[Tensor], optional): List of ground truth masks, each with shape [num_masks, Height, Width].
                Defaults to None.

        Returns:
            (List[Tuple[Tensor, Tensor]]): A list of size batch_size, each element is a tuple (index_i, index_j), where:
                - index_i is the tensor of indices of the selected predictions (in order)
                - index_j is the tensor of indices of the corresponding selected ground truth targets (in order)
                For each batch element, it holds:
                    len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, nq, nc = pred_scores.shape

        if sum(gt_groups) == 0:
            return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)) for _ in range(bs)]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]
        if not pred_scores.is_contiguous():
            pred_scores = pred_scores.contiguous()
        pred_scores = pred_scores.detach().view(-1, nc)
        pred_scores = F.sigmoid(pred_scores) if self.use_fl else F.softmax(pred_scores, dim=-1)
        # [batch_size * num_queries, 4]
        pred_bboxes = pred_bboxes.detach().view(-1, 4)

        # Compute the classification cost 匈牙利分配标签
        pred_scores = pred_scores[:, gt_cls] #根据真实目标类别（gt_cls）提取预测分数（pred_scores）中的对应类别得分，[batch_size * num_queries, num_gts]
        if self.use_fl: #如果使用focal loss
            neg_cost_class = (1 - self.alpha) * (pred_scores ** self.gamma) * (-(1 - pred_scores + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - pred_scores) ** self.gamma) * (-(pred_scores + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -pred_scores #这里为什么直接给负数呢？

        # Compute the L1 cost between boxes
        cost_bbox = (pred_bboxes.unsqueeze(1) - gt_bboxes.unsqueeze(0)).abs().sum(-1)  # (bs*num_queries, num_gt)

        # Compute the GIoU cost between boxes, (bs*num_queries, num_gt)
        cost_giou = 1.0 - bbox_iou(pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0), xywh=True, RIOU=True).squeeze(-1)

        # Final cost matrix
        C = self.cost_gain['class'] * cost_class + \
            self.cost_gain['bbox'] * cost_bbox + \
            self.cost_gain['giou'] * cost_giou
        # Compute the mask cost and dice cost
        if self.with_mask:
            C += self._cost_mask(bs, gt_groups, masks, gt_mask)

        # Set invalid values (NaNs and infinities) to 0 (fixes ValueError: matrix contains invalid numeric entries)
        C[C.isnan() | C.isinf()] = 0.0

        C = C.view(bs, nq, -1).cpu()
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(gt_groups, -1))]
        gt_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        # (idx for queries, idx for gt)
        return [(torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long) + gt_groups[k])
                for k, (i, j) in enumerate(indices)]

    # This function is for future RT-DETR Segment models
    # def _cost_mask(self, bs, num_gts, masks=None, gt_mask=None):
    #     assert masks is not None and gt_mask is not None, 'Make sure the input has `mask` and `gt_mask`'
    #     # all masks share the same set of points for efficient matching
    #     sample_points = torch.rand([bs, 1, self.num_sample_points, 2])
    #     sample_points = 2.0 * sample_points - 1.0
    #
    #     out_mask = F.grid_sample(masks.detach(), sample_points, align_corners=False).squeeze(-2)
    #     out_mask = out_mask.flatten(0, 1)
    #
    #     tgt_mask = torch.cat(gt_mask).unsqueeze(1)
    #     sample_points = torch.cat([a.repeat(b, 1, 1, 1) for a, b in zip(sample_points, num_gts) if b > 0])
    #     tgt_mask = F.grid_sample(tgt_mask, sample_points, align_corners=False).squeeze([1, 2])
    #
    #     with torch.cuda.amp.autocast(False):
    #         # binary cross entropy cost
    #         pos_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.ones_like(out_mask), reduction='none')
    #         neg_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.zeros_like(out_mask), reduction='none')
    #         cost_mask = torch.matmul(pos_cost_mask, tgt_mask.T) + torch.matmul(neg_cost_mask, 1 - tgt_mask.T)
    #         cost_mask /= self.num_sample_points
    #
    #         # dice cost
    #         out_mask = F.sigmoid(out_mask)
    #         numerator = 2 * torch.matmul(out_mask, tgt_mask.T)
    #         denominator = out_mask.sum(-1, keepdim=True) + tgt_mask.sum(-1).unsqueeze(0)
    #         cost_dice = 1 - (numerator + 1) / (denominator + 1)
    #
    #         C = self.cost_gain['mask'] * cost_mask + self.cost_gain['dice'] * cost_dice
    #     return C

#创建用于 denoising（去噪）过程的查询组
def get_cdn_group(batch,
                  num_classes,
                  num_queries,
                  class_embed,
                  num_dn=100,
                  cls_noise_ratio=0.5,
                  box_noise_scale=1.0,
                  training=False):
    """
    Get contrastive denoising training group. This function creates a contrastive denoising training group with positive
    and negative samples from the ground truths (gt). It applies noise to the class labels and bounding box coordinates,
    and returns the modified labels, bounding boxes, attention mask and meta information.
    获得对比去噪训练组。 该函数使用来自基本事实 (gt) 的正样本和负样本创建对比去噪训练组。 它将噪声应用于类标签和边界框坐标，并返回修改后的标签、边界框、注意力掩码和元信息。
    Args:
        batch (dict): A dict that includes 'gt_cls' (torch.Tensor with shape [num_gts, ]), 'gt_bboxes'
            (torch.Tensor with shape [num_gts, 4]), 'gt_groups' (List(int)) which is a list of batch size length
            indicating the number of gts of each image.
            它是批大小长度的列表，指示每个图像的 gts 数量。
        num_classes (int): Number of classes.
        num_queries (int): Number of queries.
        class_embed (torch.Tensor): Embedding weights to map class labels to embedding space.
        num_dn (int, optional): Number of denoising. Defaults to 100.
        cls_noise_ratio (float, optional): Noise ratio for class labels. Defaults to 0.5.
        box_noise_scale (float, optional): Noise scale for bounding box coordinates. Defaults to 1.0.
        training (bool, optional): If it's in training mode. Defaults to False.

    Returns:
        (Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Dict]]): The modified class embeddings,
            bounding boxes, attention mask and meta information for denoising. If not in training mode or 'num_dn'
            is less than or equal to 0, the function returns None for all elements in the tuple.
    """

    if (not training) or num_dn <= 0:
        return None, None, None, None
    # 这段代码获取了输入 batch 中的 gt_groups，计算了总的 ground truth 数量以及批次中最大的 ground truth 数量。
    # 如果最大数量为 0，则说明没有 ground truth，直接返回 None。
    gt_groups = batch['gt_groups']
    total_num = sum(gt_groups) # 所有图像中一共有多少gts
    max_nums = max(gt_groups) # 所有图像中最大的gts值
    if max_nums == 0:
        return None, None, None, None
    # 这段代码计算了每个样本组中的对比去噪数量，并确保至少有一个组。组数为 100/最大的gts值
    num_group = num_dn // max_nums 
    num_group = 1 if num_group == 0 else num_group
    # Pad gt to max_num of a batch
    bs = len(gt_groups)
    gt_cls = batch['cls']  # (bs*num, ) [num_gts, ] # [0,1,2,3,4,0,1,2,3,4]
    gt_bbox = batch['bboxes']  # bs*num, 4 [num_gts, 4]
    b_idx = batch['batch_idx']

    # Each group has positive and negative queries.
    dn_cls = gt_cls.repeat(2 * num_group)  # (2*num_group*bs*num, ) 复制两份以及分组用于生产dn
    dn_bbox = gt_bbox.repeat(2 * num_group, 1)  # 2*num_group*bs*num, 4
    dn_b_idx = b_idx.repeat(2 * num_group).view(-1)  # (2*num_group*bs*num, )

    # Positive and negative mask
    # (bs*num*num_group, ), the second total_num*num_group part as negative samples
    # PyTorch 中的 torch.arange 函数创建了一个从 0 到 total_num * num_group - 1 的整数序列 
    # 由于 total_num * num_group 代表了总的样本数量，所以加上这个值相当于将序列中的索引移动到表示负样本的位置。
    neg_idx = torch.arange(total_num * num_group, dtype=torch.long, device=gt_bbox.device) + num_group * total_num

    if cls_noise_ratio > 0:
        # Half of bbox prob
        mask = torch.rand(dn_cls.shape) < (cls_noise_ratio * 0.5) #根据概率有四分之一的的mask为true（1），四分之三为false（0）保留原张量维度
        idx = torch.nonzero(mask).squeeze(-1) #获取要添加噪声的索引
        # Randomly put a new one here
        new_label = torch.randint_like(idx, 0, num_classes, dtype=dn_cls.dtype, device=dn_cls.device) # 随机放一个新的类别
        dn_cls[idx] = new_label

    if box_noise_scale > 0:
        # 将边界框坐标 dn_bbox 从 (x_center, y_center, width, height) 的格式转换为 (x_min, y_min, x_max, y_max) 的格式，并将结果存储在 known_bbox 中。
        known_bbox = xywh2xyxy(dn_bbox)
        # 首先，它从 dn_bbox 的最后两个维度（即宽度和高度）中提取了值，然后乘以 0.5，表示噪声范围为边界框宽度和高度的一半。
        # 接着，使用 repeat() 方法将这个结果沿着第二维重复了两次，以便应用于每个边界框的宽度和高度
        diff = (dn_bbox[..., 2:] * 0.5).repeat(1, 2) * box_noise_scale  # 2*num_group*bs*num, 4
        # 这行代码生成了一个与 dn_bbox 张量相同形状的随机整数张量，值为 0 或 1，并乘以 2.0 后减去 1.0，得到了一个包含 -1 和 1 的张量，用于表示噪声的方向（左右上下）
        rand_sign = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0
        # 这行代码生成了一个与 dn_bbox 张量相同形状的随机张量，其中的每个元素都是从均匀分布 [0, 1) 中随机采样的。
        rand_part = torch.rand_like(dn_bbox)
        # 这行代码对部分随机噪声增加了 1.0，用于生成一部分边界框的噪声，确保它们是负样本。让这部分样本在[1,2]之间，远离真实值形成负样本
        rand_part[neg_idx] += 1.0
        # 这行代码将随机噪声乘以随机方向，以获取最终的噪声值
        rand_part *= rand_sign
        #  这行代码将噪声应用到边界框的坐标上。
        known_bbox += rand_part * diff
        # 这行代码确保边界框的坐标在合理的范围内，即在图像范围内。
        known_bbox.clip_(min=0.0, max=1.0)
        dn_bbox = xyxy2xywh(known_bbox)
        dn_bbox = torch.logit(dn_bbox, eps=1e-6)  # inverse sigmoid

    num_dn = int(max_nums * 2 * num_group)  # total denoising queries
    # class_embed = self.denoising_class_embed.weght = nn.Embedding(nc + 1, hd) nc+1肯定包含了dn_cls的范围 0-80
    dn_cls_embed = class_embed[dn_cls]  #  bs*num * 2 * num_group, 256
    
    # 这两行代码创建了两个用零填充的张量，用于存储修改后的类别嵌入和边界框坐标
    padding_cls = torch.zeros(bs, num_dn, dn_cls_embed.shape[-1], device=gt_cls.device)
    padding_bbox = torch.zeros(bs, num_dn, 4, device=gt_bbox.device)
    
    # 形成一个长整数序列 0，1，2，3，4，0，1，0，1，2，3 （对应三张图像）
    map_indices = torch.cat([torch.tensor(range(num), dtype=torch.long) for num in gt_groups])
   
    # 对于每个组（group），将 map_indices 中的每个值都加上相应的偏移量，偏移量是每个组中的最大数量 max_nums 与组的索引 i 的乘积。
    # 然后，使用 torch.stack() 函数将这些张量堆叠起来，形成一个张量，其形状为 (num_group, num_gts)，其中 num_group 是组的数量，num_gts 是一个批次中 ground truth 的总数量。
    pos_idx = torch.stack([map_indices + max_nums * i for i in range(num_group)], dim=0)
   
    #  这部分代码通过列表推导式生成了一个张量列表，其中每个张量都是 map_indices 加上相应偏移量的结果。 能有效区分不同组
    # （0，1，2，3，4，0，1，0，1，2，3）， （0，1，2，3，4，0，1，0，1，2，3）+ 5 ...... 一共有2 * num_group个 维度为2 * num_group * num_gts
    map_indices = torch.cat([map_indices + max_nums * i for i in range(2 * num_group)])

    '''
    padding_cls[(dn_b_idx, map_indices)] = dn_cls_embed: 这行代码使用了张量的高级索引功能。
    padding_cls 是一个三维张量，其形状为 (bs, num_dn, dn_cls_embed.shape[-1])，其中 bs 是批次大小，num_dn 是对比去噪数量。
    dn_b_idx 是一个表示样本所在批次索引的一维张量，map_indices 是一个表示正样本索引的一维张量，这两个张量组合在一起，用于指定应该填充的位置。
    然后，将 dn_cls_embed 张量填充到这些位置上。
    由于 dn_cls_embed 和 padding_cls 的最后一个维度大小相同，因此它们可以直接进行赋值操作。
    '''

    padding_cls[(dn_b_idx, map_indices)] = dn_cls_embed
    padding_bbox[(dn_b_idx, map_indices)] = dn_bbox

    tgt_size = num_dn + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
    # Match query cannot see the reconstruct
    attn_mask[num_dn:, :num_dn] = True
    # Reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), max_nums * 2 * (i + 1):num_dn] = True
        if i == num_group - 1:
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), :max_nums * i * 2] = True
        else:
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), max_nums * 2 * (i + 1):num_dn] = True
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), :max_nums * 2 * i] = True
    dn_meta = {
        'dn_pos_idx': [p.reshape(-1) for p in pos_idx.cpu().split(list(gt_groups), dim=1)],
        'dn_num_group': num_group,
        'dn_num_split': [num_dn, num_queries]}

    return padding_cls.to(class_embed.device), padding_bbox.to(class_embed.device), attn_mask.to(
        class_embed.device), dn_meta


# #创建用于 denoising（去噪）过程的查询组
# def get_cdn_group_withoutcls(batch,
#                   num_queries,
#                   num_dn=100,
#                   box_noise_scale=1.0,
#                   nc=10,
#                   training=False):
#     """
#     Get contrastive denoising training group. This function creates a contrastive denoising training group with positive
#     and negative samples from the ground truths (gt). It applies noise to the class labels and bounding box coordinates,
#     and returns the modified labels, bounding boxes, attention mask and meta information.
#     获得对比去噪训练组。 该函数使用来自基本事实 (gt) 的正样本和负样本创建对比去噪训练组。 它将噪声应用于类标签和边界框坐标，并返回修改后的标签、边界框、注意力掩码和元信息。
#     Args:
#         batch (dict): A dict that includes 'gt_cls' (torch.Tensor with shape [num_gts, ]), 'gt_bboxes'
#             (torch.Tensor with shape [num_gts, 4]), 'gt_groups' (List(int)) which is a list of batch size length
#             indicating the number of gts of each image.
#             它是批大小长度的列表，指示每个图像的 gts 数量。
#         num_classes (int): Number of classes.
#         num_queries (int): Number of queries.
#         class_embed (torch.Tensor): Embedding weights to map class labels to embedding space.
#         num_dn (int, optional): Number of denoising. Defaults to 100.
#         cls_noise_ratio (float, optional): Noise ratio for class labels. Defaults to 0.5.
#         box_noise_scale (float, optional): Noise scale for bounding box coordinates. Defaults to 1.0.
#         training (bool, optional): If it's in training mode. Defaults to False.

#     Returns:
#         (Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Dict]]): The modified class embeddings,
#             bounding boxes, attention mask and meta information for denoising. If not in training mode or 'num_dn'
#             is less than or equal to 0, the function returns None for all elements in the tuple.
#     """

#     if (not training) or num_dn <= 0:
#         return None, None, None, None
#     # 这段代码获取了输入 batch 中的 gt_groups，计算了总的 ground truth 数量以及批次中最大的 ground truth 数量。
#     # 如果最大数量为 0，则说明没有 ground truth，直接返回 None。
#     gt_groups = batch['gt_groups']
#     total_num = sum(gt_groups) # 所有图像中一共有多少gts
#     max_nums = max(gt_groups) # 所有图像中最大的gts值
#     if max_nums == 0:
#         return None, None, None, None
#     # 这段代码计算了每个样本组中的对比去噪数量，并确保至少有一个组。组数为 100/最大的gts值
#     num_group = num_dn // max_nums 
#     num_group = 1 if num_group == 0 else num_group
#     # Pad gt to max_num of a batch
#     bs = len(gt_groups)
#     gt_cls = batch['cls']  # (bs*num, ) [num_gts, ]
#     gt_bbox = batch['bboxes']  # bs*num, 4 [num_gts, 4]
#     b_idx = batch['batch_idx']

#     # Each group has positive and negative queries.
#     dn_cls = gt_cls.repeat(2 * num_group)  # (2*num_group*bs*num, ) 复制两份以及分组用于生产dn
#     dn_bbox = gt_bbox.repeat(2 * num_group, 1)  # 2*num_group*bs*num, 4
#     dn_b_idx = b_idx.repeat(2 * num_group).view(-1)  # (2*num_group*bs*num, )

#     # Positive and negative mask
#     # (bs*num*num_group, ), the second total_num*num_group part as negative samples
#     # PyTorch 中的 torch.arange 函数创建了一个从 0 到 total_num * num_group - 1 的整数序列 
#     # 由于 total_num * num_group 代表了总的样本数量，所以加上这个值相当于将序列中的索引移动到表示负样本的位置。
#     neg_idx = torch.arange(total_num * num_group, dtype=torch.long, device=gt_bbox.device) + num_group * total_num

#     if box_noise_scale > 0:
#         # 将边界框坐标 dn_bbox 从 (x_center, y_center, width, height) 的格式转换为 (x_min, y_min, x_max, y_max) 的格式，并将结果存储在 known_bbox 中。
#         known_bbox = xywh2xyxy(dn_bbox)
#         # 首先，它从 dn_bbox 的最后两个维度（即宽度和高度）中提取了值，然后乘以 0.5，表示噪声范围为边界框宽度和高度的一半。
#         # 接着，使用 repeat() 方法将这个结果沿着第二维重复了两次，以便应用于每个边界框的宽度和高度
#         diff = (dn_bbox[..., 2:] * 0.5).repeat(1, 2) * box_noise_scale  # 2*num_group*bs*num, 4
#         # 这行代码生成了一个与 dn_bbox 张量相同形状的随机整数张量，值为 0 或 1，并乘以 2.0 后减去 1.0，得到了一个包含 -1 和 1 的张量，用于表示噪声的方向（左右上下）
#         rand_sign = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0
#         # 这行代码生成了一个与 dn_bbox 张量相同形状的随机张量，其中的每个元素都是从均匀分布 [0, 1) 中随机采样的。
#         rand_part = torch.rand_like(dn_bbox)
#         # 这行代码对部分随机噪声增加了 1.0，用于生成一部分边界框的噪声，确保它们是负样本。让这部分样本在[1,2]之间，远离真实值形成负样本
#         rand_part[neg_idx] += 1.0
#         # 这行代码将随机噪声乘以随机方向，以获取最终的噪声值
#         rand_part *= rand_sign
#         #  这行代码将噪声应用到边界框的坐标上。
#         known_bbox += rand_part * diff
#         # 这行代码确保边界框的坐标在合理的范围内，即在图像范围内。
#         known_bbox.clip_(min=0.0, max=1.0)
#         dn_bbox = xyxy2xywh(known_bbox)
#         dn_bbox = torch.logit(dn_bbox, eps=1e-6)  # inverse sigmoid

#     num_dn = int(max_nums * 2 * num_group)  # total denoising queries
#     # class_embed = self.denoising_class_embed.weght = nn.Embedding(nc + 1, hd) nc+1肯定包含了dn_cls的范围 0-80
#     # 这里将类别编号转为置信度分布
#     b = dn_cls.shape[0]
#     one_hot = torch.zeros((b,  nc + 1), dtype=torch.float, device=gt_cls.device)
#     #targets为图像中存在的类别，scatter_为插入操作，pred_scores为
#     one_hot.scatter_(1, dn_cls.unsqueeze(-1), 1)
#     dn_cls_embed = one_hot[..., :-1]
#     # dn_cls_embed= torch.nn.functional.one_hot(dn_cls, num_classes=nc).to(torch.float)

    
#     # 这两行代码创建了两个用零填充的张量，用于存储修改后的类别嵌入和边界框坐标
#     padding_cls = torch.zeros(bs, num_dn, dn_cls_embed.shape[-1], device=gt_cls.device)
#     padding_bbox = torch.zeros(bs, num_dn, 4, device=gt_bbox.device)
    
#     # 形成一个长整数序列 0，1，2，3，4，0，1，0，1，2，3 （对应三张图像）
#     map_indices = torch.cat([torch.tensor(range(num), dtype=torch.long) for num in gt_groups])
   
#     # 对于每个组（group），将 map_indices 中的每个值都加上相应的偏移量，偏移量是每个组中的最大数量 max_nums 与组的索引 i 的乘积。
#     # 然后，使用 torch.stack() 函数将这些张量堆叠起来，形成一个张量，其形状为 (num_group, num_gts)，其中 num_group 是组的数量，num_gts 是一个批次中 ground truth 的总数量。
#     pos_idx = torch.stack([map_indices + max_nums * i for i in range(num_group)], dim=0)
   
#     #  这部分代码通过列表推导式生成了一个张量列表，其中每个张量都是 map_indices 加上相应偏移量的结果。 能有效区分不同组
#     # （0，1，2，3，4，0，1，0，1，2，3）， （0，1，2，3，4，0，1，0，1，2，3）+ 5 ...... 一共有2 * num_group个
#     map_indices = torch.cat([map_indices + max_nums * i for i in range(2 * num_group)])

#     '''
#     padding_cls[(dn_b_idx, map_indices)] = dn_cls_embed: 这行代码使用了张量的高级索引功能。
#     padding_cls 是一个三维张量，其形状为 (bs, num_dn, dn_cls_embed.shape[-1])，其中 bs 是批次大小，num_dn 是对比去噪数量。
#     dn_b_idx 是一个表示样本所在批次索引的一维张量，map_indices 是一个表示正样本索引的一维张量，这两个张量组合在一起，用于指定应该填充的位置。
#     然后，将 dn_cls_embed 张量填充到这些位置上。
#     由于 dn_cls_embed 和 padding_cls 的最后一个维度大小相同，因此它们可以直接进行赋值操作。
#     '''
#     padding_cls[(dn_b_idx, map_indices)] = dn_cls_embed
#     padding_bbox[(dn_b_idx, map_indices)] = dn_bbox

#     tgt_size = num_dn + num_queries
#     attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
#     # Match query cannot see the reconstruct
#     attn_mask[num_dn:, :num_dn] = True
#     # Reconstruct cannot see each other
#     for i in range(num_group):
#         if i == 0:
#             attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), max_nums * 2 * (i + 1):num_dn] = True
#         if i == num_group - 1:
#             attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), :max_nums * i * 2] = True
#         else:
#             attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), max_nums * 2 * (i + 1):num_dn] = True
#             attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), :max_nums * 2 * i] = True
#     dn_meta = {
#         'dn_pos_idx': [p.reshape(-1) for p in pos_idx.cpu().split(list(gt_groups), dim=1)],
#         'dn_num_group': num_group,
#         'dn_num_split': [num_dn, num_queries]}

#     return padding_cls.to('cuda'), padding_bbox.to('cuda'), attn_mask.to(
#         'cuda'), dn_meta


#创建用于 denoising（去噪）过程的查询组
def get_cdn_group_withoutcls(batch,
                  num_queries,
                  num_dn=100,
                  box_noise_scale=1.0,
                  training=False):
    """
    Get contrastive denoising training group. This function creates a contrastive denoising training group with positive
    and negative samples from the ground truths (gt). It applies noise to the class labels and bounding box coordinates,
    and returns the modified labels, bounding boxes, attention mask and meta information.
    获得对比去噪训练组。 该函数使用来自基本事实 (gt) 的正样本和负样本创建对比去噪训练组。 它将噪声应用于类标签和边界框坐标，并返回修改后的标签、边界框、注意力掩码和元信息。
    Args:
        batch (dict): A dict that includes 'gt_cls' (torch.Tensor with shape [num_gts, ]), 'gt_bboxes'
            (torch.Tensor with shape [num_gts, 4]), 'gt_groups' (List(int)) which is a list of batch size length
            indicating the number of gts of each image.
            它是批大小长度的列表，指示每个图像的 gts 数量。
        num_classes (int): Number of classes.
        num_queries (int): Number of queries.
        class_embed (torch.Tensor): Embedding weights to map class labels to embedding space.
        num_dn (int, optional): Number of denoising. Defaults to 100.
        cls_noise_ratio (float, optional): Noise ratio for class labels. Defaults to 0.5.
        box_noise_scale (float, optional): Noise scale for bounding box coordinates. Defaults to 1.0.
        training (bool, optional): If it's in training mode. Defaults to False.

    Returns:
        (Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Dict]]): The modified class embeddings,
            bounding boxes, attention mask and meta information for denoising. If not in training mode or 'num_dn'
            is less than or equal to 0, the function returns None for all elements in the tuple.
    """

    if (not training) or num_dn <= 0:
        return None, None, None
    # 这段代码获取了输入 batch 中的 gt_groups，计算了总的 ground truth 数量以及批次中最大的 ground truth 数量。
    # 如果最大数量为 0，则说明没有 ground truth，直接返回 None。
    gt_groups = batch['gt_groups']
    total_num = sum(gt_groups) # 所有图像中一共有多少gts
    max_nums = max(gt_groups) # 所有图像中最大的gts值
    if max_nums == 0:
        return None, None, None
    # 这段代码计算了每个样本组中的对比去噪数量，并确保至少有一个组。组数为 100/最大的gts值
    num_group = num_dn // max_nums 
    num_group = 1 if num_group == 0 else num_group
    # Pad gt to max_num of a batch
    bs = len(gt_groups)
    gt_bbox = batch['bboxes']  # bs*num, 4 [num_gts, 4]
    b_idx = batch['batch_idx']

    # Each group has positive and negative queries.
    dn_bbox = gt_bbox.repeat(2 * num_group, 1)  # 2*num_group*bs*num, 4
    dn_b_idx = b_idx.repeat(2 * num_group).view(-1)  # (2*num_group*bs*num, )

    # Positive and negative mask
    # (bs*num*num_group, ), the second total_num*num_group part as negative samples
    # PyTorch 中的 torch.arange 函数创建了一个从 0 到 total_num * num_group - 1 的整数序列 
    # 由于 total_num * num_group 代表了总的样本数量，所以加上这个值相当于将序列中的索引移动到表示负样本的位置。
    neg_idx = torch.arange(total_num * num_group, dtype=torch.long, device=gt_bbox.device) + num_group * total_num

    if box_noise_scale > 0:
        # 将边界框坐标 dn_bbox 从 (x_center, y_center, width, height) 的格式转换为 (x_min, y_min, x_max, y_max) 的格式，并将结果存储在 known_bbox 中。
        known_bbox = xywh2xyxy(dn_bbox)
        # 首先，它从 dn_bbox 的最后两个维度（即宽度和高度）中提取了值，然后乘以 0.5，表示噪声范围为边界框宽度和高度的一半。
        # 接着，使用 repeat() 方法将这个结果沿着第二维重复了两次，以便应用于每个边界框的宽度和高度
        diff = (dn_bbox[..., 2:] * 0.5).repeat(1, 2) * box_noise_scale  # 2*num_group*bs*num, 4
        # 这行代码生成了一个与 dn_bbox 张量相同形状的随机整数张量，值为 0 或 1，并乘以 2.0 后减去 1.0，得到了一个包含 -1 和 1 的张量，用于表示噪声的方向（左右上下）
        rand_sign = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0
        # 这行代码生成了一个与 dn_bbox 张量相同形状的随机张量，其中的每个元素都是从均匀分布 [0, 1) 中随机采样的。
        rand_part = torch.rand_like(dn_bbox)
        # 这行代码对部分随机噪声增加了 1.0，用于生成一部分边界框的噪声，确保它们是负样本。让这部分样本在[1,2]之间，远离真实值形成负样本
        rand_part[neg_idx] += 1.0
        # 这行代码将随机噪声乘以随机方向，以获取最终的噪声值
        rand_part *= rand_sign
        #  这行代码将噪声应用到边界框的坐标上。
        known_bbox += rand_part * diff
        # 这行代码确保边界框的坐标在合理的范围内，即在图像范围内。
        known_bbox.clip_(min=0.0, max=1.0)
        dn_bbox = xyxy2xywh(known_bbox)
        dn_bbox = torch.logit(dn_bbox, eps=1e-6)  # inverse sigmoid

    num_dn = int(max_nums * 2 * num_group)  # total denoising queries
    # class_embed = self.denoising_class_embed.weght = nn.Embedding(nc + 1, hd) nc+1肯定包含了dn_cls的范围 0-80
    # 这里将类别编号转为置信度分布

    
    # 这两行代码创建了两个用零填充的张量，用于存储修改后的类别嵌入和边界框坐标
    padding_bbox = torch.zeros(bs, num_dn, 4, device=gt_bbox.device)
    
    # 形成一个长整数序列 0，1，2，3，4，0，1，0，1，2，3 （对应三张图像）
    map_indices = torch.cat([torch.tensor(range(num), dtype=torch.long) for num in gt_groups])
   
    # 对于每个组（group），将 map_indices 中的每个值都加上相应的偏移量，偏移量是每个组中的最大数量 max_nums 与组的索引 i 的乘积。
    # 然后，使用 torch.stack() 函数将这些张量堆叠起来，形成一个张量，其形状为 (num_group, num_gts)，其中 num_group 是组的数量，num_gts 是一个批次中 ground truth 的总数量。
    pos_idx = torch.stack([map_indices + max_nums * i for i in range(num_group)], dim=0)
   
    #  这部分代码通过列表推导式生成了一个张量列表，其中每个张量都是 map_indices 加上相应偏移量的结果。 能有效区分不同组
    # （0，1，2，3，4，0，1，0，1，2，3）， （0，1，2，3，4，0，1，0，1，2，3）+ 5 ...... 一共有2 * num_group个
    map_indices = torch.cat([map_indices + max_nums * i for i in range(2 * num_group)])

    '''
    padding_cls[(dn_b_idx, map_indices)] = dn_cls_embed: 这行代码使用了张量的高级索引功能。
    padding_cls 是一个三维张量，其形状为 (bs, num_dn, dn_cls_embed.shape[-1])，其中 bs 是批次大小，num_dn 是对比去噪数量。
    dn_b_idx 是一个表示样本所在批次索引的一维张量，map_indices 是一个表示正样本索引的一维张量，这两个张量组合在一起，用于指定应该填充的位置。
    然后，将 dn_cls_embed 张量填充到这些位置上。
    由于 dn_cls_embed 和 padding_cls 的最后一个维度大小相同，因此它们可以直接进行赋值操作。
    '''

    padding_bbox[(dn_b_idx, map_indices)] = dn_bbox

    tgt_size = num_dn + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
    # Match query cannot see the reconstruct
    attn_mask[num_dn:, :num_dn] = True
    # Reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), max_nums * 2 * (i + 1):num_dn] = True
        if i == num_group - 1:
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), :max_nums * i * 2] = True
        else:
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), max_nums * 2 * (i + 1):num_dn] = True
            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), :max_nums * 2 * i] = True
    dn_meta = {
        'dn_pos_idx': [p.reshape(-1) for p in pos_idx.cpu().split(list(gt_groups), dim=1)],
        'dn_num_group': num_group,
        'dn_num_split': [num_dn, num_queries]}

    return padding_bbox.to('cuda'), attn_mask.to('cuda'), dn_meta
