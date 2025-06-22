# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Transformer modules."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from .conv import Conv
from .utils import _get_clones, inverse_sigmoid, multi_scale_deformable_attn_pytorch, multi_scale_deformable_attn_pytorch_cls, multi_scale_deformable_attn_pytorch_box

__all__ = ('TransformerEncoderLayer', 'TransformerLayer', 'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'DecouplingDeformableTransformerDecoderLayer', 'DecouplingDFLDeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP', 'TextDeformableTransformerDecoder', 'locationDeformableTransformerDecoder', 'DecouplingTextDeformableTransformerDecoder', 'DecouplingDFLTextDeformableTransformerDecoder')


class TransformerEncoderLayer(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        from ...utils.torch_utils import TORCH_1_9
        if not TORCH_1_9:
            raise ModuleNotFoundError(
                'TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True).')
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos=None):
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with pre-normalization."""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        """Forward pass for the AIFI transformer layer."""
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Builds 2D sine-cosine position embedding."""
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]

class TransformerLayer(nn.Module):
    """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""

    def __init__(self, c, num_heads):
        """Initializes a self-attention mechanism using linear transformations and multi-head attention."""
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        """Apply a transformer block to the input x and return the output."""
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x


class TransformerBlock(nn.Module):
    """Vision Transformer https://arxiv.org/abs/2010.11929."""

    def __init__(self, c1, c2, num_heads, num_layers):
        """Initialize a Transformer module with position embedding and specified number of heads and layers."""
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        """Forward propagates the input through the bottleneck module."""
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class MLPBlock(nn.Module):
    """Implements a single block of a multi-layer perceptron."""

    def __init__(self, embedding_dim, mlp_dim, act=nn.GELU):
        """Initialize the MLPBlock with specified embedding dimension, MLP dimension, and activation function."""
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLPBlock."""
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):
    """Implements a simple multi-layer perceptron (also called FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """Initialize the MLP with specified input, hidden, output dimensions and number of layers."""
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        """Forward pass for the entire MLP."""
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LayerNorm2d(nn.Module):
    """
    2D Layer Normalization module inspired by Detectron2 and ConvNeXt implementations.

    Original implementations in
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
    and
    https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py.
    """

    def __init__(self, num_channels, eps=1e-6):
        """Initialize LayerNorm2d with the given parameters."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        """Perform forward pass for 2D layer normalization."""
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MSDeformAttn(nn.Module):
    """
    Multi-Scale Deformable Attention Module based on Deformable-DETR and PaddleDetection implementations.

    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """Initialize MSDeformAttn with the given parameters."""
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'd_model must be divisible by n_heads, but got {d_model} and {n_heads}')
        _d_per_head = d_model // n_heads
        # Better to set _d_per_head to a power of 2 which is more efficient in a CUDA implementation
        assert _d_per_head * n_heads == d_model, '`d_model` must be divisible by `n_heads`'

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """Reset module parameters."""
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(
            1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        """
        Perform forward pass for multiscale deformable attention.

        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py

        Args:
            query (torch.Tensor): [bs, query_length, C] [bs, nd+nq, C]
            refer_bbox (torch.Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (torch.Tensor): [bs, value_length, C]
            value_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, len_q = query.shape[:2] # bs 和 nd+nq
        len_v = value.shape[1] # [20x20 + 40x40 + 80x80]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads) # 化分为头 
        # 与query有关的是sampling_offsets和attention_weights
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2) # 8 4 4 [bs, query_length, 8, n_levels（4）, 4, 2] 
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points) # [bs, query_length, 8, n_levels（4）* 4] 
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        num_points = refer_bbox.shape[-1] # num_points = 4
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            # refer_bbox[:, :, None, :, None, 2:] = [bs, query_length, 1, n_levels（1）, 1, 2]  # 提取了宽高信息（假设 num_points=4）
            # sampling_offsets = [bs, query_length, 8, n_levels（4）, 4, 2] 
            # 将偏移量缩放为相对于采样点的比例，将采样点的偏移比例映射到实际宽高范围内
            # sampling_locations = [bs, query_length, 8, 4, 4, 2]
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f'Last dim of reference_points must be 2 or 4, but got {num_points}.')
        # value = [bs, 20x20 + 40x40 + 80x80, 8, 256/8] value_shapes = [20x20 + 40x40 + 80x80] 
        # sampling_locations = [bs, query_length, 8, 3 * 4, 2] attention_weights = [bs, query_length, 8, n_levels（3）, 4, 2] 
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)
    

class MSDeformAttncls(nn.Module):
    """
    Multi-Scale Deformable Attention Module based on Deformable-DETR and PaddleDetection implementations.

    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """Initialize MSDeformAttn with the given parameters."""
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'd_model must be divisible by n_heads, but got {d_model} and {n_heads}')
        _d_per_head = d_model // n_heads
        # Better to set _d_per_head to a power of 2 which is more efficient in a CUDA implementation
        assert _d_per_head * n_heads == d_model, '`d_model` must be divisible by `n_heads`'

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """Reset module parameters."""
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(
            1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        """
        Perform forward pass for multiscale deformable attention.

        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py

        Args:
            query (torch.Tensor): [bs, query_length, C] [bs, nd+nq, C]
            refer_bbox (torch.Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (torch.Tensor): [bs, value_length, C]
            value_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, len_q = query.shape[:2] # bs 和 nd+nq
        len_v = value.shape[1] # [20x20 + 40x40 + 80x80]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads) # 化分为头 
        # 与query有关的是sampling_offsets和attention_weights
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points, 2) # 8 4 4 [bs, query_length, 8, n_levels（4）, 4, 2] 
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points) # [bs, query_length, 8, n_levels（4）* 4] 
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        num_points = refer_bbox.shape[-1] # num_points = 4
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            # refer_bbox[:, :, None, :, None, 2:] = [bs, query_length, 1, n_levels（1）, 1, 2]  # 提取了宽高信息（假设 num_points=4）
            # sampling_offsets = [bs, query_length, 8, n_levels（4）, 4, 2] 
            # 将偏移量缩放为相对于采样点的比例，将采样点的偏移比例映射到实际宽高范围内
            # sampling_locations = [bs, query_length, 8, 4, 4, 2]
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, :2] + add
        else:
            raise ValueError(f'Last dim of reference_points must be 2 or 4, but got {num_points}.')
        # value = [bs, 20x20 + 40x40 + 80x80, 8, 256/8] value_shapes = [20x20 + 40x40 + 80x80] 
        # sampling_locations = [bs, query_length, 8, 3 * 4, 2] attention_weights = [bs, query_length, 8, n_levels（3）, 4, 2] 
        output = multi_scale_deformable_attn_pytorch_cls(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)
    

class MSDeformAttnbox(nn.Module):
    """
    Multi-Scale Deformable Attention Module based on Deformable-DETR and PaddleDetection implementations.

    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """Initialize MSDeformAttn with the given parameters."""
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'd_model must be divisible by n_heads, but got {d_model} and {n_heads}')
        _d_per_head = d_model // n_heads
        # Better to set _d_per_head to a power of 2 which is more efficient in a CUDA implementation
        assert _d_per_head * n_heads == d_model, '`d_model` must be divisible by `n_heads`'

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """Reset module parameters."""
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(
            1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        """
        Perform forward pass for multiscale deformable attention.

        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py

        Args:
            query (torch.Tensor): [bs, query_length, C] [bs, nd+nq, C]
            refer_bbox (torch.Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (torch.Tensor): [bs, value_length, C]
            value_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, len_q = query.shape[:2] # bs 和 nd+nq
        len_v = value.shape[1] # [20x20 + 40x40 + 80x80]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads) # 化分为头 
        # 与query有关的是sampling_offsets和attention_weights
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points, 2) # 8 4 4 [bs, query_length, 8, n_levels（4）, 4, 2] 
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points) # [bs, query_length, 8, n_levels（4）* 4] 
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        num_points = refer_bbox.shape[-1] # num_points = 4
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            # refer_bbox[:, :, None, :, None, 2:] = [bs, query_length, 1, n_levels（1）, 1, 2]  # 提取了宽高信息（假设 num_points=4）
            # sampling_offsets = [bs, query_length, 8, n_levels（4）, 4, 2] 
            # 将偏移量缩放为相对于采样点的比例，将采样点的偏移比例映射到实际宽高范围内
            # sampling_locations = [bs, query_length, 8, 4, 4, 2]
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, :2] + add
        else:
            raise ValueError(f'Last dim of reference_points must be 2 or 4, but got {num_points}.')
        # value = [bs, 20x20 + 40x40 + 80x80, 8, 256/8] value_shapes = [20x20 + 40x40 + 80x80] 
        # sampling_locations = [bs, query_length, 8, 3 * 4, 2] attention_weights = [bs, query_length, 8, n_levels（3）, 4, 2] 
        output = multi_scale_deformable_attn_pytorch_box(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)


class DeformableTransformerDecoderLayer(nn.Module):
    """
    Deformable Transformer Decoder Layer inspired by PaddleDetection and Deformable-DETR implementations.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0., act=nn.ReLU(), n_levels=4, n_points=4):
        """Initialize the DeformableTransformerDecoderLayer with the given parameters."""
        super().__init__()

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Add positional embeddings to the input tensor, if provided."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        """Perform forward pass through the Feed-Forward Network part of the layer."""
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None, attn_mask=None, query_pos=None):
        """Perform the forward pass through the entire decoder layer."""

        # Self attention
        # 将代表类别[bs, num_dn + nq, dn_cls_embed = hd（256）] 和 代表位置refer_bbox->MLP->query_pos (bs, num_queries + num_dn, 256) 加起来
        q = k = self.with_pos_embed(embed, query_pos) #q k = embed + query_pos
        # 使用多头注意力机制，输入包括 q k embed=v attn_mask 让dn和nq之间不能相互注意 这里用embed做v 让类别信息和位置信息进行匹配，并强化类别信息
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1),
                             attn_mask=attn_mask)[0].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # Cross attention
        tgt = self.cross_attn(self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes,
                              padding_mask)
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # FFN
        return self.forward_ffn(embed)
    

class DecouplingDeformableTransformerDecoderLayer(nn.Module):
    """
    Deformable Transformer Decoder Layer inspired by PaddleDetection and Deformable-DETR implementations.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0., act=nn.ReLU(), n_levels=4, n_points=4):
        """Initialize the DeformableTransformerDecoderLayer with the given parameters."""
        super().__init__()

        # Self attention
        self.self_attn1 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # self.self_attn2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        # self.dropout2 = nn.Dropout(dropout)
        # self.norm2 = nn.LayerNorm(d_model)

        # Cross attention
        self.cross_attn_cls = MSDeformAttncls(d_model, n_levels, n_heads, n_points)
        self.cross_attn_box = MSDeformAttnbox(d_model, n_levels, n_heads, n_points)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm4 = nn.LayerNorm(d_model)

        # FFN1
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout5 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout6 = nn.Dropout(dropout)
        self.norm5 = nn.LayerNorm(d_model)

        # FFN2
        self.linear3 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout7 = nn.Dropout(dropout)
        self.linear4 = nn.Linear(d_ffn, d_model)
        self.dropout8 = nn.Dropout(dropout)
        self.norm6 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Add positional embeddings to the input tensor, if provided."""
        return tensor if pos is None else tensor + pos

    def forward_ffn1(self, tgt):
        """Perform forward pass through the Feed-Forward Network part of the layer."""
        tgt2 = self.linear2(self.dropout5(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout6(tgt2)
        return self.norm5(tgt)
    
    def forward_ffn2(self, tgt):
        """Perform forward pass through the Feed-Forward Network part of the layer."""
        tgt2 = self.linear4(self.dropout7(self.act(self.linear3(tgt))))
        tgt = tgt + self.dropout8(tgt2)
        return self.norm6(tgt)

    def forward(self, embed, embed1, refer_bbox, feats, shapes, padding_mask=None, attn_mask=None, query_pos=None, dn_meta=None):
        """Perform the forward pass through the entire decoder layer."""
    # 类别输出
        # Self attention
        # 前k个特征和dn类别值[bs, num_dn + nq, dn_cls_embed = hd（256）] 和 代表位置refer_bbox->MLP->query_pos (bs,  num_dn+nq, 256) 加起来
        # feat是所有特征
        q = k = self.with_pos_embed(embed, query_pos) #q k = embed + query_pos
        # 使用多头注意力机制，输入包括 q k embed=v attn_mask 让dn和nq之间不能相互注意 这里用embed做v 让类别信息和位置信息进行匹配，并强化类别信息
        # if dn_meta is not None:
        #     dn_bboxes, _ = torch.split(query_pos, dn_meta['dn_num_split'], dim=1)
        #     embed_box =  torch.cat([dn_bboxes, embed_box], 1)


        tgt = self.self_attn1(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1),
                             attn_mask=attn_mask)[0].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # tgt = self.self_attn2(embed_box.transpose(0, 1), embed_box.transpose(0, 1), embed_box.transpose(0, 1),
        #                      attn_mask=attn_mask)[0].transpose(0, 1)
        # embed_box = embed_box + self.dropout2(tgt)
        # embed_box = self.norm2(embed_box)

        # Cross attention refer_bbox在中插一个维度(bs, num_dn + nq, 1, 4)
        tgt = self.cross_attn_cls(self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes,
                              padding_mask)
        embed = embed + self.dropout3(tgt)
        embed = self.norm3(embed)

        # 这里我们给他调整为时序模型，而不是注意力，manba？把曼巴结合这个
        tgt = self.cross_attn_box(self.with_pos_embed(embed1, query_pos), refer_bbox.unsqueeze(2), feats, shapes,
                              padding_mask)
        embed1 = embed1 + self.dropout4(tgt)
        embed1 = self.norm4(embed1)

        # FFN
        return self.forward_ffn1(embed), self.forward_ffn2(embed1)
        #return self.forward_ffn1(embed)


class DeformableTransformerDecoder(nn.Module):
    """
    Implementation of Deformable Transformer Decoder based on PaddleDetection.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        """Initialize the DeformableTransformerDecoder with the given parameters."""
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
            self,
            embed,  # decoder embeddings [bs, num_dn + nq, dn_cls_embed = hd（256）]
            refer_bbox,  # anchor (bs, num_queries + num_dn, 4)
            feats,  # image features [b, 20x20 + 40x40 + 80x80, c]
            shapes,  # feature shapes [nl, 2]
            bbox_head, # self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])
            score_head, # self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
            pos_mlp, # self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2) 
            attn_mask=None, # [num_dn + num_queries, num_dn + num_queries]
            padding_mask=None):
        """Perform the forward pass through the entire decoder."""
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        # 默认有6层解码器 output = (bs, num_queries + num_dn, 256)
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))

            bbox = bbox_head[i](output) # 和解码器层数一样，将经过自注意力和交叉注意力的特征映射为坐标（为什么一共要搞6个？）
            # inverse_sigmoid 逆sigmoid (bs, num_queries + num_dn, 4)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox)) 
            # 训练阶段 类别置信度一直累加 边界框也一直加，但使用sigmoid平衡6个解码器
            if self.training:
                dec_cls.append(score_head[i](output)) # [bs, num_dn + nq, dn_cls_embed = hd（256）] -> [bs, num_dn + nq, nc]
                if i == 0:
                    dec_bboxes.append(refined_bbox) 
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx: # 到最后了
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox
        # 将6个结果stack起来，默认维度为0
        return torch.stack(dec_bboxes), torch.stack(dec_cls)
    

####################
class DecouplingTextDeformableTransformerDecoder(nn.Module):
    """
    Implementation of Deformable Transformer Decoder based on PaddleDetection.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        """Initialize the DeformableTransformerDecoder with the given parameters."""
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
            self, 
            embed,  # decoder embeddings [bs, num_dn + nq, dn_cls_embed = hd（256）] topk_feature (decoder queries可以看成是anchor位置的学习）
            refer_bbox,  # anchor (bs, num_queries + num_dn, 4)
            feats,  # image features [b, 20x20 + 40x40 + 80x80, c]
            shapes,  # feature shapes [nl, 2]
            text, # text feature [b, k, c]
            bbox_head, # self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])
            score_head, # self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
            pos_mlp, # self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2) 
            dn_meta,
            attn_mask=None, # [num_dn + num_queries, num_dn + num_queries]
            padding_mask=None):
        """Perform the forward pass through the entire decoder."""
        output1 = embed
        output2 = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        # 默认有6层解码器 使用MLP将refer_bbox编码到(bs, num_queries + num_dn, 256)作为位置编码，在自注意力中和类别编码放到一起
        for i, layer in enumerate(self.layers):
            output1, output2 = layer(output1, output2, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox), dn_meta)
            #可不可以像RNN将每个解码器的输出结果进行正负样本分配
            #在每一层decoder中都会去预测相对偏移量，并去更新检测框，得到一个更加精确的检测框预测 
            bbox = bbox_head[i](output2) # 和解码器层数一样，将经过自注意力和交叉注意力的特征映射为坐标偏移量（为什么一共要搞6个？）
            # inverse_sigmoid 逆sigmoid (bs, num_queries + num_dn, 4)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox)) 
            # 训练阶段 类别置信度一直累加 边界框也一直加，但使用sigmoid平衡6个解码器
            if self.training:
                dec_cls.append(score_head[i](output1, text)) # [bs, num_dn + nq, dn_cls_embed = hd（256）] -> [bs, num_dn + nq, nc]
                if i == 0:
                    dec_bboxes.append(refined_bbox) 
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx: # 到最后了
                dec_cls.append(score_head[i](output1, text))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox #更新检测框
        # 将6个结果stack起来，默认维度为0
        return torch.stack(dec_bboxes), torch.stack(dec_cls)
    

class locationDeformableTransformerDecoder(nn.Module):
    """
    Implementation of Deformable Transformer Decoder based on PaddleDetection.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        """Initialize the DeformableTransformerDecoder with the given parameters."""
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
            self, 
            embed,  # decoder embeddings [bs, num_dn + nq, dn_cls_embed = hd（256）] topk_feature (decoder queries可以看成是anchor位置的学习）
            refer_bbox,  # anchor (bs, num_queries + num_dn, 4)
            feats,  # image features [b, 20x20 + 40x40 + 80x80, c]
            shapes,  # feature shapes [nl, 2]
            bbox_head, # self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])
            pos_mlp, # self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2) 
            attn_mask=None, # [num_dn + num_queries, num_dn + num_queries]
            padding_mask=None):
        """Perform the forward pass through the entire decoder."""
        output = embed
        dec_bboxes = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        # 默认有6层解码器 使用MLP将refer_bbox编码到(bs, num_queries + num_dn, 256)作为位置编码，在自注意力中和类别编码放到一起
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))
            #可不可以像RNN将每个解码器的输出结果进行正负样本分配
            #在每一层decoder中都会去预测相对偏移量，并去更新检测框，得到一个更加精确的检测框预测 
            bbox = bbox_head[i](output) # 和解码器层数一样，将经过自注意力和交叉注意力的特征映射为坐标偏移量（为什么一共要搞6个？）
            # inverse_sigmoid 逆sigmoid (bs, num_queries + num_dn, 4)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox)) 
            # 训练阶段 类别置信度一直累加 边界框也一直加，但使用sigmoid平衡6个解码器
            if self.training:
                if i == 0:
                    dec_bboxes.append(refined_bbox) 
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx: # 到最后了
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox #更新检测框
        # 将6个结果stack起来，默认维度为0
        return torch.stack(dec_bboxes)

####################
class TextDeformableTransformerDecoder(nn.Module):
    """
    Implementation of Deformable Transformer Decoder based on PaddleDetection.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        """Initialize the DeformableTransformerDecoder with the given parameters."""
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
            self, 
            embed,  # decoder embeddings [bs, num_dn + nq, dn_cls_embed = hd（256）] topk_feature (decoder queries可以看成是anchor位置的学习）
            refer_bbox,  # anchor (bs, num_queries + num_dn, 4)
            feats,  # image features [b, 20x20 + 40x40 + 80x80, c]
            shapes,  # feature shapes [nl, 2]
            text, # text feature [b, k, c]
            bbox_head, # self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])
            score_head, # self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
            pos_mlp, # self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2) 
            attn_mask=None, # [num_dn + num_queries, num_dn + num_queries]
            padding_mask=None):
        """Perform the forward pass through the entire decoder."""
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        # 默认有6层解码器 使用MLP将refer_bbox编码到(bs, num_queries + num_dn, 256)作为位置编码，在自注意力中和类别编码放到一起
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))
            #可不可以像RNN将每个解码器的输出结果进行正负样本分配
            #在每一层decoder中都会去预测相对偏移量，并去更新检测框，得到一个更加精确的检测框预测 
            bbox = bbox_head[i](output) # 和解码器层数一样，将经过自注意力和交叉注意力的特征映射为坐标偏移量（为什么一共要搞6个？）
            # inverse_sigmoid 逆sigmoid (bs, num_queries + num_dn, 4)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox)) 
            # 训练阶段 类别置信度一直累加 边界框也一直加，但使用sigmoid平衡6个解码器
            if self.training:
                dec_cls.append(score_head[i](output, text)) # [bs, num_dn + nq, dn_cls_embed = hd（256）] -> [bs, num_dn + nq, nc]
                if i == 0:
                    dec_bboxes.append(refined_bbox) 
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx: # 到最后了
                dec_cls.append(score_head[i](output, text))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox #更新检测框
        # 将6个结果stack起来，默认维度为0
        return torch.stack(dec_bboxes), torch.stack(dec_cls)
    

####################
class DecouplingDFLTextDeformableTransformerDecoder(nn.Module):
    """
    Implementation of Deformable Transformer Decoder based on PaddleDetection.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        """Initialize the DeformableTransformerDecoder with the given parameters."""
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
            self, 
            embed,  # decoder embeddings [bs, num_dn + nq, dn_cls_embed = hd（256）] topk_feature (decoder queries可以看成是anchor位置的学习）
            refer_bbox,  # anchor (bs, num_queries + num_dn, 4)
            feats,  # image features [b, 20x20 + 40x40 + 80x80, c]
            shapes,  # feature shapes [nl, 2]
            text, # text feature [b, k, c]
            bbox_head, # self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])
            score_head, # self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
            pos_mlp, # self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2) 
            dn_meta,
            attn_mask=None, # [num_dn + num_queries, num_dn + num_queries]
            padding_mask=None):
        """Perform the forward pass through the entire decoder."""
        output1 = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        # 默认有6层解码器 使用MLP将refer_bbox编码到(bs, num_queries + num_dn, 256)作为位置编码，在自注意力中和类别编码放到一起
        for i, layer in enumerate(self.layers):
            output1, output2 = layer(output1, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox), dn_meta)
            #可不可以像RNN将每个解码器的输出结果进行正负样本分配
            #在每一层decoder中都会去预测相对偏移量，并去更新检测框，得到一个更加精确的检测框预测 
            bbox = bbox_head[i](output2) # 和解码器层数一样，将经过自注意力和交叉注意力的特征映射为坐标偏移量（为什么一共要搞6个？）
            # inverse_sigmoid 逆sigmoid (bs, num_queries + num_dn, 4)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox)) 
            # 训练阶段 类别置信度一直累加 边界框也一直加，但使用sigmoid平衡6个解码器
            if self.training:
                dec_cls.append(score_head[i](output1, text)) # [bs, num_dn + nq, dn_cls_embed = hd（256）] -> [bs, num_dn + nq, nc]
                if i == 0:
                    dec_bboxes.append(refined_bbox) 
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx: # 到最后了
                dec_cls.append(score_head[i](output1, text))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox #更新检测框
        # 将6个结果stack起来，默认维度为0
        return torch.stack(dec_bboxes), torch.stack(dec_cls)
    

class DecouplingDFLDeformableTransformerDecoderLayer(nn.Module):
    """
    Deformable Transformer Decoder Layer inspired by PaddleDetection and Deformable-DETR implementations.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0., act=nn.ReLU(), n_levels=4, n_points=4):
        """Initialize the DeformableTransformerDecoderLayer with the given parameters."""
        super().__init__()

        # Self attention
        self.self_attn1 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Cross attention
        self.cross_attn1 = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.cross_attn2 = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm4 = nn.LayerNorm(d_model)

        # FFN1
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout5 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout6 = nn.Dropout(dropout)
        self.norm5 = nn.LayerNorm(d_model)

        # FFN2
        self.linear3 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout7 = nn.Dropout(dropout)
        self.linear4 = nn.Linear(d_ffn, d_model)
        self.dropout8 = nn.Dropout(dropout)
        self.norm6 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Add positional embeddings to the input tensor, if provided."""
        return tensor if pos is None else tensor + pos

    def forward_ffn1(self, tgt):
        """Perform forward pass through the Feed-Forward Network part of the layer."""
        tgt2 = self.linear2(self.dropout5(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout6(tgt2)
        return self.norm5(tgt)
    
    def forward_ffn2(self, tgt):
        """Perform forward pass through the Feed-Forward Network part of the layer."""
        tgt2 = self.linear4(self.dropout7(self.act(self.linear3(tgt))))
        tgt = tgt + self.dropout8(tgt2)
        return self.norm6(tgt)

    def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None, attn_mask=None, query_pos=None, dn_meta=None):
        """Perform the forward pass through the entire decoder layer."""
    # 类别输出
        # Self attention
        # 前k个特征和dn类别值[bs, num_dn + nq, dn_cls_embed = hd（256）] 和 代表位置refer_bbox->MLP->query_pos (bs,  num_dn+nq, 256) 加起来
        # feat是所有特征
        q = k = self.with_pos_embed(embed, query_pos) #q k = embed + query_pos
        # 使用多头注意力机制，输入包括 q k embed=v attn_mask 让dn和nq之间不能相互注意 这里用embed做v 让类别信息和位置信息进行匹配，并强化类别信息
        if dn_meta is None:
            embed1 = embed
        else:
            dn_pos, _ = torch.split(query_pos, dn_meta['dn_num_split'], dim=1)
            _, top_k_feature = torch.split(embed, dn_meta['dn_num_split'], dim=1)
            embed1 = torch.cat([dn_pos, top_k_feature], 1)
        # 
        tgt = self.self_attn1(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1),
                             attn_mask=attn_mask)[0].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        tgt = self.self_attn2(q.transpose(0, 1), k.transpose(0, 1), embed1.transpose(0, 1),
                             attn_mask=attn_mask)[0].transpose(0, 1)
        embed1 = embed1 + self.dropout2(tgt)
        embed1 = self.norm2(embed1)

        # Cross attention refer_bbox在中插一个维度(bs, num_dn + nq, 1, 4)
        tgt = self.cross_attn1(self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes,
                              padding_mask)
        embed = embed + self.dropout3(tgt)
        embed = self.norm3(embed)

        tgt = self.cross_attn2(self.with_pos_embed(embed1, query_pos), refer_bbox.unsqueeze(2), feats, shapes,
                              padding_mask)
        embed1  = embed1  + self.dropout4(tgt)
        embed1 = self.norm4(embed1)

        # FFN
        return self.forward_ffn1(embed), self.forward_ffn2(embed1)