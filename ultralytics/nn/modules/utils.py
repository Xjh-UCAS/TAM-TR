# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Module utils."""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import uniform_

__all__ = 'multi_scale_deformable_attn_pytorch', 'inverse_sigmoid'


def _get_clones(module, n):
    """Create a list of cloned modules from the given module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def bias_init_with_prob(prior_prob=0.01):
    """Initialize conv/fc bias value according to a given probability value."""
    return float(-np.log((1 - prior_prob) / prior_prob))  # return bias_init


def linear_init_(module):
    """Initialize the weights and biases of a linear module."""
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    if hasattr(module, 'bias') and module.bias is not None:
        uniform_(module.bias, -bound, bound)


def inverse_sigmoid(x, eps=1e-5):
    """Calculate the inverse sigmoid function for a tensor."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def multi_scale_deformable_attn_pytorch(value: torch.Tensor, value_spatial_shapes: torch.Tensor,
                                        sampling_locations: torch.Tensor,
                                        attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Multi-scale deformable attention.

    https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    # value = [bs, 20x20 + 40x40 + 80x80, 8, 256/8] value_shapes = [20x20 + 40x40 + 80x80] 
    # sampling_locations = [bs, query_length, 8, 3 * 4, 2] attention_weights = [bs, query_length, 8, n_levels（3）, 4, 2] 
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    # 按 value_spatial_shapes 的大小，将 value 按层拆分为不同的分辨率特征
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    # F.grid_sample 要求采样点的坐标范围在[-1,1],这里将采样点坐标从[0,1]到[-1,1]
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = (value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_))
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        # 函数解析：输入特征图，形状为[N,C,H,W]
        # 采样网格，形状为[N,Hout,Wout,2]
        # 输出为，[N,C,Hout,Wout]
        sampling_value_l_ = F.grid_sample(value_l_,
                                          sampling_grid_l_,
                                          mode='bilinear',
                                          padding_mode='zeros',
                                          align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs*num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(bs * num_heads, 1, num_queries,
                                                                  num_levels * num_points)
    #(bs×num_heads,embed_dims,num_queries,num_levels×num_points)
    # 将num_levels×num_points求和
    output = ((torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(
        bs, num_heads * embed_dims, num_queries))
    return output.transpose(1, 2).contiguous()


def multi_scale_deformable_attn_pytorch_cls(value: torch.Tensor, value_spatial_shapes: torch.Tensor,
                                        sampling_locations: torch.Tensor,
                                        attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Multi-scale deformable attention.

    https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    # value = [bs, 20x20 + 40x40 + 80x80, 8, 256/8] value_shapes = [20x20 + 40x40 + 80x80] 
    # sampling_locations = [bs, query_length, 8, 3 * 4, 2] attention_weights = [bs, query_length, 8, n_levels（3）, 4 
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points = attention_weights.shape
    # 按 value_spatial_shapes 的大小，将 value 按层拆分为不同的分辨率特征
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    # F.grid_sample 要求采样点的坐标范围在[-1,1],这里将采样点坐标从[0,1]到[-1,1]
    sampling_grids = 2 * sampling_locations - 1
    grids_list = torch.split(sampling_grids, [2, 4, 6], dim=-2)
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = (value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_))
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = grids_list[level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        # 函数解析：输入特征图，形状为[N,C,H,W]
        # 采样网格，形状为[N,Hout,Wout,2]
        # 输出为，[N,C,Hout,Wout]
        sampling_value_l_ = F.grid_sample(value_l_,
                                          sampling_grid_l_,
                                          mode='bilinear',
                                          padding_mode='zeros',
                                          align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs*num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(bs * num_heads, 1, num_queries,
                                                                  num_levels * num_points)
    #(bs×num_heads,embed_dims,num_queries,num_levels×num_points)
    # 将num_levels×num_points求和
    output = ((torch.cat(sampling_value_list, dim=-1) * attention_weights).sum(-1).view(
        bs, num_heads * embed_dims, num_queries))
    return output.transpose(1, 2).contiguous()


def multi_scale_deformable_attn_pytorch_box(value: torch.Tensor, value_spatial_shapes: torch.Tensor,
                                        sampling_locations: torch.Tensor,
                                        attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Multi-scale deformable attention.

    https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    # value = [bs, 20x20 + 40x40 + 80x80, 8, 256/8] value_shapes = [20x20 + 40x40 + 80x80] 
    # sampling_locations = [bs, query_length, 8, 3 * 4, 2] attention_weights = [bs, query_length, 8, n_levels（3）, 4, 2] 
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points = attention_weights.shape
    # 按 value_spatial_shapes 的大小，将 value 按层拆分为不同的分辨率特征
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    # F.grid_sample 要求采样点的坐标范围在[-1,1],这里将采样点坐标从[0,1]到[-1,1]
    sampling_grids = 2 * sampling_locations - 1
    grids_list = torch.split(sampling_grids, [6, 4, 2], dim=-2)
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = (value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_))
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = grids_list[level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        # 函数解析：输入特征图，形状为[N,C,H,W]
        # 采样网格，形状为[N,Hout,Wout,2]
        # 输出为，[N,C,Hout,Wout]
        sampling_value_l_ = F.grid_sample(value_l_,
                                          sampling_grid_l_,
                                          mode='bilinear',
                                          padding_mode='zeros',
                                          align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs*num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(bs * num_heads, 1, num_queries,
                                                                  num_levels * num_points)
    #(bs×num_heads,embed_dims,num_queries,num_levels×num_points)
    # 将num_levels×num_points求和
    output = ((torch.cat(sampling_value_list, dim=-1) * attention_weights).sum(-1).view(
        bs, num_heads * embed_dims, num_queries))
    return output.transpose(1, 2).contiguous()