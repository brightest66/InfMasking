"""Historical label-proxy InfMasking loss; numerical behavior is preserved."""

import torch
import torch.nn as nn


def _validate_mask_outputs(mask_out_list):
    if not mask_out_list:
        raise ValueError("A masking objective requires at least one masked embedding.")


def infmask_loss(fc, mask_out_list, label, ratio=1.0):
    """计算InfMask损失"""
    _validate_mask_outputs(mask_out_list)
    proxies = fc.weight  # [C, D]

    # 获取基本形状
    B, E = mask_out_list[0].shape
    T = len(mask_out_list)

    # 拼接成 [T*B, E]
    k_pos = torch.cat(mask_out_list, dim=0)

    # ======== compute mean & covariance ============
    k_new = k_pos.view(T, B, -1)  # [T, B, E]

    k_means = torch.mean(k_new, dim=0, keepdim=True)  # [1, B, E]
    value_minus_mean = k_new - k_means  # [T, B, E]

    v1 = value_minus_mean.permute(1, 0, 2)  # [B, T, E]
    v2 = value_minus_mean.permute(1, 2, 0)  # [B, E, T]

    k_sigma = torch.bmm(v2, v1) / T  # [B, E, E]
    k_sigma = k_sigma * ratio  # scale

    k_means = k_means.squeeze(0)  # [B, E]

    # =========== compute logits =============
    part_1 = torch.einsum("be,ce->bc", k_means, proxies)  # [B,C]
    temp = torch.einsum("bee,ce->bce", k_sigma, proxies)  # [B,C,E]
    part_2 = 0.5 * torch.einsum("ce,bce->bc", proxies, temp)

    logits = part_1 + part_2

    # =========== CE Loss =====================
    ce_loss = nn.CrossEntropyLoss()(logits, label)

    # =========== mask loss ====================
    w_c = proxies[label]  # [B,D]
    sigma_w_c = torch.bmm(k_sigma, w_c.unsqueeze(-1)).squeeze(-1)
    w_sigma_w = torch.einsum("be,be->b", w_c, sigma_w_c)
    mask_loss = 0.5 * torch.mean(w_sigma_w)

    return ce_loss + mask_loss
