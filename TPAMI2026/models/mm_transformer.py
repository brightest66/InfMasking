"""
From https://github.com/GeWu-Lab/RollingQ_ICML2025/blob/check/models/mm_transformer.py
Modified by DaiQun
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils.utils import get_pooling


class MTModel(nn.Module):

    def __init__(self, args, n_classes, pooling):
        super(MTModel, self).__init__()
        self.args = args
        edim = args.edim
        self.pooling = pooling
        self.use_single = getattr(args, "use_single", True)
        self.use_masking = args.use_masking
        self.mask_objective = getattr(args, "mask_objective", "infmask")
        if self.use_masking:
            if len(args.modalities) != 2:
                raise ValueError(
                    "MT masking baselines are implemented only for two modalities."
                )
            self.mask_ratio = args.mask_ratio
            self.num_mask = args.num_mask
            self.mask_token = nn.Parameter(torch.zeros(1, 1, edim))
            nn.init.normal_(self.mask_token, std=0.02)
        self.fusion_module = AttnFusion(
            output_dim=n_classes,
            embed_dim=edim,
            num_heads=args.num_heads if hasattr(args, "num_heads") else 1,
            dropout=getattr(args, "MT_attn_dropout", 0.2),
        )
        if self.use_single:
            self.a_head = nn.Linear(edim, n_classes)
            self.v_head = nn.Linear(edim, n_classes)
            self.t_head = nn.Linear(edim, n_classes)

    def random_mask_features(self, x, mask_ratio=None):
        """
        在序列长度维度上进行掩码
        x: [B, L, E] 或 [B, E]
        返回: [B, L, E]（掩码后的序列）
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        B, L, E = x.shape
        dim_feature = 2
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.zeros([B, L], device=x.device)
        mask[:, :len_keep] = 1
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask = mask.unsqueeze(dim_feature)
        mask_tokens = self.mask_token.expand(B, L, -1)
        x_masked = x * mask + mask_tokens * (1 - mask)
        return x_masked

    def forward(self, X):
        X_list = []
        fm_list = []
        for modality, pooling in self.pooling.items():
            if modality not in self.args.modalities:
                continue
            if pooling == "second":
                X_list.append(X[self.args.modalities.index(modality)][0])
                if self.use_single:
                    fm_list.append(X[self.args.modalities.index(modality)][1])
            else:
                X_list.append(X[self.args.modalities.index(modality)])
                if self.use_single:
                    fm_list.append(
                        get_pooling(pooling)(X[self.args.modalities.index(modality)])
                    )
        m1, m2 = X_list
        masked_fused_list = None
        if self.use_masking:
            masked_fused_list = []
            for _ in range(self.num_mask):
                mask_token_a = self.random_mask_features(m1, mask_ratio=self.mask_ratio)
                mask_token_v = self.random_mask_features(m2, mask_ratio=self.mask_ratio)
                _, _, mask_out = self.fusion_module(mask_token_a, mask_token_v)
                masked_fused_list.append(mask_out[-1])
        _, _, out = self.fusion_module(m1, m2)
        output = {"out": out, "masked_fused_list": masked_fused_list}
        if self.use_single:
            fm_1, fm_2 = fm_list
        if self.use_single:
            output["out_a"] = self.a_head(fm_1)
            output["out_v"] = self.v_head(fm_2)
        return output


class MT2Model(nn.Module):

    def __init__(self, args, n_classes, pooling):
        super(MT2Model, self).__init__()
        self.args = args
        edim = args.edim
        self.edim = edim
        self.pooling = pooling
        self.use_single = getattr(args, "use_single", True)
        self.use_masking = args.use_masking
        self.mask_objective = getattr(args, "mask_objective", "infmask")
        if self.use_masking:
            if len(args.modalities) != 2:
                raise ValueError(
                    "MT2 masking baselines are implemented only for two modalities."
                )
            self.mask_ratio = args.mask_ratio
            self.num_mask = args.num_mask
            self.mask_token = nn.Parameter(torch.zeros(1, 1, edim))
            nn.init.normal_(self.mask_token, std=0.02)
        self.fusion_module = TransformerFusion(
            embed_dim=edim,
            output_dim=n_classes,
            num_layers=getattr(args, "MT2_num_layers", 2),
            num_heads=args.num_heads if hasattr(args, "num_heads") else 1,
            mlp_ratio=getattr(args, "MT2_mlp_ratio", 4.0),
            dropout=getattr(args, "MT2_dropout", getattr(args, "fusion_dropout", 0.2)),
        )
        if self.use_single:
            self.a_head = nn.Linear(edim, n_classes)
            self.v_head = nn.Linear(edim, n_classes)
            self.t_head = nn.Linear(edim, n_classes)

    def random_mask_features(self, x, mask_ratio=None):
        """
        在序列长度维度上进行掩码
        x: [B, L, E] 或 [B, E]
        返回: [B, L, E]（掩码后的序列）
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        B, L, E = x.shape
        dim_feature = 2
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.zeros([B, L], device=x.device)
        mask[:, :len_keep] = 1
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask = mask.unsqueeze(dim_feature)
        mask_tokens = self.mask_token.expand(B, L, -1)
        x_masked = x * mask + mask_tokens * (1 - mask)
        return x_masked

    def forward(self, X):
        X_list = []
        fm_list = []
        for modality, pooling in self.pooling.items():
            if modality not in self.args.modalities:
                continue
            if pooling == "second":
                X_list.append(X[self.args.modalities.index(modality)][0])
                if self.use_single:
                    fm_list.append(X[self.args.modalities.index(modality)][1])
            else:
                X_list.append(X[self.args.modalities.index(modality)])
                if self.use_single:
                    fm_list.append(
                        get_pooling(pooling)(X[self.args.modalities.index(modality)])
                    )
        m1, m2 = X_list
        masked_fused_list = None
        if self.use_masking:
            masked_fused_list = []
            for _ in range(self.num_mask):
                mask_token_a = self.random_mask_features(m1, mask_ratio=self.mask_ratio)
                mask_token_v = self.random_mask_features(m2, mask_ratio=self.mask_ratio)
                _, _, mask_out = self.fusion_module(mask_token_a, mask_token_v)
                masked_fused_list.append(mask_out[-1])
        _, _, out = self.fusion_module(m1, m2)
        output = {"out": out, "masked_fused_list": masked_fused_list}
        if self.use_single:
            fm_1, fm_2 = fm_list
        if self.use_single:
            output["out_a"] = self.a_head(fm_1)
            output["out_v"] = self.v_head(fm_2)
        return output


class TransformerFusion(nn.Module):
    """Two-layer Transformer fusion module.

    Expects x/y in shape [B, L, E] (ble). Uses CLS pooling for fused representation.
    Keeps the same return convention as AttnFusion: (x, y, [logits, attn_matrix, infos, cls_embed]).
    """

    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.FC = nn.Linear(embed_dim, output_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        bs = x.shape[0]
        cls_tokens = self.cls_token.expand(bs, -1, -1)
        seq = torch.cat([cls_tokens, x, y], dim=1)
        out_seq = self.encoder(seq)
        cls_embed = out_seq[:, 0]
        logits = self.FC(cls_embed)
        attn_matrix = None
        infos = None
        return (x, y, [logits, attn_matrix, infos, cls_embed])


class AttnFusion(nn.Module):

    def __init__(self, embed_dim=512, output_dim=100, num_heads=1, dropout=0.2):
        super(AttnFusion, self).__init__()
        self.embed_dim = embed_dim
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.FC = nn.Linear(embed_dim, output_dim)
        self._cls_token()

    def _cls_token(self):
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, y):
        bs = x.shape[0]
        assert x.shape[0] == y.shape[0]
        cls_tokens = self.cls_token.expand(bs, -1, -1)
        seq = torch.cat([cls_tokens, x, y], dim=1)
        out_seq, attn_matrix, infos = self.attn(seq, seq, seq)
        cls_embed = out_seq[:, 0]
        out = self.FC(cls_embed)
        return (x, y, [out, attn_matrix, infos, cls_embed])


"\nFrom https://github.com/yaohungt/Multimodal-Transformer/blob/master/modules/multihead_attention.py\n"


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1000000000.0)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return (output, attn)


"\nFrom https://github.com/jadore801120/attention-is-all-you-need-pytorch\n"


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(
        self, embed_dim, num_heads, bias=True, d_k=None, d_v=None, dropout=0.0
    ):
        super().__init__()
        d_model = embed_dim
        n_head = num_heads
        if d_k is None:
            d_k = d_model
        if d_v is None:
            d_v = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=bias)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=bias)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=bias)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=bias)
        self.attention = ScaledDotProductAttention(
            temperature=d_k**0.5, attn_dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-06)

    def forward(self, q, k, v, mask=None, affine=None):
        d_k, d_v, n_head = (self.d_k, self.d_v, self.n_head)
        sz_b, len_q, len_k, len_v = (q.size(0), q.size(1), k.size(1), v.size(1))
        qn = self.w_qs(q)
        kn = self.w_ks(k)
        vn = self.w_vs(v)
        if affine is not None:
            "\n                y = R @ x\n                but here, qcls -> [batch, edim] = stack(x^T)\n                out: [B, edim] <--> y^T = x^T @ R^T\n"
            qcls = qn[:, 0, :]
            qcls_affined = qcls @ affine.T.to(qcls.device)
            qn[:, 0, :] = qcls_affined
        q = qn.view(sz_b, len_q, n_head, d_k)
        k = kn.view(sz_b, len_k, n_head, d_k)
        v = vn.view(sz_b, len_v, n_head, d_v)
        q, k, v = (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        if mask is not None:
            mask = mask.unsqueeze(1)
        out, attn = self.attention(q, k, v, mask=mask)
        if self.n_head == 1:
            attn = attn.squeeze(2)
        out = out.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        out = self.dropout(self.fc(out))
        infos = {"q": qn, "k": kn, "v": vn}
        return (out, attn, infos)
