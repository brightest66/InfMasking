"""Encoders retained from the historical supervised InfMasking implementation."""

import math
from typing import Optional
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from einops import rearrange
from transformers import BertModel


class MLP(torch.nn.Module):
    """Two layered perceptron."""

    def __init__(
        self,
        indim,
        hiddim,
        outdim,
        dropout=False,
        dropoutp=0.1,
        output_each_layer=False,
    ):
        """Initialize two-layered perceptron.

        Args:
            indim (int): Input dimension
            hiddim (int): Hidden layer dimension
            outdim (int): Output layer dimension
            dropout (bool, optional): Whether to apply dropout or not. Defaults to False.
            dropoutp (float, optional): Dropout probability. Defaults to 0.1.
            output_each_layer (bool, optional): Whether to return outputs of each layer as a list. Defaults to False.
        """
        super(MLP, self).__init__()
        self.fc = nn.Linear(indim, hiddim)
        self.fc2 = nn.Linear(hiddim, outdim)
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.dropout = dropout
        self.output_each_layer = output_each_layer
        self.lklu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """Apply MLP to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        output = F.relu(self.fc(x))
        if self.dropout:
            output = self.dropout_layer(output)
        output2 = self.fc2(output)
        if self.dropout:
            output2 = self.dropout_layer(output)
        if self.output_each_layer:
            return [0, x, output, self.lklu(output2)]
        return output2


class GRU(torch.nn.Module):
    """Implements Gated Recurrent Unit (GRU)."""

    def __init__(
        self,
        indim,
        hiddim,
        dropout=False,
        dropoutp=0.1,
        flatten=False,
        has_padding=False,
        last_only=False,
        batch_first=True,
    ):
        """Initialize GRU Module.

        Args:
            indim (int): Input dimension
            hiddim (int): Hidden dimension
            dropout (bool, optional): Whether to apply dropout layer or not. Defaults to False.
            dropoutp (float, optional): Dropout probability. Defaults to 0.1.
            flatten (bool, optional): Whether to flatten output before returning. Defaults to False.
            has_padding (bool, optional): Whether the input has padding or not. Defaults to False.
            last_only (bool, optional): Whether to return only the last output of the GRU. Defaults to False.
            batch_first (bool, optional): Whether to batch before applying or not. Defaults to True.
        """
        super(GRU, self).__init__()
        self.gru = nn.GRU(indim, hiddim, batch_first=True)
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.flatten = flatten
        self.has_padding = has_padding
        self.last_only = last_only
        self.batch_first = batch_first

    def forward(self, x):
        """Apply GRU to input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        if self.has_padding:
            x = pack_padded_sequence(
                x[0], x[1], batch_first=self.batch_first, enforce_sorted=False
            )
            out = self.gru(x)[1][-1]
        elif self.last_only:
            out = self.gru(x)[1][0]
            return out
        else:
            out, l = self.gru(x)
        if self.dropout:
            out = self.dropout_layer(out)
        if self.flatten:
            out = torch.flatten(out, 1)

        return out


class Transformer(nn.Module):
    """Extends nn.Transformer as defined in FactorCL for MultiBench data."""

    def __init__(
        self,
        args,
        n_features: int,
        dim: int,
        max_seq_length: int = 50,
        return_seq: bool = True,
        positional_encoding: bool = False,
        pad_value: Optional[float] = None,
    ):
        """Initialize Transformer object.

        Args:
            n_features: Number of features in the input.
            dim: Dimension which to embed upon / Hidden dimension size.
            max_seq_length: Maximum expected sequence length.
            return_seq: If True, returns a sequence of encoded features.
                Otherwise, returns an embedding.
            positional_encoding: Whether to add positional embedding to input tokens.
            pad_value: Padding values added to the end of each sequence (used to compute attn mask)
        """
        super().__init__()
        self.embed_dim = dim
        self.n_features = n_features
        self.conv = nn.Conv1d(
            n_features, self.embed_dim, kernel_size=1, padding=0, bias=False
        )
        self.return_seq = return_seq
        self.use_positional_embedding = positional_encoding
        self.pad_value = pad_value
        self.positional_embedding = nn.Parameter(
            build_1d_sincos_posemb(max_seq_length, self.embed_dim), requires_grad=False
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=5, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=5)

    def forward(self, x):
        """Apply Transformer to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        # From features to tokens
        x = self.conv(rearrange(x, "b l n -> b n l"))
        x = rearrange(x, "b n l -> b l n")
        if self.use_positional_embedding:
            x = x + self.positional_embedding[:, : x.size(1)]
        # Encode tokens
        x = self.transformer(x)
        # Get the embedding
        if not self.return_seq:
            x = x[:, -1]
        return x


def build_1d_sincos_posemb(max_seq_len: int, embed_dim: int = 1024):
    """Build positional embedding matrix with `batch_first`
    :param max_seq_len: maximum length of the input sequence
    :param embed_dim: dimension of each token
    :return: Tensor with shape (1, max_seq_length, embed_dim)
    """
    if embed_dim % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dim (got dim={:d})".format(embed_dim)
        )
    pe = torch.zeros(max_seq_len, embed_dim)
    position = torch.arange(0, max_seq_len).unsqueeze(1)
    div_term = torch.exp(
        (
            torch.arange(0, embed_dim, 2, dtype=torch.float)
            * -(math.log(10000.0) / embed_dim)
        )
    )
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    pe = pe[None, :, :]
    return pe


class BertEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", max_seq_length=49, freeze=True):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name, output_hidden_states=True)
        self.max_seq_len = max_seq_length
        self.freeze = freeze
        # 冻结参数
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
            self.bert.eval()

    def forward(self, encoded_inputs):
        """
        高效前向传播 (无循环)

        Args:
            encoded_inputs: dict 包含:
                - input_ids: [batch_size, seq_len]
                - attention_mask: [batch_size, seq_len]
                - token_type_ids (可选): [batch_size, seq_len]

        Returns:
            final_features: [batch_size, seq_len, hidden_dim]
        """
        # 设备对齐 (自动匹配模型设备)
        device = next(self.bert.parameters()).device
        batch_inputs = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in encoded_inputs.items()
        }
        # # 1. 动态序列截断 (如果超过max_seq_length)
        if self.max_seq_len > 0 and "input_ids" in batch_inputs:
            seq_len = batch_inputs["input_ids"].size(1)
            if seq_len > self.max_seq_len:
                batch_inputs["input_ids"] = batch_inputs["input_ids"][
                    :, : self.max_seq_len
                ]
                if "attention_mask" in batch_inputs:
                    batch_inputs["attention_mask"] = batch_inputs["attention_mask"][
                        :, : self.max_seq_len
                    ]
                if "token_type_ids" in batch_inputs:
                    batch_inputs["token_type_ids"] = batch_inputs["token_type_ids"][
                        :, : self.max_seq_len
                    ]
        # 2. 核心: 单次前向传播 (无循环)
        # 包含多个属性，last_hidden_state, pooler_output, hidden_states
        with torch.set_grad_enabled(not self.freeze):
            bert_outputs = self.bert(**batch_inputs)

        final_features = (
            bert_outputs.last_hidden_state
        )  # [batch_size, seq_len, hidden_dim]

        # # 4. 可选: 调整序列长度 (填充/截断到固定长度)
        if 0 < self.max_seq_len < final_features.size(1):
            final_features = final_features[:, : self.max_seq_len, :]
        elif self.max_seq_len > 0 and final_features.size(1) < self.max_seq_len:
            pad_size = self.max_seq_len - final_features.size(1)
            padding = torch.zeros(
                final_features.size(0),
                pad_size,
                final_features.size(2),
                device=final_features.device,
                dtype=final_features.dtype,
            )
            final_features = torch.cat([final_features, padding], dim=1)

        return final_features, bert_outputs.pooler_output


class TvFeatureExtractor(nn.Module):
    """自动解包 timm features_only 模型的输出"""

    def __init__(self, model_name, freeze=True):
        super().__init__()
        self.model = torchvision.models.get_model(model_name, weights="DEFAULT")
        modules = list(self.model.children())[:-2]  # 去掉最后的池化和全连接层
        self.feature_extractor = nn.Sequential(*modules)
        self.freeze = freeze
        if freeze:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.feature_extractor.eval()

    def forward(self, x):
        """Apply TvFeatureExtractor to layer input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        return self.feature_extractor(x)
