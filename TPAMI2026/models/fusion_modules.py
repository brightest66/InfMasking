"""Shared encoder / adapter / fusion wrapper adapted from MultiBench."""

import torch.nn as nn


class MultimodalFusionBase(nn.Module):
    def __init__(
        self,
        args,
        encoders,
        input_adapters,
        fusion_module,
        has_padding=False,
        use_enc=True,
    ):
        super(MultimodalFusionBase, self).__init__()
        self.args = args
        self.use_enc = use_enc
        self.encoders = (
            encoders if isinstance(encoders, nn.ModuleDict) else nn.ModuleDict(encoders)
        )
        self.input_adapters = (
            input_adapters
            if isinstance(input_adapters, nn.ModuleDict)
            else nn.ModuleDict(input_adapters)
        )
        self.fusion = fusion_module
        self.has_padding = has_padding
        self.embedding_corruptor = None

    def forward(self, inputs):
        if self.use_enc:
            embeddings = []
            if self.has_padding:
                for i, modality in enumerate(self.args.modalities):
                    m_token = self.encoders[modality]([inputs[0][i], inputs[1][i]])
                    embeddings.append(m_token)
            else:
                for i, modality in enumerate(self.args.modalities):
                    m_token = self.encoders[modality](inputs[i])
                    embeddings.append(m_token)
        else:
            embeddings = inputs
        embeddings = [
            (
                self.input_adapters[modality](embeddings[i])
                if self.input_adapters[modality] is not None
                else embeddings[i]
            )
            for i, modality in enumerate(self.args.modalities)
        ]
        if self.embedding_corruptor is not None:
            embeddings = self.embedding_corruptor(embeddings)
        fused = self.fusion(embeddings)
        return fused
