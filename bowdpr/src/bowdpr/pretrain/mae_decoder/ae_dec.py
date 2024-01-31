#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertLayer

class BertForAutoEncodingDecoder(nn.Module):
    def __init__(
        self,
        config: BertConfig,
        n_layers: int = 1,
    ):
        super(BertForAutoEncodingDecoder, self).__init__()
        self.config = config
        self.n_layers = n_layers
        self.decoder = nn.ModuleList(
                [BertLayer(config) for _ in range(n_layers)]
            )
        self.decoder.apply(self._init_weights)
    
    def forward(
        self,
        input_embedding: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        c_head_hiddens = input_embedding
        for layer in self.decoder:
            layer_out = layer(
                c_head_hiddens,
                attention_mask,
            )
            c_head_hiddens = layer_out[0]
        
        return c_head_hiddens

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
