#!/usr/bin/python
# -*- encoding: utf-8 -*-
import math
from typing import Optional, Dict, Tuple, List, Dict, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers.models.bert.configuration_bert import BertConfig
from transformers.modeling_utils import Conv1D
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

class BertForAutoRegressionDecoder(nn.Module):
    def __init__(
        self,
        config: BertConfig,
        n_layers: int = 1,
        prefix_width: Optional[int] = None,
        attn_window: Optional[int] = None,
    ):
        super(BertForAutoRegressionDecoder, self).__init__()
        self.n_layers = n_layers
        self.prefix_width = prefix_width
        self.attn_window = attn_window

        # Cast a GPT2Config from BERTConfig
        gpt2_config = GPT2Config(
            vocab_size=config.vocab_size, 
            n_positions=config.max_position_embeddings,
            n_embd=config.hidden_size,
            n_layer=config.num_hidden_layers,
            n_head=config.num_attention_heads,
            activation_function=config.hidden_act,
            resid_pdrop=config.hidden_dropout_prob,
            embd_pdrop=config.hidden_dropout_prob,
            attn_pdrop=config.attention_probs_dropout_prob,
            layer_norm_epsilon=config.layer_norm_eps,
            initializer_range=config.initializer_range,
        )

        self.config = gpt2_config

        ar_dec_head = nn.ModuleList(
            [GPT2Block(gpt2_config, layer_idx=i) for i in range(n_layers)]
        )
        # Adjust to Prefix Attn Mask if `_prompt_input_ids` in data_args
        if prefix_width:
            assert prefix_width >= 1
            # get diag ones
            casual_attn_mask = torch.tril(torch.ones((gpt2_config.n_positions, gpt2_config.n_positions), dtype=torch.uint8)).view(
                                                1, 1, gpt2_config.n_positions, gpt2_config.n_positions)
            # Customized casual attention mask with `prefix_width`
            for i in range(0, casual_attn_mask.size(-2)):   # row [0, n_positions]
                casual_attn_mask[0][0][i][:prefix_width+1] = 1
            _block: GPT2Block = None
            for _block in ar_dec_head:
                del _block.attn.bias
                _block.attn.register_buffer("bias", casual_attn_mask)

        if attn_window and attn_window > 0:
            casual_attn_mask = torch.tril(torch.ones((gpt2_config.n_positions, gpt2_config.n_positions), dtype=torch.uint8)).view(
                                                1, 1, gpt2_config.n_positions, gpt2_config.n_positions
                                            )
            # Customized casual attention mask, attention only on cls & tokens within attn_window
            for i in range(attn_window + 1, casual_attn_mask.size(-2)):   # row start from attn_window + 1
                for j in range(1, i-attn_window+1):    # col start from 1
                    casual_attn_mask[0][0][i][j] = 0
            assert attn_window > 0
            _block: GPT2Block = None
            for _block in ar_dec_head:
                del _block.attn.bias
                _block.attn.register_buffer("bias", casual_attn_mask)
        
        self.decoder = ar_dec_head
        # A Dropout is applied before embedding going into model in GPT2 settings
        self.dropout = nn.Dropout(gpt2_config.embd_pdrop)
        # A LN is applied after output hidden states of GPT2Block in GPT2 settings
        self.layernorm = nn.LayerNorm(gpt2_config.n_embd, eps=gpt2_config.layer_norm_epsilon)

        self.decoder.apply(self._init_weights)
    
    def forward(
        self,
        input_embedding: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        d_head_hiddens = input_embedding
        d_head_hiddens = self.dropout(d_head_hiddens) # Dropout on top of embeddings, following GPT2
        for layer in self.decoder:
            layer_out = layer(
                d_head_hiddens,
                attention_mask=attention_mask,
            )
            d_head_hiddens = layer_out[0]
        d_head_hiddens = self.layernorm(d_head_hiddens) # LN after hidden states, following GPT2
        
        return d_head_hiddens

    def _init_weights(self, module):
        """Initialize the weights. Fetch from GPT2PreTrainedModel"""
        if isinstance(module, (nn.Linear, Conv1D)):
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

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if "c_proj" in name and "weight" in name:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.n_layers)))