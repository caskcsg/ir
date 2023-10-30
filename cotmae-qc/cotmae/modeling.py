#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Model Implementation

@Author  :   Ma (Ma787639046@outlook.com)
'''
import os
import copy
import math
import warnings
from typing import Optional, Dict
from dataclasses import dataclass
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from transformers import BertForMaskedLM
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertLayer, BertPreTrainedModel, BertModel
from transformers.modeling_outputs import MaskedLMOutput
from arguments import DataTrainingArguments, ModelArguments, CotMAEPreTrainingArguments as TrainingArguments
from transformers.modeling_utils import Conv1D, PreTrainedModel, PretrainedConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.auto.modeling_auto import AutoModelForMaskedLM
import logging
logger = logging.getLogger(__name__)

@dataclass
class MaskedLMOutputWithLogs(MaskedLMOutput):
    logs: Optional[Dict[str, any]] = None

class BertForCotMAE(nn.Module):
    _keys_to_ignore_on_save = None

    def __init__(
        self,
        bert: BertModel,
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        train_args: TrainingArguments,
    ):
        super(BertForCotMAE, self).__init__()
        self.lm = bert
        self.config = bert.config
        self.cross_entropy = nn.CrossEntropyLoss()
        self.model_args: ModelArguments = model_args
        self.train_args: TrainingArguments = train_args
        self.data_args: DataTrainingArguments = data_args

        if self.model_args.freeze_bert:
            for param in self.lm.parameters():
                param.requires_grad = False

        # Init a normal transformers-encoder based head
        if model_args.use_enc_head:
            self.c_head = nn.ModuleList(
                [BertLayer(bert.config) for _ in range(model_args.n_enc_head_layers)]
            )
            self.c_head.apply(self._init_weights)
        
        # Init a transformers-decoder based GPT2 Blocks as head 
        if model_args.use_dec_head:
            self.d_head, self.d_head_drop, self.d_head_ln_f = self.build_ar_dec_head(bert.config, data_args, model_args)
            self.d_head.apply(self._init_weights)

        # Contrastive Learning Target Labels, a diag matrix
        contrast_bsz = train_args.per_device_train_batch_size * self._world_size()
        target_diag = torch.arange(contrast_bsz, dtype=torch.long)
        self.register_buffer('target_diag', target_diag, persistent=False)


    @staticmethod
    def build_ar_dec_head(config, data_args, model_args):
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
        ar_dec_head = nn.ModuleList(
            [GPT2Block(gpt2_config, layer_idx=i) for i in range(model_args.n_dec_head_layers)]
        )
        # Adjust to Prefix Attn Mask if `_prompt_input_ids` in data_args
        if hasattr(data_args, 'prefix_width'):
            assert data_args.prefix_width >= 1
            # get diag ones
            casual_attn_mask = torch.tril(torch.ones((gpt2_config.n_positions, gpt2_config.n_positions), dtype=torch.uint8)).view(
                                                1, 1, gpt2_config.n_positions, gpt2_config.n_positions)
            # Customized casual attention mask with `prefix_width`
            for i in range(0, casual_attn_mask.size(-2)):   # row [0, n_positions]
                casual_attn_mask[0][0][i][:data_args.prefix_width+1] = 1
            _block: GPT2Block = None
            for _block in ar_dec_head:
                del _block.attn.bias
                _block.attn.register_buffer("bias", casual_attn_mask)

        if model_args.attn_window != -1:
            casual_attn_mask = torch.tril(torch.ones((gpt2_config.n_positions, gpt2_config.n_positions), dtype=torch.uint8)).view(
                                                1, 1, gpt2_config.n_positions, gpt2_config.n_positions
                                            )
            # Customized casual attention mask, attention only on cls & tokens within model_args.attn_window
            for i in range(model_args.attn_window + 1, casual_attn_mask.size(-2)):   # row start from model_args.attn_window + 1
                for j in range(1, i-model_args.attn_window+1):    # col start from 1
                    casual_attn_mask[0][0][i][j] = 0
            assert model_args.attn_window > 0
            _block: GPT2Block = None
            for _block in ar_dec_head:
                del _block.attn.bias
                _block.attn.register_buffer("bias", casual_attn_mask)
        
        # A Dropout is applied before embedding going into model in GPT2 settings
        ar_dec_head_dropout_layer = nn.Dropout(gpt2_config.embd_pdrop)
        # A LN is applied after output hidden states of GPT2Block in GPT2 settings
        ar_dec_head_ln_layer = nn.LayerNorm(gpt2_config.n_embd, eps=gpt2_config.layer_norm_epsilon)
        
        return ar_dec_head, ar_dec_head_dropout_layer, ar_dec_head_ln_layer
    

    def forward(self, **model_input):
        with nullcontext() if not self.model_args.freeze_bert else torch.no_grad():
            lm_out: MaskedLMOutput = self.lm.forward(
                input_ids = model_input['input_ids'],
                attention_mask = model_input['attention_mask'],
                labels=model_input['labels'],
                output_hidden_states=True,
                return_dict=True
            )

        cls_hiddens = lm_out.hidden_states[-1][:, 0]

        logs = dict()

        # add last layer mlm loss
        loss = 0.0
        if not self.model_args.disable_bert_mlm_loss:
            loss = lm_out.loss
            logs["bert_mlm_loss"] = lm_out.loss.item()
        
        if self.model_args.use_enc_head:
            # Get the embedding of decoder inputs
            enc_head_emb = self.lm.bert.embeddings(input_ids=model_input['enc_head_input_ids'])
            enc_head_attn_mask = self.lm.get_extended_attention_mask(
                                        model_input['enc_head_attention_mask'],
                                        model_input['enc_head_attention_mask'].shape,
                                        model_input['enc_head_attention_mask'].device
                                    )
            # Concat cls-hiddens of span A & embedding of span B
            c_head_hiddens = torch.cat([cls_hiddens.unsqueeze(1), enc_head_emb[:, 1:]], dim=1)

            # Detach grad if freeze bert
            if self.model_args.freeze_bert:
                c_head_hiddens = c_head_hiddens.detach()
            
            for layer in self.c_head:
                layer_out = layer(
                    c_head_hiddens,
                    enc_head_attn_mask,
                )
                c_head_hiddens = layer_out[0]
            
            if self.model_args.enable_enc_head_mlm:
                # add head-layer mlm loss
                enc_head_mlm_loss = self.mlm_loss(c_head_hiddens, model_input['enc_head_labels']) \
                                        * self.model_args.enc_head_mlm_coef
                logs["enc_head_mlm_loss"] = enc_head_mlm_loss.item()
                loss += enc_head_mlm_loss

        if self.model_args.use_dec_head:
            dec_head_emb = self.lm.bert.embeddings(input_ids=model_input['dec_head_input_ids'])
            dec_head_attn_mask = self.lm.get_extended_attention_mask(
                                        model_input['dec_head_attention_mask'],
                                        model_input['dec_head_attention_mask'].shape,
                                        model_input['dec_head_attention_mask'].device
                                    )
            # d_head_hiddens: [bz, tgt_len, hid], (CLS + tgt emb[1:]) at dim 1
            d_head_hiddens = torch.cat([cls_hiddens.unsqueeze(1), dec_head_emb[:, 1:]], dim=1)

            # Detach grad if freeze bert
            if self.model_args.freeze_bert:
                d_head_hiddens = d_head_hiddens.detach()

            d_head_hiddens = self.d_head_drop(d_head_hiddens) # Dropout on top of embeddings, following GPT2
            for layer in self.d_head:
                layer_out = layer(
                    d_head_hiddens,
                    attention_mask=dec_head_attn_mask,
                )
                d_head_hiddens = layer_out[0]
            d_head_hiddens = self.d_head_ln_f(d_head_hiddens) # LN after hidden states, following GPT2

            if self.model_args.enable_dec_head_loss:
                # labels for generative AR-Dec head is natually its input ids sliced with [..., 1:, :]
                # we will do slice (or logits shift) inside the func `casual_loss`
                # Here we fill the labels (or input ids of AR-Dec) [PAD] area with -100, to avoid loss calculatation
                # on [PAD] area by CrossEntropy loss function
                # Ignore klloss on [PAD]:
                # [cls] xxx xxx xxx [SEP] [PAD] [PAD]
                #   0    0   0   0    0   -100  -100
                # Size [bs, seq_len]
                dec_head_labels = model_input['dec_head_input_ids'].clone()
                dec_head_labels.masked_fill_(~(model_input['dec_head_attention_mask'].bool()), -100)
                # add decoder head layer loss
                dec_head_casl_loss = self.casual_loss(d_head_hiddens, dec_head_labels) \
                                        * self.model_args.dec_head_coef
                logs["dec_head_casual_loss"] = dec_head_casl_loss.item()
                loss += dec_head_casl_loss

        return MaskedLMOutputWithLogs(
            loss=loss,
            logits=lm_out.logits,
            hidden_states=lm_out.hidden_states,
            attentions=lm_out.attentions,
            logs=logs,
        )

    # Compute KL loss
    # inputs: input logits, [bs, logit_dim] or [bs, seq_len, logit_dim]
    # targets: target logits, [bs, logit_dim] or [bs, seq_len, logit_dim]
    # pad_mask: [bs, 1] or [bs, seq_len], should be broadcastable to 
    #           klloss shape [bs, logit_dim] or [bs, seq_len]
    def compute_kl_loss(self, inputs, targets, pad_mask=None):
        kl_loss = F.kl_div(F.log_softmax(inputs, dim=-1), F.softmax(targets, dim=-1), 
                                      reduction='none')
        if kl_loss.dim() > 3:
            raise NotImplementedError()
        elif kl_loss.dim() == 3:
            # Mean with last dim
            kl_loss = torch.mean(kl_loss, dim=-1)
        # pad_mask will fill the pos will zero
        if pad_mask is not None:
            kl_loss.masked_fill_(pad_mask, 0.)
        # Sum Reduce 
        # You can choose whether to use function "sum" and "mean" depending on your task
        kl_loss = kl_loss.sum()
        return kl_loss

    def casual_loss(self, hiddens, labels):
        pred_logits = self.lm.cls(hiddens)
        shift_logits = pred_logits[..., :-1, :].contiguous()    # Only measure the generation between 0~n-1
        shift_labels = labels[..., 1:].contiguous()     # No first token, label=1~n for generation
        loss = self.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        return loss

    def mlm_loss(self, hiddens, labels):
        pred_scores = self.lm.cls(hiddens)
        loss = self.cross_entropy(
            pred_scores.view(-1, self.config.vocab_size),
            labels.view(-1)
        )
        return loss
    
    @staticmethod
    def _world_size():
        return dist.get_world_size() if dist.is_initialized() else 1

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
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.model_args.n_dec_head_layers)))
    
    @staticmethod
    def _load_extra_weights(model: torch.nn.Module, path: str):
        if os.path.exists(os.path.join(path, 'model.pt')):
            warnings.warn('loading extra weights from local files')
            state_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            load_result = model.load_state_dict(state_dict, strict=False)
            # release memory
            del state_dict

    @classmethod
    def from_pretrained(
        cls, 
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        train_args: TrainingArguments,
        *args, **kwargs
    ):
        path = args[0]
        # Load BERT Encoder
        hf_model = AutoModelForMaskedLM.from_pretrained(*args, **kwargs)
        
        # Init model
        model = cls(hf_model, model_args, data_args, train_args)
        model._load_extra_weights(model, path)
        return model

    def load(self, *args, **kwargs):
        path = args[0]
        # Load BERT Encoder
        hf_model = AutoModelForMaskedLM.from_pretrained(*args, **kwargs)
        load_results = self.lm.load_state_dict(hf_model.state_dict())
        self._load_extra_weights(self, path)
        # release memory
        del hf_model

    @classmethod
    def from_config(
            cls,
            config: PretrainedConfig,
            model_args: ModelArguments,
            data_args: DataTrainingArguments,
            train_args: TrainingArguments,
    ):
        hf_model = AutoModelForMaskedLM.from_config(config)
        model = cls(hf_model, model_args, data_args, train_args)
        return model

    def save_pretrained(self, output_dir: str, *args, **kwargs):
        # Save BERT Encoder
        self.lm.save_pretrained(output_dir, *args, **kwargs)
        # Save head weights
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('lm')]
        warnings.warn(f'Omiting {len(hf_weight_keys)} transformer weights')
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))