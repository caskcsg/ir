#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Model Implementation

@Author  :   Ma (Ma787639046@outlook.com)
'''
import os
import warnings
from typing import Optional, Dict, Tuple, List, Dict, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForMaskedLM, BatchEncoding
from transformers.models.bert.modeling_bert import BertModel
from transformers.modeling_outputs import MaskedLMOutput
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from transformers.models.auto.modeling_auto import AutoModelForMaskedLM

from .arguments import DataTrainingArguments, ModelArguments, BoWPredictionPreTrainingArguments as TrainingArguments
from .mae_decoder.ae_dec import BertForAutoEncodingDecoder
from .mae_decoder.ar_dec import BertForAutoRegressionDecoder

import logging
logger = logging.getLogger(__name__)

def pooling(
        hidden_states: Tuple[torch.Tensor]=None, 
        attention_mask: torch.Tensor=None,
        pooling_strategy: str='cls',):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation without BERT/RoBERTa's MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    last_hidden = hidden_states[-1]
    if pooling_strategy == 'cls':
        return last_hidden[:, 0]
    elif pooling_strategy == "avg":
        return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
    elif pooling_strategy == "avg_first_last":
        first_hidden = hidden_states[0]
        last_hidden = hidden_states[-1]
        return ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
    elif pooling_strategy == "avg_top2":
        second_last_hidden = hidden_states[-2]
        last_hidden = hidden_states[-1]
        return ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
    else:
        raise NotImplementedError()

@dataclass
class MaskedLMOutputWithLogs(MaskedLMOutput):
    logs: Optional[Dict[str, any]] = None

class BertForBoWPrediction(nn.Module):
    _keys_to_ignore_on_save = None

    def __init__(
        self,
        bert: BertModel,
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        train_args: TrainingArguments,
    ):
        super(BertForBoWPrediction, self).__init__()
        self.lm = bert
        self.config = bert.config
        self.cross_entropy = nn.CrossEntropyLoss()
        self.model_args: ModelArguments = model_args
        self.train_args: TrainingArguments = train_args
        self.data_args: DataTrainingArguments = data_args

        # Scaling factor for BoW Loss
        self.scaling_factor: float = 1.0

        # Init a normal transformers-encoder based head
        if model_args.use_enc_head:
            self.c_head = BertForAutoEncodingDecoder(bert.config, n_layers=model_args.n_enc_head_layers)
        
        # Init a transformers-decoder based GPT2 Blocks as head 
        if model_args.use_dec_head:
            self.d_head = BertForAutoRegressionDecoder(bert.config, n_layers=model_args.n_dec_head_layers, prefix_width=None, attn_window=model_args.attn_window)
        
    
    def forward(
        self, 
        encoder_input: Optional[BatchEncoding] = None, 
        ae_dec_head_input: Optional[BatchEncoding] = None, 
        ar_dec_head_input: Optional[BatchEncoding] = None, 
        bow_weights: Optional[torch.Tensor] = None,
        **kwargs
    ):
        lm_out: MaskedLMOutput = self.lm.forward(
            **encoder_input,
            output_hidden_states=True,
            return_dict=True
        )

        reps: torch.Tensor = pooling(
            hidden_states=lm_out.hidden_states,
            attention_mask=encoder_input['attention_mask'],
            pooling_strategy=self.model_args.pooling_strategy,
        )

        logs = dict()

        # Encoder mlm loss
        loss = lm_out.loss
        logs["bert_mlm_loss"] = lm_out.loss.item()
        
        if self.model_args.use_enc_head:
            """ Auto-Encoding Decoder """
            # Get the embedding of decoder inputs
            enc_head_emb = self.lm.bert.embeddings(input_ids=ae_dec_head_input['input_ids'])
            enc_head_attn_mask = self.lm.get_extended_attention_mask(
                                        ae_dec_head_input['attention_mask'],
                                        ae_dec_head_input['attention_mask'].shape,
                                        ae_dec_head_input['attention_mask'].device
                                    )
            # Concat cls-hiddens of span A & embedding of span B
            c_head_hiddens = torch.cat([reps.unsqueeze(1), enc_head_emb[:, 1:]], dim=1)
            
            if self.model_args.enable_enc_head_mlm:
                c_head_hiddens = self.c_head.forward(
                    input_embedding=c_head_hiddens,
                    attention_mask=enc_head_attn_mask,
                )

                # add head-layer mlm loss
                enc_head_mlm_loss = self.mlm_loss(c_head_hiddens, ae_dec_head_input['labels']) * self.model_args.enc_head_mlm_coef
                logs["enc_head_mlm_loss"] = enc_head_mlm_loss.item()
                loss += enc_head_mlm_loss

        if self.model_args.use_dec_head:
            """ Auto-Regression Decoder """
            dec_head_emb = self.lm.bert.embeddings(input_ids=ar_dec_head_input['input_ids'])
            dec_head_attn_mask = self.lm.get_extended_attention_mask(
                                        ar_dec_head_input['attention_mask'],
                                        ar_dec_head_input['attention_mask'].shape,
                                        ar_dec_head_input['attention_mask'].device
                                    )
            # d_head_hiddens: [bz, tgt_len, hid], (CLS + tgt emb[1:]) at dim 1
            d_head_hiddens = torch.cat([reps.unsqueeze(1), dec_head_emb[:, 1:]], dim=1)

            if self.model_args.enable_dec_head_loss:
                d_head_hiddens = self.d_head.forward(d_head_hiddens, dec_head_attn_mask)
                dec_head_casl_loss = self.casual_loss(d_head_hiddens, ar_dec_head_input["labels"]) * self.model_args.dec_head_coef
                logs["dec_head_casual_loss"] = dec_head_casl_loss.item()
                loss += dec_head_casl_loss
        
        if self.model_args.enable_dec_bow_loss:
            """ BoW Prediction Decoding """
            if self.model_args.pooling_strategy == 'cls':
                rep_logits = lm_out.logits[:, 0]
            elif self.model_args.pooling_strategy == "avg":
                rep_logits = ((lm_out.logits * encoder_input['attention_mask'].unsqueeze(-1)).sum(1) / encoder_input['attention_mask'].sum(-1).unsqueeze(-1))
            else:
                raise NotImplementedError()

            dec_bow_loss = self.bow_loss(rep_logits, bag_word_weight=bow_weights) * self.model_args.dec_bow_loss_coef * self.scaling_factor
            loss += dec_bow_loss

            logs["dec_bow_loss"] = dec_bow_loss.item()
            if self.train_args.bow_factor_cosine_decay_to > 0:
                logs["dec_bow_coef"] = self.model_args.dec_bow_loss_coef * self.scaling_factor

        return MaskedLMOutputWithLogs(
            loss=loss,
            logits=lm_out.logits,
            hidden_states=lm_out.hidden_states,
            attentions=lm_out.attentions,
            logs=logs,
        )

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
    
    def bow_loss(self, reps, bag_word_weight):
        input = F.log_softmax(reps, dim=-1)
        bow_loss = torch.mean(-torch.sum(bag_word_weight * input, dim=1))
        return bow_loss
    
    @staticmethod
    def _load_extra_weights(model: torch.nn.Module, model_name_or_path: str):
        head_weights_path = os.path.join(model_name_or_path, 'model.pt')
        if os.path.exists(head_weights_path):
            warnings.warn(f"loading extra weights from local files: {head_weights_path}")
            state_dict = torch.load(head_weights_path, map_location="cpu")
            load_results = model.load_state_dict(state_dict, strict=False)
            # release memory
            del state_dict

    def load(self, *args, **kwargs):
        model_name_or_path = args[0]
        # Load BERT Encoder
        hf_model = BertForMaskedLM.from_pretrained(*args, **kwargs)
        load_results = self.lm.load_state_dict(hf_model.state_dict())
        self._load_extra_weights(self, model_name_or_path)
        # release memory
        del hf_model

    @classmethod
    def from_pretrained(
        cls, 
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        train_args: TrainingArguments,
        *args, **kwargs
    ):
        model_name_or_path = args[0]
        # Load BERT Encoder
        hf_model = AutoModelForMaskedLM.from_pretrained(*args, **kwargs)

        # Init model
        model = cls(hf_model, model_args, data_args, train_args)
        if os.path.exists(os.path.join(model_name_or_path, 'model.pt')):
            warnings.warn('loading extra weights from local files')
            model_dict = torch.load(os.path.join(model_name_or_path, 'model.pt'), map_location="cpu")
            model.load_state_dict(model_dict, strict=False)
        
        return model

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
        if model_dict:
            torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))
