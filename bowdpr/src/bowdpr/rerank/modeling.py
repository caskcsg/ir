#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
SentenceTransformers Wrapper Model Class for contrastive loss.

@Time    :   2023/11/06
@Author  :   Ma (Ma787639046@outlook.com)
'''
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
from torch import nn, Tensor

from transformers import (
    AutoModelForSequenceClassification, 
    PreTrainedModel,
    BatchEncoding,
    PretrainedConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from .arguments import ModelArguments, DataArguments, TrainingArguments


@dataclass
class SequenceClassifierOutputWithLogs(SequenceClassifierOutput):
    logs: Optional[Dict[str, any]] = None

class CrossEncoder(nn.Module):
    """
    A simple warpper for `AutoModelForSequenceClassification` for reranking task
    """
    def __init__(
        self, 
        lm: PreTrainedModel,
        model_args: ModelArguments, 
        data_args: DataArguments,
        training_args: TrainingArguments,
        *args, 
        **kwargs
    ):
        super().__init__()
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.lm = lm
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.register_buffer(
            'target_label',
            torch.zeros(self.training_args.per_device_train_batch_size, dtype=torch.long)
        )
        
        self.sigmoid = None
        if model_args.sigmoid_normalize:
            self.sigmoid = nn.Sigmoid()
    
    def forward(self, batch: BatchEncoding):
        lm_out: SequenceClassifierOutput = self.lm(**batch, return_dict=True)
        if self.sigmoid is not None:
            lm_out.logits = self.sigmoid(lm_out.logits)

        if self.training:   # Training with Listwise Ranking Loss
            logits = lm_out.logits / self.model_args.temperature
            scores = logits.view(
                self.training_args.per_device_train_batch_size,
                self.data_args.train_n_passages
            )
            loss = self.cross_entropy(scores, self.target_label)

            return SequenceClassifierOutputWithLogs(
                loss=loss,
                logits=logits,
                hidden_states=lm_out.hidden_states,
                attentions=lm_out.attentions,
                logs={"temperature": self.model_args.temperature}
            )
        else:   # Evaluating or Inferencing
            return SequenceClassifierOutputWithLogs(**lm_out)
    
    @classmethod
    def from_pretrained(
        cls, 
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
        *args, **kwargs
    ):
        hf_model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        model = cls(hf_model, model_args, data_args, training_args)
        return model

    @classmethod
    def from_config(
            cls,
            config: PretrainedConfig,
            model_args: ModelArguments,
            data_args: DataArguments,
            training_args: TrainingArguments,
    ):
        hf_model = AutoModelForSequenceClassification.from_config(config)
        model = cls(hf_model, model_args, data_args, training_args)
        return model

    def save_pretrained(self, output_dir: str, *args, **kwargs):
        self.lm.save_pretrained(output_dir, *args, **kwargs)
