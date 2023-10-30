# Copyright 2021 Condenser Author All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import os
import warnings

import torch
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from transformers import BertModel, BertConfig, AutoModel, AutoModelForMaskedLM, AutoConfig, PretrainedConfig, \
    RobertaModel, BertForMaskedLM
from transformers.models.bert.modeling_bert import BertPooler, BertOnlyMLMHead, BertPreTrainingHeads, BertLayer
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPooling, MaskedLMOutput
from transformers.models.roberta.modeling_roberta import RobertaLayer

from arguments import DataTrainingArguments, ModelArguments, CoCondenserPreTrainingArguments
from transformers import TrainingArguments
import logging

logger = logging.getLogger(__name__)


class CondenserForPretraining(nn.Module):
    def __init__(
        self,
        bert: BertModel,
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        train_args: TrainingArguments
    ):
        super(CondenserForPretraining, self).__init__()
        self.lm = bert
        self.c_head = nn.ModuleList(
            [BertLayer(bert.config) for _ in range(model_args.n_head_layers)]
        )
        self.c_head.apply(self.lm._init_weights)
        self.cross_entropy = nn.CrossEntropyLoss()

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

    def forward(self, model_input, labels):
        attention_mask = self.lm.get_extended_attention_mask(
            model_input['attention_mask'],
            model_input['attention_mask'].shape,
            model_input['attention_mask'].device
        )

        lm_out: MaskedLMOutput = self.lm(
            **model_input,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        cls_hiddens = lm_out.hidden_states[-1][:, :1]
        skip_hiddens = lm_out.hidden_states[self.model_args.skip_from]

        hiddens = torch.cat([cls_hiddens, skip_hiddens[:, 1:]], dim=1)
        for layer in self.c_head:
            layer_out = layer(
                hiddens,
                attention_mask,
            )
            hiddens = layer_out[0]

        loss = self.mlm_loss(hiddens, labels)
        if self.model_args.late_mlm:
            loss += lm_out.loss

        return loss


    def mlm_loss(self, hiddens, labels):
        pred_scores = self.lm.cls(hiddens)
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.lm.config.vocab_size),
            labels.view(-1)
        )
        return masked_lm_loss

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

class RobertaCondenserForPretraining(CondenserForPretraining):
    def __init__(
            self,
            roberta: RobertaModel,
            model_args: ModelArguments,
            data_args: DataTrainingArguments,
            train_args: TrainingArguments
    ):
        super(CondenserForPretraining, self).__init__()
        self.lm = roberta
        self.c_head = nn.ModuleList(
            [RobertaLayer(roberta.config) for _ in range(model_args.n_head_layers)]
        )
        self.c_head.apply(self.lm._init_weights)
        # self.mlm_head = BertOnlyMLMHead(bert.config)
        self.cross_entropy = nn.CrossEntropyLoss()

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

    def mlm_loss(self, hiddens, labels):
        pred_scores = self.lm.lm_head(hiddens)
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.lm.config.vocab_size),
            labels.view(-1)
        )
        return masked_lm_loss

class CoCondenserForPretraining(CondenserForPretraining):
    def __init__(
            self,
            bert: BertModel,
            model_args: ModelArguments,
            data_args: DataTrainingArguments,
            train_args: CoCondenserPreTrainingArguments
    ):
        super(CoCondenserForPretraining, self).__init__(bert, model_args, data_args, train_args)

        # effective_bsz = train_args.per_device_train_batch_size * self._world_size() * 2
        # target = torch.arange(effective_bsz, dtype=torch.long).view(-1, 2).flip([1]).flatten().contiguous()

        # self.register_buffer(
        #     'co_target', target
        # )

    def _gather_tensor(self, t: Tensor):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        return all_tensors

    def gather_tensors(self, *tt: Tensor):
        tt = [torch.cat(self._gather_tensor(t)) for t in tt]
        return tt

    def forward(self, model_input, labels, grad_cache: Tensor = None, chunk_offset: int = None):
        logs = dict()
        loss = 0.0
        
        attention_mask = self.lm.get_extended_attention_mask(
            model_input['attention_mask'],
            model_input['attention_mask'].shape,
            model_input['attention_mask'].device
        )

        lm_out: MaskedLMOutput = self.lm(
            **model_input,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )

        cls_hiddens = lm_out.hidden_states[-1][:, :1]
        if self.train_args.local_rank > -1 and grad_cache is None:
            co_cls_hiddens = self.gather_tensors(cls_hiddens.squeeze().contiguous())[0]
        else:
            co_cls_hiddens = cls_hiddens.squeeze()

        skip_hiddens = lm_out.hidden_states[self.model_args.skip_from]
        hiddens = torch.cat([cls_hiddens, skip_hiddens[:, 1:]], dim=1)

        for layer in self.c_head:
            layer_out = layer(
                hiddens,
                attention_mask,
            )
            hiddens = layer_out[0]

        head_mlm_loss: torch.Tensor = self.mlm_loss(hiddens, labels)  # Head loss
        logs['head_mlm_loss'] = head_mlm_loss.item()
        loss += head_mlm_loss

        if self.model_args.late_mlm:
            logs['bert_mlm_loss'] = lm_out.loss.item()
            loss += lm_out.loss

        if grad_cache is None:
            co_loss = self.compute_contrastive_loss(co_cls_hiddens)
            logs['clloss'] = co_loss.item()
            return {
                'loss': loss + co_loss,
                'logs': logs,
            }
        else:
            loss = loss * (float(hiddens.size(0)) / self.train_args.per_device_train_batch_size)
            cached_grads = grad_cache[chunk_offset: chunk_offset + co_cls_hiddens.size(0)]
            surrogate = torch.dot(cached_grads.flatten(), co_cls_hiddens.flatten())
            return {
                'loss': loss,
                'logs': logs,
                'surrogate': surrogate,
            }

    @staticmethod
    def _world_size():
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1

    def compute_contrastive_loss(self, co_cls_hiddens):
        similarities = torch.matmul(co_cls_hiddens, co_cls_hiddens.transpose(0, 1)) / self.model_args.cl_temp
        similarities.fill_diagonal_(float('-inf'))
        target = torch.arange(similarities.shape[0], dtype=torch.long, device=similarities.device).view(-1, 2).flip([1]).flatten().contiguous()

        co_loss = F.cross_entropy(similarities, target) * self._world_size() * self.model_args.cl_coef
        return co_loss
