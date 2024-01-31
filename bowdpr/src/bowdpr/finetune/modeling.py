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
import torch.distributed as dist
import torch.nn.functional as F

from transformers import (
    BatchEncoding,
)
from transformers.file_utils import ModelOutput

from sentence_transformers import SentenceTransformer, models

from .arguments import ModelArguments, DataArguments, TrainingArguments

@dataclass
class SentenceTransformerEvalOutput(ModelOutput):
    scores: Optional[Tensor] = None

@dataclass
class SentenceTransformerOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    logs: Optional[Dict[str, any]] = None

class SentenceTransformerforCL(SentenceTransformer):
    _keys_to_ignore_on_save = []
    
    def __init__(
        self, 
        model_args: ModelArguments, 
        data_args: DataArguments,
        training_args: TrainingArguments,
        model_name_or_path: str,
        device: str,
        *args, 
        **kwargs
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        
        if os.path.exists(os.path.join(model_name_or_path, 'modules.json')):    # Load existing SentenceTransformer model
            super(SentenceTransformerforCL, self).__init__(model_name_or_path=model_name_or_path, device=device)
        else:
            # 0: Emb Model
            word_embedding_model = models.Transformer(model_name_or_path, max_seq_length=data_args.p_max_len)
            # 1: Pooling Method
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=model_args.pooling_strategy)
            modules=[word_embedding_model, pooling_model]
            # 2: Normalize (Optional)
            if model_args.score_function == 'cos_sim':
                modules.append(models.Normalize())

            super(SentenceTransformerforCL, self).__init__(modules=modules, device=device)  
    
    def forward(
        self, 
        query: BatchEncoding = None, 
        passage: BatchEncoding = None, 
        ce_scores: torch.Tensor = None
    ):
        # Compute Query and Passage Embeddings
        qry_emb, psg_emb = None, None
        if query is not None:
            qry_emb = super(SentenceTransformerforCL, self).forward(query)['sentence_embedding']
        if passage is not None:
            psg_emb = super(SentenceTransformerforCL, self).forward(passage)['sentence_embedding']
        
        # Inference
        if query is None or passage is None:
            return SentenceTransformerOutput(
                q_reps=qry_emb,
                p_reps=psg_emb,
            )

        # Customized Logs
        logs = dict()
        
        # Training with contrastive loss
        scores = None
        if self.training:
            clloss = 0.
            if self.model_args.clloss_coef > 0:
                if dist.is_initialized() and dist.get_world_size() > 1:
                    qry_emb_full = self._dist_gather_tensor(qry_emb)
                    psg_emb_full = self._dist_gather_tensor(psg_emb)
                else:
                    qry_emb_full = qry_emb
                    psg_emb_full = psg_emb
                
                scores = self.compute_similarity(qry_emb_full, psg_emb_full) / self.model_args.temperature
                assert scores.size(0) == qry_emb_full.size(0)

                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * (psg_emb_full.size(0) // qry_emb_full.size(0))      # [1, 2, 3 ...] * train_n_passages

                clloss = nn.CrossEntropyLoss(reduction='mean')(scores, target) * self.model_args.clloss_coef
                logs['clloss'] = round(clloss.item(), 4)
                logs['cl_coef'] = self.model_args.clloss_coef

            klloss = 0.
            if self.model_args.distillation:
                qry_emb_student = qry_emb.unsqueeze(1)      # B 1 D
                psg_emb_student = psg_emb.view(qry_emb.shape[0], -1, qry_emb.shape[-1]) # B N D
                student_scores = self.compute_similarity(qry_emb_student, psg_emb_student).squeeze(1) / self.model_args.temperature        # B 1 N -> B N
                teacher_scores = ce_scores.view(student_scores.shape[0], student_scores.shape[1])
                
                klloss = self.klloss(student_scores, teacher_scores)
                logs['klloss'] = round(klloss.item(), 4)

            return SentenceTransformerOutput(
                loss=clloss + klloss,
                q_reps=qry_emb,
                p_reps=psg_emb,
                scores=scores,
                logs={'temp': self.model_args.temperature},
            )
        
        # Eval
        else:
            qry_emb_eval = qry_emb.unsqueeze(1)      # B 1 D
            psg_emb_eval = psg_emb.view(qry_emb.shape[0], -1, qry_emb.shape[-1]) # B N D
            scores = self.compute_similarity(qry_emb_eval, psg_emb_eval).squeeze(1)      # B N
            return SentenceTransformerEvalOutput(scores=scores)
        
    @staticmethod
    def compute_similarity(qry_emb: torch.Tensor, psg_emb: torch.Tensor):
        """
        Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
        Input:  1) qry_emb: [batch_size, hidden_dim]
                   psg_emb: [batch_size * train_n_passages, hidden_dim]
                   return:  [batch_size, train_n_passages]
                2) qry_emb: [batch_size, 1, hidden_dim]
                   psg_emb: [batch_size, train_n_passages, hidden_dim]
                   return:  [batch_size, 1, train_n_passages]
        Return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
        """
        return torch.matmul(qry_emb, psg_emb.transpose(-2, -1))
    
    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)

        all_tensors[self.training_args.local_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors
    
    @staticmethod
    def klloss(student_scores: torch.Tensor, teacher_scores: torch.Tensor) -> torch.Tensor:
        """ A parameter free KL loss implementation """
        # Calculate klloss for distilation from teacher to student
        klloss = F.kl_div(F.log_softmax(student_scores, dim=-1), 
                        F.softmax(teacher_scores, dim=-1), 
                        reduction='batchmean')
        return klloss  # choose 'sum' or 'mean' depending on loss scale
