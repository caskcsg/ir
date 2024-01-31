#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Training datasets.

@Time    :   2023/11/06
@Author  :   Ma (Ma787639046@outlook.com)
'''
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset

import datasets
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding

from .arguments import DataArguments
from ..trainer import ContrastiveTrainer
from ..utils.data_utils import read_corpus, build_corpus_idx_to_row

import logging
logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    """ Wrapper for Sampling Positive / Negative Passages """
    def __init__(
            self,
            data_args: DataArguments,
            dataset: str,                                  # String Path to Training Triples with Negatives
            query_collection: Optional[str] = None,        # String Path to query corpus
            passage_collection: Optional[str] = None,      # String Path to passage corpus
            trainer: ContrastiveTrainer = None,
            train_n_passages: int = 8,
            positive_passage_no_shuffle: bool = False,
            negative_passage_no_shuffle: bool = False,
    ):
        self.train_data = read_corpus(dataset)
        self.trainer = trainer
        self.data_args = data_args
        self.train_n_passages = train_n_passages
        self.positive_passage_no_shuffle = positive_passage_no_shuffle
        self.negative_passage_no_shuffle = negative_passage_no_shuffle

        self.read_text_from_collections = (query_collection is not None) and (passage_collection is not None)
        if query_collection is not None:
            # Load query corpus
            self.query_dataset: datasets.Dataset = read_corpus(query_collection)
            self.idx_to_query: Dict[str, int] = build_corpus_idx_to_row(self.query_dataset)
        
        if passage_collection is not None:
            # Load passage corpus
            self.passage_dataset: datasets.Dataset = read_corpus(passage_collection)
            self.idx_to_passage: Dict[str, int] = build_corpus_idx_to_row(self.passage_dataset)
    
    def get_query(self, _id: str) -> dict:
        return self.query_dataset[self.idx_to_query[_id]]
    
    def get_passage(self, _id: str) -> dict:
        return self.passage_dataset[self.idx_to_passage[_id]]
    
    def __len__(self):
        return len(self.train_data) 

    def __getitem__(self, item) -> Dict[str, any]:
        group = self.train_data[item]
        _hashed_seed = hash(item + self.trainer.args.seed)

        epoch = int(self.trainer.state.epoch)

        # Read Query
        if self.read_text_from_collections:
            qry: str = self.get_query(group['_id'])['text']
        else:
            qry: str = group['text']

        # Sample One Positive
        group_positives = group['positive_passages']
        if self.positive_passage_no_shuffle:
            pos_psg: Dict[str, any] = group_positives[0]
        else:
            pos_psg: Dict[str, any] = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        
        if self.read_text_from_collections:
            pos_psg.update(self.get_passage(pos_psg['docid']))

        # Sample Negatives
        group_negatives = group['negative_passages']
        negative_size = self.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.train_n_passages == 1:
            negs = []
        elif self.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]
        
        if self.read_text_from_collections:
            negs_w_texts = list()
            for item in negs:
                item.update(self.get_passage(item['docid']))
                negs_w_texts.append(item)
            negs = negs_w_texts

        return {
            "query": qry,
            "positive_passages": [pos_psg],
            "negative_passages": negs,
        }


@dataclass
class TrainCollator(DataCollatorWithPadding):
    """
    DataCollator for processing & tokenize train dataset.
    """
    q_max_len: int = 512
    p_max_len: int = 512

    def __post_init__(self):
        # self.separator = getattr(self.tokenizer, "sep_token", ' ')  # [SEP]
        self.separator = " "        # WhiteSpace
    
    def _get_passage_text(self, item: Dict[str, str]):
        if "title" in item:
            return item["title"] + self.separator + item["text"]
        else:
            return item["text"]

    def __call__(self, features: List[dict]):
        # Tokenize `Query`
        q_texts: List[str] = [f['query'] for f in features]
        q_tokenized: BatchEncoding = self.tokenizer(
            q_texts,
            max_length=self.q_max_len,
            truncation='only_first',
            padding=self.padding,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors=self.return_tensors,
        )

        # Process `Passage` & `Negatives`
        p_texts = list()
        for item in features:
            p_texts.append(self._get_passage_text(item["positive_passages"][0]))     # Add Positive Texts
            for _neg in item["negative_passages"]:
                p_texts.append(self._get_passage_text(_neg))    # Add Negative Texts
        
        # Tokenize Passage
        p_tokenized: BatchEncoding = self.tokenizer(
            p_texts,
            max_length=self.p_max_len,
            truncation='only_first',
            padding=self.padding,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors=self.return_tensors,
        )

        # Sample CE Scores for Distillation
        ce_scores = None
        if 'ce_score' in features[0]["positive_passages"][0]:
            ce_scores: List[float] = list()
            for item in features:
                ce_scores.append(float(item["positive_passages"][0]['ce_score']))
                for _neg in item["negative_passages"]:
                    ce_scores.append(float(_neg['ce_score']))
            ce_scores = torch.tensor(ce_scores, dtype=torch.float32)

        processed = {
            "query": q_tokenized,
            "passage": p_tokenized,
            "ce_scores": ce_scores,
        }

        return processed


@dataclass
class EncodeCollator(DataCollatorWithPadding):
    """
    DataCollator for processing & tokenize encode dataset.
    """
    max_length: int = 512

    def __post_init__(self):
        # self.separator = getattr(self.tokenizer, "sep_token", ' ')  # [SEP]
        self.separator = " "        # WhiteSpace
    
    def _get_passage_text(self, item: Dict[str, str]):
        if "title" in item:
            return item["title"] + self.separator + item["text"]
        else:
            return item["text"]

    def __call__(self, features: List[dict]):
        ids = list()
        texts = list()
        for item in features:
            ids.append(item["_id"])
            texts.append(self._get_passage_text(item))
        
        encoded: BatchEncoding = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation='only_first',
            padding=self.padding,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors=self.return_tensors,
        )
        return ids, encoded

