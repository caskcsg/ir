#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Training arguments.

@Time    :   2023/11/06
@Author  :   Ma (Ma787639046@outlook.com)
'''

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict
import os
from transformers import TrainingArguments

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    query_collection: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    passage_collection: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    corpus_path: Optional[str] = field(
        default=None, metadata={"help": "Path to train triples / encode corpus data"}
    )
    dev_path: Optional[str] = field(
        default=None, metadata={"help": "Path to development triples, the same format as training negative triples."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    q_max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    p_max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: Union[bool] = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )

    train_n_passages: int = field(default=8)
    eval_n_passages: int = field(default=8)
    positive_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first positive passage"})
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first negative passages"})

    qrel_path: Optional[str] = field(
        default=None, metadata={"help": "Path to qrels for filtering out queries to encode."}
    )
    encoded_save_prefix: str = field(default=None, metadata={"help": "where to save the encode"})
    encode_is_qry: bool = field(default=False)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default='sentence-transformers',
        metadata={
            "help": "The model archeticture used in training."
                    "Choose among ['transformers', 'sentence-transformers', 'instructor']."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature scale for clloss."}
    )
    clloss_coef: float = field(
        default=1.0,
        metadata={"help": "Scale factor for clloss."}
    )
    pooling_strategy: str = field(
        default="cls",
        metadata={
            "help": "Pooling strategy. Choose between mean/max/cls."
        },
    )
    score_function: str = field(
        default="dot",
        metadata={
            "help": "Pooling strategy. Choose between dot/cos_sim."
        },
    )

    distillation: bool = field(
        default=False,
        metadata={"help": "KL loss between Retriever query-passage scores and CrossEncoder scores."}
    )


@dataclass
class CLTrainingArguments(TrainingArguments):
    min_lr_ratio: float = field(default=0.0)
    
    logging_path: Optional[str] = field(
        default=None, metadata={"help": "Path for redirecting Transformers logs to local file."}
    )

    def __post_init__(self):
        super().__post_init__()

        if self.resume_from_checkpoint is not None:
            if self.resume_from_checkpoint.lower() in ["false", 'f']:
                self.resume_from_checkpoint = None
            elif self.resume_from_checkpoint.lower() in ["true", 't']:
                self.resume_from_checkpoint = True
