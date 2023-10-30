#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Training Arguments

@Author  :   Ma (Ma787639046@outlook.com)
'''
from dataclasses import dataclass, field
from typing import Optional, Union
import os
from transformers import TrainingArguments

@dataclass
class DataTrainingArguments:
    """
    Arguments control input data path, mask behaviors
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    train_path: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated. Default to the max input length of the model."
        },
    )
    min_seq_length: int = field(default=16)
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    data_type: str = field(
        default='origin_context_query',
        metadata={
            "help": "Choose between 'origin_context_query', 'ori_mixed', 'cotmae'/'mixed', 'random_sampled', 'nearby', 'overlap'"
                    "'random_sampled+nearby', 'random_sampled+overlap', 'nearby+overlap'"
        },
    )
    sample_from_spans: bool = field(
        default=False, 
        metadata={"help": 
                  "Whether to sample from a bunch of spans from same document."
                  "This ensures no spans from same document appear in one batch."
                  "Useful when you want to add contrastive learning object to CotMAE."
        }
    )
    bert_mask_ratio: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for BERT"}
    )
    enc_head_mask_ratio: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for Transformers-encoder based head"}
    )

    def __post_init__(self):
        if self.train_dir is not None:
            files = os.listdir(self.train_dir)
            self.train_path = [
                os.path.join(self.train_dir, f)
                for f in files
                if f.endswith('tsv') or f.endswith('json')
            ]
        if '+' in self.data_type:
            _data_types = self.data_type.split('+')
            self.data_type = [i.strip() for i in _data_types]

@dataclass
class ModelArguments:
    """
    Arguments control model config, decoder head config
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default='bert',
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
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    freeze_bert: bool = field(
        default=False,
        metadata={"help": "Whether to freeze BERT training. This option will not affect the grad of heads."},
    )
    disable_bert_mlm_loss: bool = field(
        default=False,
        metadata={"help": "Whether to disable BERT MLM loss"},
    )


    # Transformers-Encoder-based head mlm
    n_enc_head_layers: int = field(default=1)
    use_enc_head: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to use a transformer-encoder head of MAE, please set to True"}
    )
    enable_enc_head_mlm: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to add encoder-based-head layer mlm loss"}
    )
    enc_head_mlm_coef: Optional[float] = field(default=1.0)

    
    # Transformers-Decoder-based head mlm
    n_dec_head_layers: int = field(default=1)
    use_dec_head: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to use a transformer-decoder head for autoregression generation, please set to True"}
    )
    enable_dec_head_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to add decode-based-head layer CE loss"}
    )
    dec_head_coef: Optional[float] = field(default=1.0)
    
    # Customized casual attention mask, attention only on cls & tokens within model_args.attn_window
    attn_window: int = field(
        default=-1,
        metadata={"help": "Set a triangle casual attention mask with attention window span restrictions."
                          "-1 to disable this act."
        }
    )


@dataclass
class CotMAEPreTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    remove_unused_columns: bool = field(default=False)

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
