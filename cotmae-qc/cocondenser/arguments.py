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

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict
import os
from transformers import TrainingArguments

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    train_path: Optional[str] = field(
        default=None, metadata={"help": "Path to train data"}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    train_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input train ref data file for whole word masking in Chinese."},
    )
    validation_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input validation ref data file for whole word masking in Chinese."},
    )
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
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    data_type: Optional[str] = field(
        default="query", metadata={"help": "The name of the dataset keys to use."}
    )

    # Interleavable datasets
    domain_config_path: Optional[str] = field(
        default=None, metadata={"help": "Path to json format domain config."}
    )
    preprocessed_dir: Optional[str] = field(
        default=None, metadata={"help": "Root folder path of all processed domains."}
    )
    add_domain_id: bool = field(
        default=False, metadata={"help": "Add domain index."}
    )
    stopping_strategy: str = field(
        default="all_exhausted", metadata={"help": "Set to 'first_exhausted' for less sampling "
                                "or 'all_exhausted' for oversampling."
                                "See `datasets.interleave_datasets`"}
    )

    @staticmethod
    def process_customized_data_types(data_type: str) -> Union[List, Dict]:
        if '+' in data_type:    # Split data type by '+'
            data_type = data_type.split('+')
            for i in data_type:
                i = i.strip()
            if ':' in data_type[0]:   # Split data weight if exists
                ret_types = defaultdict(list)
                for i in data_type:
                    _type_and_weight = i.split(':')
                    ret_types['data_type'].append(_type_and_weight[0].strip())
                    ret_types['weight'].append(float(_type_and_weight[1]))
            else:
                ret_types = data_type
            return ret_types
        else:
            return data_type

    def __post_init__(self):
        if self.train_path is not None:
            self.train_path = [self.train_path]
        else:
            if self.train_dir is not None:
                files = os.listdir(self.train_dir)
                self.train_path = [
                    os.path.join(self.train_dir, f)
                    for f in files
                    if f.endswith('tsv') or f.endswith('json')
                ]
        
        # Post Process data type
        self.data_type = self.process_customized_data_types(self.data_type)

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
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    n_head_layers: int = field(default=2)
    skip_from: int = field(default=2)
    late_mlm: bool = field(default=False)

    # CL Coef & Temperature
    cl_coef: float = field(
        default=1.0,
        metadata={"help": "Coef scale for clloss."}
    )
    cl_temp: float = field(
        default=1.0,
        metadata={"help": "Temperature scale for clloss."}
    )

@dataclass
class CondenserPreTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
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



@dataclass
class CoCondenserPreTrainingArguments(CondenserPreTrainingArguments):
    cache_chunk_size: int = field(default=-1)
