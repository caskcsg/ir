#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Encode Sentence Embeddings.

@Time    :   2023/12/12
@Author  :   Ma (Ma787639046@outlook.com)
'''
import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
from typing import List
from pathlib import Path
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer

import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    set_seed, 
)
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node

from .arguments import DataArguments, ModelArguments, CLTrainingArguments as TrainingArguments
from .data_utils import EncodeCollator
from .modeling import SentenceTransformerforCL
from ..utils.data_utils import read_corpus, build_corpus_idx_to_row

import logging
logger = logging.getLogger(__name__)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed inferencing: {bool(training_args.local_rank != -1)}, fp16: {training_args.fp16}, bf16: {training_args.bf16}"
    )

    # Embedding Save Path
    embedding_save_path: Path = Path(data_args.encoded_save_prefix + f".{training_args.process_index}.pt")
    embedding_save_path.parent.mkdir(parents=True, exist_ok=True)
    if embedding_save_path.exists():
        logger.warning(f"Skiping {embedding_save_path} because it exists.")
        return

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Json format dataset
    dataset: Dataset = read_corpus(data_args.corpus_path)
    logger.info(f"Encode dataset lengths in total: {len(dataset)}")

    # Ensure that `query-id` in qrels, if qrels are provided
    if data_args.qrel_path is not None:
        qrels = load_dataset(
            'csv',
            data_files=data_args.qrel_path,
            delimiter='\t',
            split="train",
        )
        query_ids: List[str] = [item["query-id"] for item in qrels]
        dataset.filter(lambda item: item["_id"] in query_ids)
        logger.info(f"Encode dataset lengths after filter: {len(dataset)}")
    
    # Split dataset by global ranks
    dataset = split_dataset_by_node(dataset, rank=training_args.process_index, world_size=training_args.world_size)

    # Tokenizer
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Data collator    
    collator = EncodeCollator(
        tokenizer=tokenizer,
        padding='max_length' if data_args.pad_to_max_length else 'longest',
        max_length=data_args.q_max_len if data_args.encode_is_qry else data_args.p_max_len,
    )
    encode_loader = DataLoader(
        dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=collator,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=True,
    )

    # Load Model
    # Sentence-Transformers
    model = SentenceTransformerforCL(
        model_args=model_args, 
        data_args=data_args,
        training_args=training_args,
        model_name_or_path=model_args.model_name_or_path, 
        device='cpu'
    )

    # debug
    print(model)

    logger.warning(f"Encode dataset lengths on Rank {training_args.process_index}/{training_args.world_size}: {len(dataset)}")

    encoded = []
    lookup_indices = []
    model = model.to(training_args.device)
    model.eval()

    dtype = None
    if training_args.fp16:
        dtype = torch.float16
    elif training_args.bf16:
        dtype = torch.bfloat16

    # Start Inferencing  
    for (batch_ids, batch) in tqdm(encode_loader, desc=f"Rank {training_args.process_index}/{training_args.world_size}"):
        lookup_indices.extend(batch_ids)
        with torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device, non_blocking=True)
                if data_args.encode_is_qry:
                    reps = model(query=batch)["q_reps"]
                else:
                    reps = model(passage=batch)["p_reps"]
                encoded.append(reps.cpu().detach().numpy())

    encoded = np.concatenate(encoded)   
    with open(embedding_save_path, 'wb') as f:
        pickle.dump((encoded, lookup_indices), f)

if __name__ == "__main__":
    main()