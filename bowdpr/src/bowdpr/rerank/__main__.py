#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Rerank

@Time    :   2023/12/13
@Author  :   Ma (Ma787639046@outlook.com)
'''
import os
import sys
import shutil
import pickle
import numpy as np
from tqdm import tqdm
from typing import List
from pathlib import Path
from collections import defaultdict
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import transformers
from transformers import (
    PreTrainedModel,
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    BatchEncoding,
    set_seed, 
)
from transformers.modeling_outputs import SequenceClassifierOutput

from .arguments import DataArguments, ModelArguments, CLTrainingArguments as TrainingArguments
from .data_utils import RerankerEncodeDataset as EncodeDataset, RerankerEncodeCollator as EncodeCollator
from .modeling import CrossEncoder

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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    dataset = EncodeDataset(
        data_args=data_args, 
        tsv_ranks_path=data_args.tsv_ranks_path, 
        query_collection=data_args.query_collection, 
        passage_collection=data_args.passage_collection,
    )
    logger.info(f"Dataset lengths in total: {len(dataset)}")
    
    # Split dataset by global ranks
    dataset.shard_(num_shards=training_args.world_size, index=training_args.process_index)

    # Config
    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )

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
        max_length=data_args.max_seq_length,
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
    # Cross Encoder
    model = CrossEncoder.from_pretrained(
        model_args=model_args, 
        data_args=data_args,
        training_args=training_args,
        pretrained_model_name_or_path=model_args.model_name_or_path, 
        config=config,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
    )

    logger.warning(f"Encode dataset lengths on Rank {training_args.process_index}/{training_args.world_size}: {len(dataset)}")

    model = model.to(training_args.device)
    model.eval()

    dtype = None
    if training_args.fp16:
        dtype = torch.float16
    elif training_args.bf16:
        dtype = torch.bfloat16

    # Start Inferencing
    q_p_dict = {}       # qid -> List[(pid, score)]

    for (qids, pids, batch) in tqdm(encode_loader, desc=f"Rank {training_args.process_index}/{training_args.world_size}"):
        with torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext():
            with torch.no_grad():
                batch.to(training_args.device)
                model_output: SequenceClassifierOutput = model(batch)
                scores = model_output.logits.squeeze().tolist()
                for qid, pid, score in zip(qids, pids, scores):
                    if qid not in q_p_dict:
                        q_p_dict[qid] = [(pid, score)]
                    else:
                        q_p_dict[qid].append((pid, score))
    
    # Embedding Save Path
    rerank_save_folder: Path = Path(data_args.rerank_save_path).parent / "temp_scores_for_rerank"
    rerank_save_folder.mkdir(parents=True, exist_ok=True)
    
    # Save partial results
    with open(rerank_save_folder / f"{training_args.process_index}.pt", 'wb') as f:
        pickle.dump(q_p_dict, f)
    
    # Wait for everyone
    dist.barrier()

    # Process all scores on rank0
    if training_args.process_index in [0, -1]:
        # Read scores of rank 1~world_size from local temp files
        for _process_idx in range(1, training_args.world_size):
            with open(rerank_save_folder / f"{_process_idx}.pt", 'rb') as f:
                local_q_p_dict = pickle.load(f)
                for qid, item in local_q_p_dict.items():
                    if qid not in q_p_dict:
                        q_p_dict[qid] = item
                    else:
                        q_p_dict[qid].extend(item)
        
        # Sort and write final scores
        with open(data_args.rerank_save_path, "w") as f:
            for qid, item in q_p_dict.items():
                for curr_idx, (pid, score) in enumerate(sorted(item, key=lambda x: x[1], reverse=True)):
                    if curr_idx > data_args.reranking_depth:
                        break
                    f.write(f'{qid}\t{pid}\t{score}\n')
        
        # Remove temp files
        shutil.rmtree(rerank_save_folder)

if __name__ == "__main__":
    main()