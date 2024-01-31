#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Training scripts.

@Time    :   2023/12/04
@Author  :   Ma (Ma787639046@outlook.com)
'''

import os
import sys

import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    set_seed, )
from transformers.trainer_utils import get_last_checkpoint

from .modeling import SentenceTransformerforCL
from .arguments import DataArguments, ModelArguments, CLTrainingArguments as TrainingArguments
from .data_utils import TrainDataset, TrainCollator
from .metrics import compute_metrics_warpper
from ..trainer import ContrastiveTrainer as Trainer

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
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    
    if training_args.logging_path:
        _log_file_folder = os.path.split(training_args.logging_path)
        if len(_log_file_folder) == 2 and _log_file_folder[0] != "":
            os.makedirs(_log_file_folder[0], exist_ok=True)
        log_file_handler = logging.FileHandler(training_args.logging_path)
        transformers.utils.logging.add_handler(log_file_handler)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len([i for i in os.listdir(training_args.output_dir) if i != "runs"]) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Json format dataset (Single dataset)
    train_set = TrainDataset(
        data_args=data_args, 
        dataset=data_args.corpus_path,
        query_collection=data_args.query_collection,
        passage_collection=data_args.passage_collection,
        train_n_passages=data_args.train_n_passages,
        positive_passage_no_shuffle=data_args.positive_passage_no_shuffle,
        negative_passage_no_shuffle=data_args.negative_passage_no_shuffle,
    )

    dev_set = None
    if data_args.dev_path is not None:
        dev_set = TrainDataset(
            data_args=data_args, 
            dataset=data_args.dev_path,
            train_n_passages=data_args.eval_n_passages,
            positive_passage_no_shuffle=data_args.positive_passage_no_shuffle,
            negative_passage_no_shuffle=True,
            # Note:
            # We encourage you to enable `--save_text` when making dev sets
            # because sometimes we want to use dev sets from another domain
            # for development set.

            # query_collection=data_args.query_collection,
            # passage_collection=data_args.passage_collection,
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

    # Model
    # Sentence-Transformers
    model = SentenceTransformerforCL(
        model_args=model_args, 
        data_args=data_args,
        training_args=training_args,
        model_name_or_path=model_args.model_name_or_path, 
        device='cpu'
    )

    # Print Model
    print(model)

    # Data collator    
    data_collator = TrainCollator(
        tokenizer=tokenizer,
        padding='max_length' if data_args.pad_to_max_length else 'longest',
        q_max_len=data_args.q_max_len,
        p_max_len=data_args.p_max_len,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set if training_args.do_train else None,
        eval_dataset=dev_set if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_warpper(metric_for_best_model=training_args.metric_for_best_model) if training_args.do_eval else None,
    )
    train_set.trainer = trainer
    if dev_set is not None:
        dev_set.trainer = trainer

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
    
    if training_args.do_eval:
        eval_results = trainer.evaluate()
        logger.info(f"Eval results: {eval_results}")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
