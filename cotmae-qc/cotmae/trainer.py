#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
A HuggingFace Trainer that support customized logs during training.

@Author  :   Ma (Ma787639046@outlook.com)
'''

import os
import json
import torch
import datetime
from typing import Dict, Optional
from collections import defaultdict, deque

from transformers import __version__
from transformers.trainer import Trainer
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import WEIGHTS_NAME, WEIGHTS_INDEX_NAME, CONFIG_NAME
from transformers.utils.import_utils import is_sagemaker_mp_enabled
from arguments import DataTrainingArguments, ModelArguments, CotMAEPreTrainingArguments as TrainingArguments

import warnings
import logging
logger = logging.getLogger(__name__)

class TrainerWithLogs(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Inject Customised logging behavior
        self.customized_logging_list = defaultdict(list)
        # Redirect loss logs to local file
        if self.args.local_rank <= 0:   # For local_rank == 0
            if hasattr(self.args, 'logging_path') and self.args.logging_path is not None and os.path.exists(self.args.logging_path) and os.path.isfile(self.args.logging_path):
                self.log_file = open(self.args.logging_path, 'a+')
    
    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save_pretrained'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save_pretrained interface')
        else:
            self.model.save_pretrained(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
    
    # Compatible with Transformers 4.17.0
    # Discarded in higher version
    def _load_state_dict_in_model(self, state_dict):
        # For lastest Transformers
        warnings.warn("Omitting model reload. Model load has already been handled outside the Trainer")
        pass
        # load_result = self.model.load_state_dict(state_dict, strict=False)
    
    # Compatible with Transformers 4.24.0
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if is_sagemaker_mp_enabled():
            raise NotImplementedError()

        if model is None:
            model = self.model
        
        model_args: ModelArguments = model.model_args
        data_args: DataTrainingArguments = model.data_args
        train_args: TrainingArguments = model.train_args

        if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)) and not os.path.isfile(
            os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        config = None
        if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
            config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warning(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if self.args.deepspeed:
            # will be resumed in deepspeed_init
            pass
        else:
            model.load(
                resume_from_checkpoint,
                from_tf=bool(".ckpt" in resume_from_checkpoint),
                # config=config,
                cache_dir=model_args.cache_dir,
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        A neat compute_loss that supports customized logging

        """
        outputs = model(**inputs)
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Inject Customised logging behavior
        try:
            logs: dict = outputs.logs
        except:
            logs = None
        if logs is not None:
            for k, v in logs.items():
                # Set maxlen of list to avoid memory leak, useful when
                # customized_logging_list has not been cleaned correctly
                if len(self.customized_logging_list[k]) < 5000: 
                    self.customized_logging_list[k].append(v)

        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        
        # Inject Customised logging behavior
        for k, v in self.customized_logging_list.items():
            if len(v) > 0:
                logs[k] = round(sum(v) / len(v), 4)
        self.customized_logging_list.clear()

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

        # Save log to file
        if self.args.local_rank <= 0 and hasattr(self, 'log_file'):
            self.log_file.write(f'{datetime.datetime.now()} - {json.dumps(output)}\n')
            self.log_file.flush()
        