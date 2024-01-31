#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Trainer for contrastive loss.

@Time    :   2023/11/06
@Author  :   Ma (Ma787639046@outlook.com)
'''

import os
import json
import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
from torch import nn, Tensor
import torch.distributed as dist
from transformers import (
    BatchEncoding,
    get_scheduler,
    __version__
)
from transformers.trainer import Trainer
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.configuration_utils import PretrainedConfig
from transformers.utils.import_utils import is_sagemaker_mp_enabled
from transformers.utils import WEIGHTS_NAME, WEIGHTS_INDEX_NAME, CONFIG_NAME, is_peft_available

from sentence_transformers import SentenceTransformer

from .scheduler import get_linear_schedule_with_warmup_minlr, get_cosine_schedule_with_warmup_minlr

import warnings
import logging
logger = logging.getLogger(__name__)

class ContrastiveTrainer(Trainer):
    """
    Huggingface Trainer for DPR
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Inject Customised logging behavior
        self.customized_logging_list = defaultdict(list)
        # Redirect loss logs to local file
        if self.args.local_rank <= 0:   # For local_rank == 0
            if hasattr(self.args, 'logging_path') and self.args.logging_path is not None and os.path.exists(self.args.logging_path) and os.path.isfile(self.args.logging_path):
                self.log_file = open(self.args.logging_path, 'a+')
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """ 
        Contrative Learning does not produce labels at dataloader.
        Here we add all zero labels for `eval_step`.
        """
        (loss, logits, labels) = super().prediction_step(model=model, inputs=inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)

        if labels is None:
            labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.int)
        
        return (loss, logits, labels)

    def _gather_tensor(self, t: Tensor):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[self.args.local_rank] = t
        return all_tensors

    def gather_tensors(self, *tt: Tensor):
        tt = [torch.cat(self._gather_tensor(t)) for t in tt]
        return tt

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
    
    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)

        # Also save state dict as `sentence_transformers` format, for downstream finetuning
        model = unwrap_model(self.model)
        if isinstance(model, SentenceTransformer):
            model.save(output_dir)
        elif hasattr(self.model, 'save_pretrained'):
            # Save a trained model and configuration using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            self.model.save_pretrained(output_dir)
        else:
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save_pretrained interface')
    
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
    def _load_from_checkpoint(self, resume_from_checkpoint: str, model: nn.Module=None):
        if is_sagemaker_mp_enabled():
            raise NotImplementedError()

        if model is None:
            model = self.model

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
            if isinstance(model, SentenceTransformer):
                # Load SentenceTransformer from local files
                ckpt: nn.Module = SentenceTransformer(resume_from_checkpoint, device='cpu')
                load_results = model.load_state_dict(ckpt.state_dict())
                del ckpt
            elif hasattr(model, "load"):
                # Load Pre-training checkpoint with header from local files
                model.load(
                    resume_from_checkpoint,
                    from_tf=bool(".ckpt" in resume_from_checkpoint),
                    config=config,
                )
            else:
                # Load all other models to `model.lm`
                super()._load_from_checkpoint(resume_from_checkpoint, model=model.lm)
    
    def _load_best_model(self):
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        self._load_from_checkpoint(self.state.best_model_checkpoint)
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            if self.args.lr_scheduler_type == "linear":
                self.lr_scheduler = get_linear_schedule_with_warmup_minlr(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    min_lr_ratio=self.args.min_lr_ratio,
                )
            elif self.args.lr_scheduler_type == "cosine":
                self.lr_scheduler = get_cosine_schedule_with_warmup_minlr(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    min_lr_ratio=self.args.min_lr_ratio,
                )
            else:
                self.lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
            self._created_lr_scheduler = True
        return self.lr_scheduler

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
