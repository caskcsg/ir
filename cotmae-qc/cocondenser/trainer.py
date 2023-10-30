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

import os
import json
import datetime
from collections import defaultdict
from contextlib import nullcontext

from typing import Dict, List, Tuple, Optional, Any, Union

import torch
import torch.distributed as dist
from torch import nn, Tensor
from torch.cuda.amp import autocast
from transformers import __version__
from transformers.trainer import Trainer
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import WEIGHTS_NAME, WEIGHTS_INDEX_NAME, CONFIG_NAME
from transformers.utils.import_utils import is_sagemaker_mp_enabled
from arguments import DataTrainingArguments, ModelArguments, CoCondenserPreTrainingArguments as TrainingArguments

try:
    from GradCache.src.grad_cache.grad_cache import GradCache
    _grad_cache_available = True
except ModuleNotFoundError:
    _grad_cache_available = False

import warnings
import logging

logger = logging.getLogger(__name__)


class CondenserPreTrainer(Trainer):
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
        labels = inputs.pop('labels')
        outputs = model(inputs, labels)
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


class CoCondenserPretrainer(CondenserPreTrainer):
    def __init__(self, *args, **kwargs):
        logger.info('Initializing Gradient Cache Trainer')
        super().__init__(*args, **kwargs)

        if self.args.fp16:
            if (not hasattr(self, "scaler")) and hasattr(self, "accelerator"):
                self.scaler = self.accelerator.scaler
            else:
                raise NotImplementedError()

        if self.args.cache_chunk_size != -1:
            if not _grad_cache_available:
                raise ValueError(
                    'Grad Cache package not available. You can obtain it from https://github.com/luyug/GradCache.')
            self.gc = GradCache(
                models=[self.model.lm],
                chunk_sizes=self.args.cache_chunk_size,
                loss_fn=self.model.compute_contrastive_loss,
                get_rep_fn=lambda x: x['hidden_states'][-1][:, 0],
                fp16=self.args.fp16,
                scaler=self.scaler,
            )

    def _gather_tensor(self, t: Tensor):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[self.args.local_rank] = t
        return all_tensors

    def gather_tensors(self, *tt: Tensor):
        tt = [torch.cat(self._gather_tensor(t)) for t in tt]
        return tt

    def compute_loss(self, model, inputs, grad_cache=None, chunk_offset=None, return_outputs=False):
        """
        A neat compute_loss that supports customized logging

        """
        labels = inputs.pop('labels')
        outputs = model(inputs, labels, grad_cache=grad_cache, chunk_offset=chunk_offset)
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Inject Customised logging behavior
        try:
            logs: dict = outputs["logs"]
        except:
            logs = None
        if logs is not None:
            for k, v in logs.items():
                # Set maxlen of list to avoid memory leak, useful when
                # customized_logging_list has not been cleaned correctly
                if len(self.customized_logging_list[k]) < 5000: 
                    self.customized_logging_list[k].append(v)

        return (loss, outputs) if return_outputs else loss

    def split_tensor_dict(self, td: Dict[str, Tensor]):
        keys = list(td.keys())
        chunked_tensors = [td[k].split(self.args.cache_chunk_size) for k in keys]
        return [dict(zip(keys, tt)) for tt in zip(*chunked_tensors)]

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        if self.args.cache_chunk_size == -1:
            return super(CoCondenserPretrainer, self).training_step(model, inputs)

        model.train()

        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop('labels')

        # Construct the gradient cache
        chunked_inputs = self.split_tensor_dict(inputs)
        for c in chunked_inputs:
            c['output_hidden_states'] = True
        cls_hiddens, rnd_states = self.gc.forward_no_grad(self.model.lm, chunked_inputs)
        if self.args.local_rank > -1:
            cls_hiddens = self.gather_tensors(cls_hiddens.contiguous())[0]
        grad_cache, total_loss = self.gc.build_cache(cls_hiddens)
        grad_cache = grad_cache[0]
        if self.args.local_rank > -1:
            total_loss = total_loss / dist.get_world_size()

        inputs['labels'] = labels
        chunked_inputs = self.split_tensor_dict(inputs)

        # Compute the full loss with cached gradients
        for local_chunk_id, chunk in enumerate(chunked_inputs):
            device_offset = max(0, self.args.local_rank) * self.args.per_device_train_batch_size * 2
            local_offset = local_chunk_id * self.args.cache_chunk_size
            chunk_offset = device_offset + local_offset
            with rnd_states[local_chunk_id]:
                if self.use_cuda_amp or self.use_cpu_amp:
                    with autocast():
                        lm_loss, outputs = self.compute_loss(model, chunk, grad_cache, chunk_offset, return_outputs=True)
                        surrogate = outputs['surrogate']
                else:
                    lm_loss, outputs = self.compute_loss(model, chunk, grad_cache, chunk_offset, return_outputs=True)
                    surrogate = outputs['surrogate']

            if self.args.gradient_accumulation_steps > 1:
                raise ValueError

            ddp_no_sync = self.args.local_rank > -1 and (local_chunk_id + 1 < len(chunked_inputs))
            with model.no_sync() if ddp_no_sync else nullcontext():
                # if self.use_cuda_amp or self.use_cpu_amp:
                if self.args.fp16:      # Compatible with Huggingface w/Accelerator Decorator
                    (self.scaler.scale(lm_loss) + surrogate).backward()
                elif self.use_apex:
                    raise ValueError
                elif self.deepspeed:
                    raise ValueError
                else:
                    (lm_loss + surrogate).backward()
            total_loss += lm_loss
        return total_loss.detach()
