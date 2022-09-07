#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Dataset for Cot-MAE

@Author  :   Ma (Ma787639046@outlook.com)
'''
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import os
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForWholeWordMask

from transformers.utils import logging
logger = logging.get_logger(__name__)

@dataclass
class CotMAECollator(DataCollatorForWholeWordMask):
    max_seq_length: int = 512
    encoder_mask_ratio: float = 0.15
    decoder_mask_ratio: float = 0.15
    data_type: str = 'mixed'

    def __post_init__(self):
        super().__post_init__()
        self.rng = random.Random()

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512, mlm_probability=0.15):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """

        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        self.rng.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def _truncate(self, example: List[int]):
        tgt_len = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(False)
        if len(example) <= tgt_len:
            return example
        trunc = len(example) - tgt_len
        trunc_left = self.rng.randint(0, trunc)
        trunc_right = trunc - trunc_left

        truncated = example[trunc_left:]
        if trunc_right > 0:
            truncated = truncated[:-trunc_right]

        if not len(truncated) == tgt_len:
            print(len(example), len(truncated), trunc_left, trunc_right, tgt_len, flush=True)
            raise ValueError
        return truncated

    def _pad(self, seq, val=0):
        tgt_len = self.max_seq_length
        assert len(seq) <= tgt_len
        return seq + [val for _ in range(tgt_len - len(seq))]
    
    def encode_batch_examples(self, examples: List[Dict[str, List[int]]], mlm_prob: float=0.15):
        encoded_examples = []
        masks = []
        mlm_masks = []

        for e in examples:
            e_trunc = self._truncate(e['text'])
            tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e_trunc]
            mlm_mask = self._whole_word_mask(tokens, mlm_probability=mlm_prob)
            mlm_mask = self._pad([0] + mlm_mask)
            mlm_masks.append(mlm_mask)

            encoded = self.tokenizer.encode_plus(
                self._truncate(e['text']),
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False,
            )
            masks.append(encoded['attention_mask'])
            encoded_examples.append(encoded['input_ids'])

        inputs, labels = self.mask_tokens(
            torch.tensor(encoded_examples, dtype=torch.long),
            torch.tensor(mlm_masks, dtype=torch.long)
        )
        attention_mask = torch.tensor(masks)

        batch = {
            "input_ids": inputs,
            "labels": labels,
            "attention_mask": attention_mask,
            "input_ids_unmasked": torch.tensor(encoded_examples, dtype=torch.long),
        }

        return batch

    def process(self, examples: List[Dict[str, List[int]]], cltypes=None):
        batch = self.encode_batch_examples(examples=examples, mlm_prob=self.encoder_mask_ratio)
        a = [i for i in range(0, len(examples), 2)]
        b = [i for i in range(1, len(examples), 2)]
        idxs = []
        for i, j in zip(a, b):
            idxs.append(j)
            idxs.append(i)
        decoder_examples = [examples[i] for i in idxs]
        decoder_batch = self.encode_batch_examples(examples=decoder_examples, mlm_prob=self.decoder_mask_ratio)
        batch['decoder_input_ids'] = decoder_batch['input_ids']
        batch['decoder_labels'] = decoder_batch['labels']
        batch['decoder_attention_mask'] = decoder_batch['attention_mask']
        
        return batch
    
    def __call__(self, examples):
        unpacked = []
        for text_dict in examples:
            unpacked.append({'text': text_dict['anchor']})
            if self.data_type == 'mixed':
                contextual_span_names = list(text_dict.keys())
                contextual_span_names.pop(contextual_span_names.index('anchor'))
                # Randomly pick a corpus cl span among 'random_sampled', 'nearby', 'overlap'
                _cltype = self.rng.choice(contextual_span_names)
            else:
                if isinstance(self.data_type, list):
                    _cltype = self.rng.choice(self.data_type)
                elif isinstance(self.data_type, str):
                    _cltype = self.data_type
                else:
                    raise NotImplementedError()
            unpacked.append({'text': text_dict[_cltype]})
        return self.process(unpacked)


class CotMAEDataset(Dataset):
    def __init__(self, dataset, data_args):
        self.dataset = dataset
        self.data_args = data_args
        self.rng = random.Random()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        spans = self.dataset[item]['spans']
        idx = self.rng.randint(0, len(spans)-1)
        return spans[idx]
