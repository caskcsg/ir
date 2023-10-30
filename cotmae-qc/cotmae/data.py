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
    bert_mask_ratio: float = 0.15
    enc_head_mask_ratio: float = 0.15
    data_type: str = 'mixed'

    def __post_init__(self):
        super().__post_init__()

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
    
    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512, mlm_probability=0.15, preprocessed_masked_lms: list=None, preprocessed_covered_indexes: set=None):
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

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * mlm_probability))))
        masked_lms = [] if not preprocessed_masked_lms else preprocessed_masked_lms
        covered_indexes = set() if not preprocessed_covered_indexes else preprocessed_covered_indexes
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

    def _truncate(self, example: List[int], tgt_len: int=512):
        if len(example) <= tgt_len:
            return example
        trunc = len(example) - tgt_len
        trunc_left = random.randint(0, trunc)
        trunc_right = trunc - trunc_left

        truncated = example[trunc_left:]
        if trunc_right > 0:
            truncated = truncated[:-trunc_right]

        if not len(truncated) == tgt_len:
            print(len(example), len(truncated), trunc_left, trunc_right, tgt_len, flush=True)
            raise ValueError
        return truncated

    def _pad(self, seq, val=0, tgt_len: int=512):
        assert len(seq) <= tgt_len
        return seq + [val for _ in range(tgt_len - len(seq))]
    
    # WWM
    def encode_batch_examples(self, 
                              examples: List[Dict[str, List[int]]], 
                              mlm_prob: float=0.15,
                              num_special_tokens_to_add: int=2, 
                              text_types: List[str]=None,
                              preprocessed_text_and_mask: dict=None,   # For further mask
                              ):
        # Encode a batch of examples with Whole Word Mask
        encoded_examples = []
        masks = []
        mlm_masks = []

        # Preserve original tokens & mlm_masks for further mask
        preserved_original_tokens = []
        preserved_mlm_masks = []

        # Dynamic padding
        tgt_len = max([len(e['text']) for e in examples])
        tgt_len = min(tgt_len + num_special_tokens_to_add, self.max_seq_length)
        tgt_len_wo_special_tokens = tgt_len - num_special_tokens_to_add

        # WWM with further mask of 'anchor'
        for idx, e in enumerate(examples):
            if (preprocessed_text_and_mask is not None) and (text_types is not None) and (text_types[idx] == 'anchor'):  # support of further mask
                e_trunc = preprocessed_text_and_mask['preserved_original_tokens'][idx][:tgt_len_wo_special_tokens]
                tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e_trunc]
                preprocessed_mlm_mask = preprocessed_text_and_mask['preserved_mlm_masks'][idx][:tgt_len_wo_special_tokens]
                preprocessed_masked_lms = []
                preprocessed_covered_indexes = set()
                for _token_idx, _token_mask in enumerate(preprocessed_mlm_mask):
                    if _token_mask == 1:
                        preprocessed_masked_lms.append(_token_idx)
                        preprocessed_covered_indexes.add(_token_idx)
                mlm_mask = self._whole_word_mask(
                                tokens, 
                                mlm_probability=mlm_prob, 
                                preprocessed_masked_lms=preprocessed_masked_lms, preprocessed_covered_indexes=preprocessed_covered_indexes
                            )  # WWM
            else:
                e_trunc = self._truncate(e['text'], tgt_len=tgt_len_wo_special_tokens) # Truncate on both side, as implemented in BERT
                tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e_trunc]
                mlm_mask = self._whole_word_mask(tokens, mlm_probability=mlm_prob)  # WWM
            preserved_original_tokens.append(e_trunc)
            preserved_mlm_masks.append(mlm_mask)
            mlm_mask = self._pad([0] + mlm_mask, tgt_len=tgt_len)
            mlm_masks.append(mlm_mask)

            encoded = self.tokenizer.encode_plus(e_trunc,
                add_special_tokens=True,
                max_length=tgt_len,
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

        preserved = {
            'preserved_original_tokens': preserved_original_tokens,
            'preserved_mlm_masks': preserved_mlm_masks,
        }

        return batch, preserved

    # Tokenize for AR-Dec
    def _tokenize_ar_dec(self, examples: List[Dict[str, List[int]]]):
        batch_encoding = self.tokenizer.batch_encode_plus(
                            batch_text_or_text_pairs=[self._truncate(e['text'], tgt_len=self.max_seq_length - 2) for e in examples],
                            add_special_tokens=True,
                            max_length=self.max_seq_length,
                            padding='longest',
                            truncation=True,
                            return_token_type_ids=False,
                            is_split_into_words=True,
                            return_tensors='pt',
                        )
        return batch_encoding

    def process(self, examples: Tuple[List[Dict[str, List[int]]], Dict[str, List[Dict[str, List[int]]]]]):
        # Here the data organization of SkipMAE is different from CotMAE
        # We only let BERT encoder forward anchor texts, not the [anchor, contextual span] texts
        # So the contextual spans are only used by head layer
        # There should not be 2x of batch size scaling when training SkipMAE
        batch, enc_preserved_text_and_mask = self.encode_batch_examples(examples=examples['anchor'], mlm_prob=self.bert_mask_ratio)

        # AE-Dec
        enc_head_batch, _ = self.encode_batch_examples(examples=examples['enc_input'], mlm_prob=self.enc_head_mask_ratio, 
                                                       text_types=examples['enc_text_type'], preprocessed_text_and_mask=enc_preserved_text_and_mask)
        batch['enc_head_input_ids'], batch['enc_head_labels'], batch['enc_head_attention_mask'] = \
            enc_head_batch['input_ids'], enc_head_batch['labels'], enc_head_batch['attention_mask']
        batch['enc_head_input_ids_unmasked'] = enc_head_batch['input_ids_unmasked']
        
        # AR-Dec
        ar_dec_head_batch = self._tokenize_ar_dec(examples['dec_input'])
        batch['dec_head_input_ids'] = ar_dec_head_batch['input_ids']
        batch['dec_head_attention_mask'] = ar_dec_head_batch['attention_mask']
        
        return batch
    
    def __call__(self, examples):
        unpacked = {'anchor': [], 'enc_input': [], 'enc_text_type': [], 
                    'dec_input': [], 'dec_text_type': [], 
                   }
        for text_dict in examples:
            # Anchor Text
            unpacked['anchor'].append({'text': text_dict['anchor']})

            contextual_span_names = list(filter(lambda x: x not in ['anchor', 'query'], text_dict.keys()))
            # Sample for AE-Dec(enc_input), AR-Dec(dec_input)
            for (_head_type, _cltypename) in [('enc_input', 'enc_text_type'), 
                                              ('dec_input', 'dec_text_type'),
                                              ]:
                if self.data_type == 'origin_context_query':
                    # Original + Contextual + query
                    _rand = random.random()
                    if _rand <= 0.33:
                        _cltype = 'anchor'  # 33% prob to sample the original span
                    elif _rand <= 0.66:
                        _cltype = 'query'  # 33% prob to sample the query span
                    else:   # 33% prob to sample a contextual span
                        # Randomly pick a corpus cl span among 'random_sampled', 'nearby', 'overlap'
                        _cltype = random.choice(contextual_span_names)
                elif self.data_type == 'ori_mixed':
                    # Original + Contextual
                    if random.random() <= 0.5:
                        _cltype = 'anchor'  # 50% prob to sample the original span
                    else:   # 50% prob to sample a contextual span
                        # Randomly pick a corpus cl span among 'random_sampled', 'nearby', 'overlap'
                        _cltype = random.choice(contextual_span_names)
                elif self.data_type == 'mixed' or self.data_type == 'cotmae':
                    # Contextual
                    # Randomly pick a corpus cl span among 'random_sampled', 'nearby', 'overlap'
                    _cltype = random.choice(contextual_span_names)
                else:
                    # Customized
                    if isinstance(self.data_type, list):
                        _cltype = random.choice(self.data_type)
                    elif isinstance(self.data_type, str):
                        _cltype = self.data_type
                    else:
                        raise NotImplementedError()
                sampled_text = self.unpack_text(text_dict, _cltype)
                unpacked[_head_type].append(sampled_text)
                unpacked[_cltypename].append(_cltype)
        
        # unpack query seperately, for Contrastive Learning
        if 'query' in examples[0]:
            unpacked['query'] = [{'text': text_dict['query']} for text_dict in examples]

        return self.process(unpacked)
    
    @staticmethod
    def unpack_text(text_dict: dict, _cltype: str):
        # Produce a unpacked text line, for tokenizing afterwards
        if isinstance(text_dict[_cltype][0], int):
            sampled_text = {'text': text_dict[_cltype]}
        elif isinstance(text_dict[_cltype][0], list):
            sampled_text = {'text': random.sample(text_dict[_cltype], 1)[0]}
        return sampled_text


class CotMAEDatasetForSpanSampling(Dataset):
    def __init__(self, dataset, data_args):
        self.dataset = dataset
        self.data_args = data_args
        random = random.Random()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        spans = self.dataset[item]['spans']
        idx = random.randint(0, len(spans)-1)
        return spans[idx]
