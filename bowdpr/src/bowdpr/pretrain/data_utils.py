#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Dataset for BoW Prediction, Auto-Encoding, Auto-Regression Pre-training for DPR Pre-training

@Author  :   Ma (Ma787639046@outlook.com)
'''
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
from transformers import DataCollatorForWholeWordMask, BatchEncoding
from transformers.utils import logging

from .arguments import DataTrainingArguments

logger = logging.get_logger(__name__)

@dataclass
class BoWPredictionCollator(DataCollatorForWholeWordMask):
    max_seq_length: int = 512
    enc_head_mask_ratio: float = 0.15
    data_type: str = 'text'       # Input field for the decoder head.

    data_args: DataTrainingArguments = None

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

    def _truncate(
        self, 
        example: List[int], 
        tgt_len: int=512, 
        ignore_special_token_on_left=True,
        ignore_special_token_on_right=True,
    ):
        # Calculate total tokens to truncate
        if len(example) <= tgt_len:
            return example
        trunc = len(example) - tgt_len
        trunc_left = random.randint(0, trunc)
        trunc_right = trunc - trunc_left

        # Whether to ignore special token on both side
        if ignore_special_token_on_left:
            special_token_left = example[0]
            example = example[1:]
        
        if ignore_special_token_on_right:
            special_token_right = example[-1]
            example = example[:-1]

        # Truncate
        truncated = example[trunc_left:]
        if trunc_right > 0:
            truncated = truncated[:-trunc_right]
        
        # Concatenate special token back
        if ignore_special_token_on_left:
            truncated.insert(0, special_token_left)
        
        if ignore_special_token_on_right:
            truncated.append(special_token_right)
        
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
                              ):
        # Tokenize examples if the input line is not pre-tokenized
        if not isinstance(examples[0], list):
            tokenized = self.tokenizer(examples, add_special_tokens=True)["input_ids"]
        else:
            tokenized = examples

        # Encode a batch of encoded_examples with Whole Word Mask
        encoded_examples: List[List[int]] = []
        mlm_masks: List[List[int]] = []
        padded_mlm_mask: List[List[int]] = []

        # Dynamic padding
        tgt_len = max(len(e) for e in tokenized)
        tgt_len = min(tgt_len, self.max_seq_length)

        # WWM
        for idx, e in enumerate(tokenized):
            e_trunc = self._truncate(e, tgt_len=tgt_len) # Truncate on both side, as implemented in BERT
            tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e_trunc]
            mlm_mask = self._whole_word_mask(tokens, mlm_probability=mlm_prob)  # WWM
            
            encoded_examples.append(e_trunc)
            mlm_masks.append(mlm_mask)
            padded_mlm_mask.append(self._pad(mlm_mask, tgt_len=tgt_len))
        
        # Pad `encoded_examples`
        batch: BatchEncoding = self.tokenizer.pad(
            {"input_ids": encoded_examples},
            padding="max_length",
            max_length=tgt_len,
            return_tensors=self.return_tensors,
        )
        input_ids_unmasked = batch["input_ids"].clone()

        # Mask with WWM
        batch["input_ids"], batch["labels"] = self.mask_tokens(
            batch["input_ids"],
            torch.tensor(padded_mlm_mask, dtype=torch.long)
        )

        return batch, input_ids_unmasked, encoded_examples, mlm_masks

    @staticmethod
    def _is_stopword_or_special_token(_id: int, tokenizer_type='bert'):
        if tokenizer_type == "bert":
            if _id <= 998:
                return True
            else:
                return False
        else:
            raise NotImplementedError()

    # Get Parameter-free BoW Weights
    def _get_bow_weights(self, batch: BatchEncoding, encoded_examples: List[List[int]]):
        # Get BoW weights
        weights = torch.zeros([batch["input_ids"].shape[0], self.tokenizer.vocab_size], dtype=torch.float32, device=batch["input_ids"].device)
        for idx, _input_ids in enumerate(encoded_examples):
            _input_ids_set = [_id for _id in set(_input_ids) if not self._is_stopword_or_special_token(_id)]
            weights[idx][_input_ids_set] = 1 / len(_input_ids_set)  # Scale with bow `len(_input_ids_set)`
        
        return weights

    def process(self, examples: Tuple[List[Dict[str, List[int]]], Dict[str, List[Dict[str, List[int]]]]]):
        batch, input_ids_unmasked, encoded_examples, mlm_masks = self.encode_batch_examples(examples=examples['anchor'], mlm_prob=self.mlm_probability)
        
        # Compute BoW weights
        bow_weights = self._get_bow_weights(batch, encoded_examples)

        rets = {
            "encoder_input": batch,
            "bow_weights": bow_weights,
        }
        
        return rets
    
    def __call__(self, examples):
        # Unpack
        anchors: List[str] = [text_dict["text"] for text_dict in examples]
        # Pre-process
        return self.process({"anchor": anchors})


@dataclass
class AutoEncodingCollator(BoWPredictionCollator):
    def process(self, examples: Tuple[List[Dict[str, List[int]]], Dict[str, List[Dict[str, List[int]]]]]):
        batch, input_ids_unmasked, encoded_examples, mlm_masks = self.encode_batch_examples(examples=examples['anchor'], mlm_prob=self.mlm_probability)

        # AE-Dec
        enc_head_batch, enc_head_input_ids_unmasked, enc_head_encoded_examples, enc_head_mlm_masks = self.encode_batch_examples(examples=examples['head'], mlm_prob=self.enc_head_mask_ratio)

        rets = {
            "encoder_input": batch,
            "ae_dec_head_input": enc_head_batch,
        }
        
        return rets
    
    def __call__(self, examples):
        anchors: List[str] = []
        head_inputs: List[str] = []
        for text_dict in examples:
            # Anchor Text for Encoder
            anchors.append(text_dict['text'])  # List[str]

            # Contextual Text for Decoder
            if self.data_type == 'cotmae':
                # Randomly pick a corpus cl span among 'random_sampled', 'nearby', 'overlap'
                _cltype = random.choice(['random_sampled', 'nearby', 'overlap'])
            # Customized
            elif isinstance(self.data_type, list):
                _cltype = random.choice(self.data_type)
            elif isinstance(self.data_type, str):
                _cltype = self.data_type
            else:
                raise NotImplementedError()

            sampled_text = text_dict[_cltype]
            if isinstance(sampled_text, list):
                sampled_text = random.choice(sampled_text)
            assert isinstance(sampled_text, str)
            if sampled_text == "":
                sampled_text = "This is a padding document."

            head_inputs.append(sampled_text)
        
        unpacked = {
            'anchor': anchors,
            'head': head_inputs,
        }

        return self.process(unpacked)


@dataclass
class AutoRegressionCollator(BoWPredictionCollator):
    # Tokenize for AR-Dec
    def _tokenize_ar_dec(self, examples: List[Dict[str, List[int]]]):
        # Tokenize examples if the input line is not pre-tokenized
        if not isinstance(examples[0], list):
            tokenized = self.tokenizer(examples, add_special_tokens=True)["input_ids"]
        else:
            tokenized = examples

        encoded_examples = [self._truncate(e, tgt_len=self.max_seq_length) for e in tokenized]
        
        # Truncate as BERT & Encode as batch
        batch_encoding = self.tokenizer.pad(
                            {"input_ids": encoded_examples},
                            padding='longest',
                            max_length=self.max_seq_length,
                            return_tensors=self.return_tensors,
                        )
        
        # labels for generative AR-Dec head is natually its input ids sliced with [..., 1:, :]
        # we will do slice (or logits shift) inside the func `casual_loss`
        # Here we fill the labels (or input ids of AR-Dec) [PAD] area with -100, to avoid loss calculatation
        # on [PAD] area by CrossEntropy loss function
        # Ignore klloss on [PAD]:
        # [cls] xxx xxx xxx [SEP] [PAD] [PAD]
        #   0    0   0   0    0   -100  -100
        # Size [bs, seq_len]
        dec_head_labels = batch_encoding['input_ids'].clone()
        dec_head_labels.masked_fill_(~(batch_encoding['attention_mask'].bool()), -100)
        batch_encoding["labels"] = dec_head_labels

        return batch_encoding

    def process(self, examples: Tuple[List[Dict[str, List[int]]], Dict[str, List[Dict[str, List[int]]]]]):
        batch, input_ids_unmasked, encoded_examples, mlm_masks = self.encode_batch_examples(examples=examples['anchor'], mlm_prob=self.mlm_probability)
        
        # AR-Dec
        ar_dec_head_batch = self._tokenize_ar_dec(examples['head'])

        rets = {
            "encoder_input": batch,
            "ar_dec_head_input": ar_dec_head_batch,
        }
        
        return rets
    
    def __call__(self, examples):
        anchors: List[str] = []
        head_inputs: List[str] = []
        for text_dict in examples:
            # Anchor Text for Encoder
            anchors.append(text_dict['text'])  # List[str]

            # Contextual Text for Decoder
            if self.data_type == 'cotmae':
                # Randomly pick a corpus cl span among 'random_sampled', 'nearby', 'overlap'
                _cltype = random.choice(['random_sampled', 'nearby', 'overlap'])
            # Customized
            elif isinstance(self.data_type, list):
                _cltype = random.choice(self.data_type)
            elif isinstance(self.data_type, str):
                _cltype = self.data_type
            else:
                raise NotImplementedError()

            sampled_text = text_dict[_cltype]
            if isinstance(sampled_text, list):
                sampled_text = random.choice(sampled_text)
            assert isinstance(sampled_text, str)
            if sampled_text == "":
                sampled_text = "This is a padding document."

            head_inputs.append(sampled_text)
        
        unpacked = {
            'anchor': anchors,
            'head': head_inputs,
        }

        return self.process(unpacked)

