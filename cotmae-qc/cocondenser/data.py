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

import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForWholeWordMask

rng = random.Random()

@dataclass
class CondenserCollator(DataCollatorForWholeWordMask):
    max_seq_length: int = 512

    def __post_init__(self):
        super(CondenserCollator, self).__post_init__()

        from transformers import BertTokenizer, BertTokenizerFast
        from transformers import RobertaTokenizer, RobertaTokenizerFast
        if isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            self.whole_word_cand_indexes = self._whole_word_cand_indexes_bert
        elif isinstance(self.tokenizer, (RobertaTokenizer, RobertaTokenizerFast)):
            self.whole_word_cand_indexes = self. _whole_word_cand_indexes_roberta
        else:
            raise NotImplementedError(f'{type(self.tokenizer)} collator not supported yet')

        self.specials = self.tokenizer.all_special_tokens

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
    
    def _whole_word_cand_indexes_bert(self, input_tokens: List[str]):
        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token in self.specials:
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
        return cand_indexes

    def _whole_word_cand_indexes_roberta(self, input_tokens: List[str]):
        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token in self.specials:
                raise ValueError('We expect only raw input for roberta for current implementation')

            if i == 0:
                cand_indexes.append([0])
            elif not token.startswith('\u0120'):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
        return cand_indexes

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """

        cand_indexes = self._whole_word_cand_indexes_bert(input_tokens)

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
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
        trunc_left = random.randint(0, trunc)
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

    def __call__(self, examples: List[Dict[str, List[int]]]):
        encoded_examples = []
        masks = []
        mlm_masks = []

        for e in examples:
            e_trunc = self._truncate(e['text'])
            tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e_trunc]
            mlm_mask = self._whole_word_mask(tokens)
            mlm_mask = self._pad([0] + mlm_mask)
            mlm_masks.append(mlm_mask)

            encoded = self.tokenizer.encode_plus(
                e_trunc,
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

        batch = {
            "input_ids": inputs,
            "labels": labels,
            "attention_mask": torch.tensor(masks),
        }

        return batch


@dataclass
class CoCondenserCollator(CondenserCollator):
    def __call__(self, examples):
        examples = sum(examples, [])
        # examples = [{'text': e} for e in examples]

        return super(CoCondenserCollator, self).__call__(examples)


class CoCondenserDataset(Dataset):
    def __init__(self, dataset, data_args):
        self.dataset = dataset
        self.data_args = data_args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if self.data_args.data_type == 'spans':
            spans = self.dataset[item]['spans']
            return random.sample(spans, 2)
        else:
            ret_spans = []
            ret_spans.append({'text': self.dataset[item]['anchor']})

            _cltype = self.get_cltype(self.data_args.data_type)
            sampled_text  = self.unpack_text(self.dataset[item], _cltype)
            ret_spans.append(sampled_text)

            # if self.data_args.data_type == 'query_random_sampled':
            #     # data_type = random.choice(['random_sampled', 'query'])
            #     if random.random() <= self.data_args.query_percent:
            #         data_type = 'query'
            #     else:
            #         data_type = 'random_sampled'
            # elif self.data_args.data_type == 'query_mix_context':
            #     data_type = random.choice(['random_sampled', 'nearby', 'overlap', 'query'])
            # else:
            #     data_type = self.data_args.data_type
            
            # if type(self.dataset[item][data_type][0]).__name__ == 'list':
            #     ret_spans.append(random.choice(self.dataset[item][data_type]))
            # else:
            #     ret_spans.append(self.dataset[item][data_type])  

            return ret_spans

    @staticmethod
    def get_cltype(data_type: str):
        # Get a corresponding span type according to input `data_type`
        contextual_span_names = ['random_sampled', 'nearby', 'overlap']
        if data_type == 'origin_context_query':
            # Original + Contextual + query
            _rand = random.random()
            if _rand <= 0.33:
                _cltype = 'anchor'  # 33% prob to sample the original span
            elif _rand <= 0.66:
                _cltype = 'query'  # 33% prob to sample the query span
            else:   # 33% prob to sample a contextual span
                # Randomly pick a corpus cl span among 'random_sampled', 'nearby', 'overlap'
                _cltype = random.choice(contextual_span_names)
        elif data_type == 'ori_mixed':
            # Original + Contextual
            if random.random() <= 0.5:
                _cltype = 'anchor'  # 50% prob to sample the original span
            else:   # 50% prob to sample a contextual span
                # Randomly pick a corpus cl span among 'random_sampled', 'nearby', 'overlap'
                _cltype = random.choice(contextual_span_names)
        elif data_type == 'mixed' or data_type == 'cotmae':
            # Contextual
            # Randomly pick a corpus cl span among 'random_sampled', 'nearby', 'overlap'
            _cltype = random.choice(contextual_span_names)
        else:
            # Customized
            if isinstance(data_type, dict):
                _cltype = random.choices(data_type['data_type'], weights=data_type['weight'])[0]
            elif isinstance(data_type, list):
                _cltype = random.choice(data_type)
            elif isinstance(data_type, str):
                _cltype = data_type
            else:
                raise NotImplementedError()
        return _cltype
    
    @staticmethod
    def unpack_text(text_dict: dict, _cltype: str):
        # Produce a unpacked text line, for tokenizing afterwards
        if isinstance(text_dict[_cltype][0], int):
            sampled_text = {'text': text_dict[_cltype]}
        elif isinstance(text_dict[_cltype][0], list):
            sampled_text = {'text': random.sample(text_dict[_cltype], 1)[0]}
        return sampled_text
