#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Download and Tokenize Wikipedia + BookCorpus

@Author  :   Ma (Ma787639046@outlook.com)
'''

import random
import nltk
import argparse
from typing import List
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, set_seed
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from collections import defaultdict
from functools import partial

set_seed(42)
print("Downloading NLTK files...")
nltk.download('punkt')      # Needs Internet access

def pad_one_article(tokenized: List[List[int]], 
                    sentences: List[str],
                    maxlen: int,
                    short_seq_prob: float=0.15,
                    ):
    # 'spans' of this article:
    #           [   [sentence1: List[int] or List[str], sentence2, sentence3 ...] → span1,
    #               [sentence1, sentence2, sentence3 ...] → span2
    #           ]
    # 
    assert len(tokenized) == len(sentences)

    all_span_input_ids = []
    all_span_raw_texts = []

    tokenized_input_ids = []
    raw_text_spans = []

    
    target_seq_len = random.randint(2, maxlen) if random.random() <= short_seq_prob else maxlen
    
    curr_len = 0

    for idx in range(len(tokenized)):
        input_ids = tokenized[idx]
        raw_text = sentences[idx]
        if len(input_ids) == 0:
            continue
        tokenized_input_ids.append(input_ids)
        raw_text_spans.append(raw_text)

        curr_len += len(input_ids)
        if curr_len > target_seq_len:
            all_span_input_ids.append(tokenized_input_ids)
            all_span_raw_texts.append(raw_text_spans)
            curr_len = 0
            tokenized_input_ids = []
            raw_text_spans = []
            target_seq_len = random.randint(2, maxlen) if random.random() <= short_seq_prob else maxlen
            
    if len(tokenized_input_ids) > 0:
        all_span_input_ids.append(tokenized_input_ids)
        all_span_raw_texts.append(raw_text_spans)
    
    return all_span_input_ids, all_span_raw_texts

def make_text_spans(all_span_input_ids, # List of padded sentence ids
                    all_span_raw_texts,    # List of padded single sentences
                    save_text = True,
                    ):
    if not all_span_input_ids:
        return None
    
    # Concat Texts
    anchor_input_ids = [sum(i, []) for i in all_span_input_ids]
    anchor_raw_texts = [" ".join(i) for i in all_span_raw_texts]
    assert len(anchor_input_ids) == len(anchor_raw_texts)

    if save_text:       # Save as Text format
        return {'text': anchor_raw_texts}
    else:
        return {'text': anchor_input_ids}

def make_text_splits(examples: dict, 
                     maxlen: int, 
                     short_seq_prob: float = 0.15):
    ret = defaultdict(list)
    assert len(examples['input_ids']) == len(examples['raw_texts'])

    for idx in range(len(examples['input_ids'])):
        all_span_input_ids = examples['input_ids'][idx]
        all_span_raw_texts = examples['raw_texts'][idx]

        all_span_input_ids, all_span_raw_texts = pad_one_article(all_span_input_ids, all_span_raw_texts, maxlen=maxlen, short_seq_prob=short_seq_prob)
        spans_w_context = make_text_spans(all_span_input_ids, all_span_raw_texts)
        for k, v in spans_w_context.items():
            ret[k].extend(v)
    return ret

def create_wiki_data(tokenizer: PreTrainedTokenizerBase,
                     maxlen: int,
                     nproc: int = 20,
                     short_seq_prob: float = 0.15):

    def tokenize_wiki_data(examples):
        sentences = nltk.sent_tokenize(examples["text"])
        return {
            "input_ids": tokenizer(sentences, 
                                   add_special_tokens=False, 
                                   truncation=False, 
                                   return_attention_mask=False,
                                   return_token_type_ids=False
                                   )["input_ids"],
            "raw_texts": sentences
        }

    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    dataset = dataset.map(tokenize_wiki_data, num_proc=nproc, remove_columns=dataset.column_names)
    dataset = dataset.map(partial(make_text_splits, maxlen=maxlen, short_seq_prob=short_seq_prob), 
                          num_proc=nproc, batched=True, 
                          remove_columns=["input_ids", "raw_texts"])

    return dataset

def create_bookcorpus_data(tokenizer: PreTrainedTokenizerBase,
                           maxlen: int,
                           nproc: int = 20,
                           short_seq_prob: float = 0.15):
    
    def tokenize_bookcorpus_data(examples):
        sentences = examples["text"]
        return {
            "input_ids": [tokenizer(sentences, 
                                   add_special_tokens=False, 
                                   truncation=False, 
                                   return_attention_mask=False,
                                   return_token_type_ids=False
                                   )["input_ids"]],
            "raw_texts": [sentences]
        }

    dataset = load_dataset("bookcorpus", split="train")
    dataset = dataset.map(tokenize_bookcorpus_data, num_proc=nproc, batched=True, batch_size=200, remove_columns=dataset.column_names)
    dataset = dataset.map(partial(make_text_splits, maxlen=maxlen, short_seq_prob=short_seq_prob), 
                          num_proc=nproc, batched=True, 
                          remove_columns=["input_ids", "raw_texts"])

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_to',
                        type=str,
                        required=True,
                        )
    parser.add_argument('--maxlen',
                        default=512,
                        type=int,
                        required=False)
    parser.add_argument('--short_seq_prob',
                        default=0.15,
                        type=float,
                        required=False,
                        help="Keep some short length sentences, for better performance")
    parser.add_argument('--n_proc',
                        default=30,
                        type=int,
                        required=False,
                        help="Multi-process thread nums")
    parser.add_argument('--tokenizer',
                        default="bert-base-uncased",
                        required=False)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    wiki = create_wiki_data(tokenizer=tokenizer, maxlen=args.maxlen, nproc=args.n_proc, short_seq_prob=args.short_seq_prob)
    bookcorpus = create_bookcorpus_data(tokenizer=tokenizer, maxlen=args.maxlen, nproc=args.n_proc, short_seq_prob=args.short_seq_prob)

    wikibook = concatenate_datasets([wiki, bookcorpus])
    wikibook.to_json(args.save_to, force_ascii=False)
