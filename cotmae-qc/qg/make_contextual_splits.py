#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Make data for Cot-MAE
Step 1: Generate Text splits & Contextual spans

This data holds spans of same document, to easily add contrastive learning objects if needed.

@Author  :   Ma (Ma787639046@outlook.com)
'''

import os
import random
import nltk
import argparse
from typing import List
import json
from transformers import AutoTokenizer
from multiprocessing import Pool
from tqdm import tqdm

def wc_count(file_name):
    import subprocess
    out = subprocess.getoutput("wc -l %s" % file_name)
    return int(out.split()[0])

seed=42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

print("Downloading NLTK files...")
nltk.download('punkt')

parser = argparse.ArgumentParser()
parser.add_argument('--file',
                    type=str,
                    required=True,
                    help="Path to txt data file. One line per article format.")
parser.add_argument('--save_to',
                    type=str,
                    required=True,
                    )
parser.add_argument('--maxlen',
                    default=144,
                    type=int,
                    required=False)
parser.add_argument('--window_size',
                    default=100,
                    type=int,
                    required=False,
                    help="Random window size of random sampled spans. Default 100 will be \
                          large enough for sampling the whole articles ",
                    )
parser.add_argument('--short_sentence_prob',
                    default=0.15,
                    type=float,
                    required=False,
                    help="Keep some short length sentences, for better performance")
parser.add_argument('--tokenizer',
                    default="bert-base-uncased",
                    required=False)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
rng = random.Random()

def _base_encode_one_span(line: str, maxlen=args.maxlen) -> List[List[List[int]]]:
    # 'spans' of this article:
    #           [   [sentence1: List[int], sentence2, sentence3 ...] → span1,
    #               [sentence1, sentence2, sentence3 ...] → span2
    #           ]
    # 
    sentences = nltk.sent_tokenize(line.strip())
    tokenized = [
        tokenizer(
            s,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"] for s in sentences
    ]

    all_span_input_ids = []
    all_span_raw_texts = []

    tokenized_input_ids = []
    raw_text_spans = []

    
    target_seq_len = rng.randint(2, maxlen) if rng.random() <= args.short_sentence_prob else maxlen
    
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
            target_seq_len = rng.randint(2, maxlen) if rng.random() <= args.short_sentence_prob else maxlen
            
    if len(tokenized_input_ids) > 0:
        all_span_input_ids.append(tokenized_input_ids)
        all_span_raw_texts.append(raw_text_spans)

    # if len(all_spans) < 2:
    #     return None
    
    return all_span_input_ids, all_span_raw_texts

def encode_one(line):
    # { 'spans':
    #           [   [token1:int, token2, token3 ...] → span1,
    #               [token1, token2, token3 ...] → span2
    #           ]
    # }
    all_span_input_ids, all_span_raw_texts = _base_encode_one_span(line)
    if not all_span_input_ids:
        return None
    return json.dumps({'spans': [sum(i, []) for i in all_span_input_ids]})

def encode_three_corpus_aware_type(line, maxlen=args.maxlen):
    # Article:
    # { 'spans':
    #           [   {'anchor': span1, 
    #                'random_sampled': random_sampled_span,     # Random sampled spans
    #                'nearby': nearby_span,     # ICT like, if short sentence, it will 50% prob keep this 
    #                                           # short sentence in the nearby_span
    #                'overlap': overlaped_span  # partial overlap with `anchor`, Contriver use this method
    #               },
    #               ……
    #           ]
    # }
    all_span_input_ids, all_span_raw_texts = _base_encode_one_span(line)
    if not all_span_input_ids:
        return None
    
    # Concat anchors
    anchor_input_ids = [sum(i, []) for i in all_span_input_ids]
    anchor_raw_texts = [" ".join(i) for i in all_span_raw_texts]
    assert len(anchor_input_ids) == len(anchor_raw_texts)

    # Sample spans randomly
    random_sampled_spans = []
    window_idx_base = [i for i in range(args.window_size * -1, 0, 1)] + [i for i in range(1, args.window_size + 1, 1)]
    for i in range(len(anchor_input_ids)):
        window_idxs = [j + i for j in window_idx_base]
        window_idxs = list(filter(lambda x: x >= 0 and x < len(anchor_input_ids), window_idxs))
        if window_idxs:
            random_sampled = anchor_raw_texts[rng.sample(window_idxs, 1)[0]]
        else:   # Short `anchors` only has one span, degenerate to itself
            random_sampled = anchor_raw_texts[i]
        random_sampled_spans.append(random_sampled)
    
    # Sample spans from anchors' nearby
    nearby_spans = []
    nearby_idx_base = [-1, 1]
    for i in range(len(anchor_raw_texts)):
        nearby_idxs = [j + i for j in nearby_idx_base]
        nearby_idxs = list(filter(lambda x: x >= 0 and x < len(anchor_raw_texts), nearby_idxs))
        if nearby_idxs:
            nearby_sampled = anchor_raw_texts[rng.sample(nearby_idxs, 1)[0]]
        else:   # Short `anchors` only has one span, degenerate to itself
            nearby_sampled = anchor_raw_texts[i]
        nearby_spans.append(nearby_sampled)
    
    # Sample partial overlaped spans
    overlap_spans = []
    for i in range(len(all_span_input_ids)):
        if i < len(all_span_input_ids)-1:
            # How many sentences are from the anchor
            ceil_boundary = len(all_span_input_ids[i]) // 2
            if ceil_boundary < 2:
                idx_anchor = 1
            else:
                idx_anchor = rng.randint(1, ceil_boundary)     # len(spans[i])//2 means we want overlap 
                                                                    # precent to be <= 50% in most cases
            overlap = []
            overlap.extend(all_span_raw_texts[i][-idx_anchor:])
            remain_target_len = maxlen - sum([len(_ids) for _ids in all_span_input_ids[i][-idx_anchor:]])

            for j in range(i+1, len(all_span_raw_texts)):
                if remain_target_len <= 0:
                    break
                for idx in range(len(all_span_input_ids[j])):
                    overlap.append(all_span_raw_texts[j][idx])
                    remain_target_len -= len(all_span_input_ids[j][idx])
                    if remain_target_len <= 0:
                        break
        else:   # last span: special case
            ceil_boundary = len(all_span_input_ids[i]) // 2
            if ceil_boundary < 2:
                idx_anchor = 1
            else:
                idx_anchor = rng.randint(1, ceil_boundary)     # len(spans[i])//2 means we want overlap 
                                                                    # precent to be <=50% in most cases
            overlap = []
            overlap.extend(all_span_raw_texts[i][:idx_anchor])
            remain_target_len = maxlen - sum([len(_ids) for _ids in all_span_input_ids[i][:idx_anchor]])

            for j in range(i-1, -1, -1):
                if remain_target_len <= 0:
                    break
                for k in range(len(all_span_input_ids[j])-1, -1, -1):
                    overlap.insert(0, all_span_raw_texts[j][k])
                    remain_target_len -= len(all_span_input_ids[j][k])
                    if remain_target_len <= 0:
                        break
        overlap_spans.append(" ".join(overlap))
    
    lengths = [len(anchor_raw_texts), len(random_sampled_spans), len(nearby_spans), len(overlap_spans)]
    assert len(set(lengths)) == 1, print(lengths)

    final_spans = [{'anchor': anch, 
                    'random_sampled': rand, 
                    'nearby': near, 
                    'overlap': overlap
                    } for anch, rand, near, overlap in zip(anchor_raw_texts, random_sampled_spans, nearby_spans,
                                                        overlap_spans)]

    return final_spans

with open(args.save_to, 'w') as f:
    # Multiprocess is highly recommended
    cnt = 0
    with Pool() as p:
        all_tokenized = p.imap_unordered(
            encode_three_corpus_aware_type,
            tqdm(open(args.file), total=wc_count(args.file)),
            chunksize=1000,
        )
        for spans in all_tokenized:
            if not spans:
                continue
            for _span in spans:
                _out = {"docid": str(cnt)}
                _out.update(_span)
                f.write(json.dumps(_out, ensure_ascii=False) + '\n')
                cnt += 1

