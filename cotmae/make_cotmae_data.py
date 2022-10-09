#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Make data for Cot-MAE

@Author  :   Ma (Ma787639046@outlook.com)
'''

import os
import random
import nltk
import argparse
from typing import List
from math import floor
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
                    required=True,
                    help="Path to txt data file. One line per article format.")
parser.add_argument('--save_to',
                    required=True)
parser.add_argument('--maxlen',
                    default=128,
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

    all_spans = []
    tokenized_span = []

    if rng.random() <= args.short_sentence_prob:
        target_seq_len = rng.randint(2, maxlen)
    else:
        target_seq_len = maxlen
    
    curr_len = 0

    for sent in tokenized:
        if len(sent) == 0:
            continue
        tokenized_span.append(sent)
        curr_len += len(sent)
        if curr_len > target_seq_len:
            all_spans.append(tokenized_span)
            curr_len = 0
            tokenized_span = []
            if rng.random() <= args.short_sentence_prob:
                target_seq_len = rng.randint(2, maxlen)
            else:
                target_seq_len = maxlen
            
    if len(tokenized_span) > 0:
        all_spans.append(tokenized_span)

    if len(all_spans) < 2:
        return None
    
    return all_spans

def encode_one(line):
    # { 'spans':
    #           [   [token1:int, token2, token3 ...] → span1,
    #               [token1, token2, token3 ...] → span2
    #           ]
    # }
    spans = _base_encode_one_span(line)
    if spans is None:
        return None
    return json.dumps({'spans': [sum(i, []) for i in spans]})

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
    spans = _base_encode_one_span(line)
    if spans is None:
        return None
    
    # Concat anchors
    anchors = [sum(i, []) for i in spans]

    random_sampled_spans = []
    window_idx_base = [i for i in range(args.window_size * -1, 0, 1)] + [i for i in range(1, args.window_size + 1, 1)]
    for i in range(len(anchors)):
        window_idxs = [j + i for j in window_idx_base]
        window_idxs = list(filter(lambda x: x >= 0 and x < len(anchors), window_idxs))
        if window_idxs:
            random_sampled = anchors[rng.sample(window_idxs, 1)[0]]
        else:
            random_sampled = anchors[i]
        random_sampled_spans.append(random_sampled)
    
    # Sample spans from anchors' nearby
    nearby_spans = []
    nearby_idx_base = [-1, 1]
    for i in range(len(anchors)):
        nearby_idxs = [j + i for j in nearby_idx_base]
        nearby_idxs = list(filter(lambda x: x >= 0 and x < len(anchors), nearby_idxs))
        nearby_sampled = anchors[rng.sample(nearby_idxs, 1)[0]]
        nearby_spans.append(nearby_sampled)
    
    # Sample partial overlaped spans
    overlap_spans = []
    for i in range(len(spans)):
        if i < len(spans)-1:
            # How many sentences are from the anchor
            ceil_boundary = floor(len(spans[i])/2)
            if ceil_boundary < 2:
                idx_anchor = 1
            else:
                idx_anchor = rng.randint(1, ceil_boundary)     # len(spans[i])/2 means we want overlap 
                                                                    # precent to be <= 50% in most cases
            overlap = []
            overlap.extend(spans[i][-idx_anchor:])
            remain_target_len = maxlen - sum([len(i) for i in overlap])

            for j in range(i+1, len(spans)):
                if remain_target_len <= 0:
                    break
                for sent in spans[j]:
                    overlap.append(sent)
                    remain_target_len -= len(sent)
                    if remain_target_len <= 0:
                        break
        else:   # last span: special case
            ceil_boundary = floor(len(spans[i])/2)
            if ceil_boundary < 2:
                idx_anchor = 1
            else:
                idx_anchor = rng.randint(1, ceil_boundary)     # len(spans[i])/2 means we want overlap 
                                                                    # precent to be <=50% in most cases
            overlap = []
            overlap.extend(spans[i][:idx_anchor])
            remain_target_len = maxlen - sum([len(i) for i in overlap])

            for j in range(i-1, -1, -1):
                if remain_target_len <= 0:
                    break
                for k in range(len(spans[j])-1, -1, -1):
                    overlap.insert(0, spans[j][k])
                    remain_target_len -= len(spans[j][k])
                    if remain_target_len <= 0:
                        break
        overlap_spans.append(sum(overlap, []))
    
    lengths = [len(anchors), len(random_sampled_spans), len(nearby_spans), len(overlap_spans)]
    assert len(set(lengths)) == 1, print(lengths)
    final_spans = [{'anchor': anch, 'random_sampled': rand, 'nearby': near, 'overlap': overlap} \
                    for anch, rand, near, overlap in zip(anchors, random_sampled_spans, nearby_spans,
                                                        overlap_spans)]
        
    return json.dumps({'spans': final_spans})

with open(args.save_to, 'w') as f:
    # Multiprocess is highly recommended
    with Pool() as p:
        all_tokenized = p.imap_unordered(
            encode_three_corpus_aware_type,
            tqdm(open(args.file), total=wc_count(args.file)),
            chunksize=1000,
        )
        for x in all_tokenized:
            if x is None:
                continue
            f.write(x + '\n')
    
    # For debug
    # for line in tqdm(open(args.file), total=wc_count(args.file)):
    #     encoded_line = encode_three_corpus_aware_type(line)

