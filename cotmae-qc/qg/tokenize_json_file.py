#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Make data for Cot-MAE
Step 3: Tokenize json text

@Author  :   Ma (Ma787639046@outlook.com)
'''

import os
import random
import argparse
import json
from transformers import AutoTokenizer
from multiprocessing import Pool
from tqdm import tqdm
from collections import defaultdict
from functools import partial

def wc_count(file_name):
    import subprocess
    out = subprocess.getoutput("wc -l %s" % file_name)
    return int(out.split()[0])

seed=42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--file',
                    type=str,
                    default="data/debug/doct5query.json",
                    # required=True,
                    help="Path to processed jsonl file.")
parser.add_argument('--save_to',
                    type=str,
                    default="data/debug/msmarco_mlen128_w_query_5000.json",
                    # required=True,
                    )
parser.add_argument('--tokenizer',
                    default="bert-base-uncased",
                    required=False)
parser.add_argument('--save_queries_as_list',
                    action='store_true')
parser.add_argument('--query_list_depth',
                    type=int,
                    default=None,
                    required=False,
                    help="Numbers of queries per passage to preserve.")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

def tokenize_one_line(line_text: str, save_queries_as_list: bool=False):
    ret = dict()
    for k, v in json.loads(line_text).items():
        if k in ['anchor', 'text']:
            ret['anchor'] = tokenizer.encode(v, add_special_tokens=False, truncation=False,)
        elif k in ['anchor', 'random_sampled', 'nearby', 'overlap']:
            ret[k] = tokenizer.encode(v, add_special_tokens=False, truncation=False,)
        elif 'query' in k:
            tokenized_queries = [tokenizer.encode(i, add_special_tokens=False, truncation=False,) for i in v]
            tokenized_queries = list(filter(lambda x: len(x) > 0, tokenized_queries))
            if len(tokenized_queries) > 0:
                if save_queries_as_list:
                    ret[k] = tokenized_queries
                    if args.query_list_depth:
                        ret[k] = ret[k][:args.query_list_depth]
                else:
                    ret[k] = random.choice(tokenized_queries)
            else:
                return None
    return json.dumps(ret)

with open(args.save_to, 'w') as f:
    # Multiprocess is highly recommended
    with Pool() as p:
        all_tokenized = p.imap_unordered(
            partial(tokenize_one_line, save_queries_as_list=args.save_queries_as_list),
            tqdm(open(args.file), total=wc_count(args.file)),
            chunksize=1000,
        )
        for _span in all_tokenized:
            if _span:
                f.write(_span + '\n')

    # For debug
    # for line in tqdm(open(args.file), total=wc_count(args.file)):
    #     _span = tokenize_one_line(line)
    #     f.write(_span + '\n')