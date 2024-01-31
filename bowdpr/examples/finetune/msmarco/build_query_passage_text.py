# Transform original MS-MARCO queries & passages tsv into Jsonl
# Query Data Fields:
# {
#     "query_id": string,
#     "query": string,
# }

# Passage Data Fields:
# {
#     "docid": string,
#     "title": string,
#     "text": string,
# }

import os
import sys
import json
import fire

import csv
from csv import DictReader
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

def wc_count(file_name):
    import subprocess
    out = subprocess.getoutput("wc -l %s" % file_name)
    return int(out.split()[0])

def main(
    file: str,
    save_to: str,
    n_splits: int=1,
    is_query: bool=False,
):
    n_lines = wc_count(file)
    split_size = (n_lines // n_splits) if (n_lines % n_splits == 0) else (n_lines // n_splits + 1)
    
    if is_query:
        if n_splits > 1:
            print("Ignore `n_splits` when parsing queries.")
            n_splits = 1
        
        os.makedirs(os.path.split(save_to)[0], exist_ok=True)
        fieldnames = ["query_id", "query"]
        with open(file) as fin:
            csv_iter = DictReader(fin, fieldnames=fieldnames, delimiter='\t')
            pbar = tqdm(csv_iter, total=n_lines)
            with open(save_to, 'w') as f:
                for idx, item in enumerate(pbar):
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        os.makedirs(save_to, exist_ok=True)
        fieldnames = ["docid", "title", "text"]
        with open(file) as fin:
            csv_iter = DictReader(fin, fieldnames=fieldnames, delimiter='\t')
            pbar = tqdm(csv_iter, total=n_lines)
            for i in range(n_splits):
                with open(os.path.join(save_to, f'split{i:02d}.jsonl'), 'w') as f:
                    # pbar.set_description(f'split - {i:02d}')
                    for idx, item in enumerate(pbar, start=1):
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        if idx % split_size == 0:
                            break


if __name__ == '__main__':
    fire.Fire(main)
