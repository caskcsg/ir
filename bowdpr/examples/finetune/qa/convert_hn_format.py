#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
import gzip
import json
import fire
from tqdm import tqdm
os.chdir(os.path.split(os.path.realpath(__file__))[0])

def convert(
    input_path: str,
    output_path: str,
):
    with gzip.open(input_path, 'rb') as f:
        lines = json.load(f)

    idx = 0

    with open(output_path, "w") as f:
        for line in tqdm(lines):
            if len(line["positive_ctxs"]) < 1 or len(line["hard_negative_ctxs"]) < 1:
                continue

            item = {
                "query_id": str(idx),
                "query": line["question"],
                "answers": line["answers"],
                "positive_passages": [],
                "negative_passages": [],
            }

            for pos_psg in line["positive_ctxs"]:
                curr_psg = {
                    "docid": pos_psg.pop("psg_id") if "psg_id" in pos_psg else pos_psg.pop("passage_id"), # passage_id, psg_id
                    "title": pos_psg.pop("title"),
                    "text": pos_psg.pop("text"),
                }
                item["positive_passages"].append(curr_psg)
            
            for neg_psg in line["hard_negative_ctxs"]:
                curr_psg = {
                    "docid": neg_psg.pop("psg_id") if "psg_id" in neg_psg else neg_psg.pop("passage_id"),
                    "title": neg_psg.pop("title"),
                    "text": neg_psg.pop("text"),
                }
                item["negative_passages"].append(curr_psg)
            
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            idx += 1


if __name__ == '__main__':
    fire.Fire(convert)
