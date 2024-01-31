#!/bin/python
import json
import fire
from typing import List, Dict

def convert(
    input_path: str,
    output_path: str,
    type: str = "corpus",   # Choose between corpus & query
):
    fout = open(output_path, 'w')
    idx = 0
    with open(input_path, 'r') as f:
        if type == "corpus":
            f.readline()    # Omit header line of `psgs_w100.tsv`
        
        for line in f:
            lines: List[str] = line.split('\t')
            if type == "corpus":     # Corpus 
                item: Dict[str, str] = {
                    "docid": lines[0],
                    "title": lines[1],
                    "text": lines[2],
                }
            elif type == "query":   # Test
                item: Dict[str, str] = {
                    "query_id": str(idx),
                    "query": lines[0],
                    "answers": eval(lines[1]),
                }
                idx += 1
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

    fout.close()

if __name__ == '__main__':
    fire.Fire(convert)