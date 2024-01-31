# Jsonl Text files holds untokenized text format, useful for reranking & distill
# For Traditional json format hn, please turn back to build_train_hn.py
# Data Fields:
# {
#     "query_id": string,
#     "query": string,
#     "positive_passages": List of {
#         "docid": string,
#         "title": string,
#         "text": string,
#     },
#     "negative_passages": List of {
#         "docid": string,
#         "title": string,
#         "text": string,
#     },
# }
from argparse import ArgumentParser
from transformers import set_seed
import os
import random
from tqdm import tqdm
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Dict
from .preprocessor import JsonlTrainPreProcessor

def load_ranking(
        rank_file: str, 
        relevance: Dict[str, List[Tuple[str, float]]],   # qid -> List of (pid, score)
        n_sample: int, 
        depth: int,
    ):
    with open(rank_file) as rf:
        lines = iter(rf)
        q_0, p_0, score_0 = next(lines).strip().split()

        curr_q = q_0
        negatives = [] if p_0 in [pid for pid, score in relevance[q_0]] else [(p_0, score_0)]

        while True:
            try:
                q, p, score_curr = next(lines).strip().split()
                if q != curr_q:
                    negatives = negatives[depth[0]:depth[1]]
                    random.shuffle(negatives)
                    yield curr_q, relevance[curr_q], negatives[:n_sample]
                    curr_q = q
                    negatives = [] if p in [pid for pid, score in relevance[q_0]] else [(p, score_curr)]
                else:
                    if p not in [pid for pid, score in relevance[q_0]]:
                        negatives.append((p, score_curr))
            except StopIteration:
                negatives = negatives[depth[0]:depth[1]]
                random.shuffle(negatives)
                yield curr_q, relevance[curr_q], negatives[:n_sample]
                return


parser = ArgumentParser()
parser.add_argument('--hn_file', required=True)
parser.add_argument('--qrels', required=True)
parser.add_argument('--queries', required=True)
parser.add_argument('--collection', required=True)
parser.add_argument('--save_to', required=True)

parser.add_argument('--n_sample', type=int, default=128)
parser.add_argument('--depth', type=int, nargs='+', default=[2, 200])
parser.add_argument('--mp_chunk_size', type=int, default=500, help='Not used by ThreadPool.')
parser.add_argument('--shard_size', type=int, default=45000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_workers', type=int, default=32)
parser.add_argument('--prefix', type=str, default="")
parser.add_argument('--save_text', action='store_true', help='Whether to save texts. Saving texts will comsume a lot of disk spaces. Texts can be fetched from corpus collections during the training process.')
parser.add_argument('--save_score', action='store_true', help='Whether to save the reranker scores from `qrels` and `hn_file` tsv files (qid, pid, score) to field `ce_score`. This is used for the distillation from a reranker to the retirever.')

args = parser.parse_args()

# Check n_sample
args.depth: list
if len(args.depth) == 1:
    args.depth.insert(0, 0)
assert len(args.depth) == 2, f"Please pass 1 or 2 integers for `depth`, as the range of selected negatives."

set_seed(args.seed)

qrel = JsonlTrainPreProcessor.read_qrel(args.qrels)
processor = JsonlTrainPreProcessor(
    query_file=args.queries,
    collection_file=args.collection,
    save_text=args.save_text,
    save_score=args.save_score,
)

counter = 0
shard_id = 0
f = None
os.makedirs(args.save_to, exist_ok=True)

pbar = tqdm(load_ranking(args.hn_file, qrel, args.n_sample, args.depth))
with ThreadPoolExecutor(args.max_workers) as p:
    for x in p.map(processor.process_one, pbar, chunksize=args.mp_chunk_size):
        counter += 1
        if f is None:
            f = open(os.path.join(args.save_to, f'split{shard_id:02d}.{args.prefix}hn.text.jsonl'), 'w')
            pbar.set_description(f'split - {shard_id:02d}')
        f.write(x + '\n')

        if counter == args.shard_size:
            f.close()
            f = None
            shard_id += 1
            counter = 0

if f is not None:
    f.close()