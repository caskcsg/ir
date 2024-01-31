# Jsonl Text files holds untokenized text format
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
import os
import csv
import json
import random
from tqdm import tqdm
from multiprocessing import Pool
from argparse import ArgumentParser
from dataclasses import dataclass

import datasets
from transformers import set_seed

@dataclass
class JsonlTrainPreProcessor:
    query_file: str
    collection_file: str

    columns = ['text_id', 'title', 'text']
    title_field = 'title'
    text_field = 'text'

    def __post_init__(self):
        self.queries = self.read_queries(self.query_file)
        self.collection = datasets.load_dataset(
            'csv',
            data_files=self.collection_file,
            column_names=self.columns,
            delimiter='\t',
        )['train']

    @staticmethod
    def read_queries(queries):
        qmap = {}
        with open(queries) as f:
            for l in f:
                qid, qry = l.strip().split('\t')
                qmap[qid] = qry
        return qmap

    @staticmethod
    def read_qrel(relevance_file):
        qrel = {}
        with open(relevance_file, encoding='utf8') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for [topicid, _, docid, rel] in tsvreader:
                assert rel == "1"
                if topicid in qrel:
                    qrel[topicid].append(docid)
                else:
                    qrel[topicid] = [docid]
        return qrel

    def get_query(self, query_id):
        return query_id, self.queries[query_id]

    def get_passage(self, docid):
        entry = self.collection[int(docid)]
        title = entry[self.title_field]
        title = "" if title is None else title
        text = entry[self.text_field]
        return {"docid": docid, "title": title, "text": text}

    def process_one(self, train):
        q, pp, nn = train[:3]
        query_id, query = self.get_query(q)
        train_example = {
            'query_id': query_id,
            'query': query,
            'positive_passages': [self.get_passage(p) for p in pp],
            'negative_passages': [self.get_passage(n) for n in nn],
        }

        return json.dumps(train_example, ensure_ascii=False)


set_seed(42)
parser = ArgumentParser()
parser.add_argument('--negative_file', required=True)
parser.add_argument('--qrels', required=True)
parser.add_argument('--queries', required=True)
parser.add_argument('--collection', required=True)
parser.add_argument('--save_to', required=True)

parser.add_argument('--n_sample', type=int, default=200)
parser.add_argument('--mp_chunk_size', type=int, default=500)
parser.add_argument('--shard_size', type=int, default=45000)

args = parser.parse_args()


qrel = JsonlTrainPreProcessor.read_qrel(args.qrels)


def read_line(l):
    q, nn = l.strip().split('\t')
    nn = nn.split(',')
    random.shuffle(nn)
    return q, qrel[q], nn[:args.n_sample]


processor = JsonlTrainPreProcessor(
    query_file=args.queries,
    collection_file=args.collection,
)

counter = 0
shard_id = 0
f = None
os.makedirs(args.save_to, exist_ok=True)

with open(args.negative_file) as nf:
    pbar = tqdm(map(read_line, nf))
    with Pool() as p:
        for x in p.imap(processor.process_one, pbar, chunksize=args.mp_chunk_size):
            counter += 1
            if f is None:
                f = open(os.path.join(args.save_to, f'split{shard_id:02d}.jsonl'), 'w')
                pbar.set_description(f'split - {shard_id:02d}')
            f.write(x + '\n')

            if counter == args.shard_size:
                f.close()
                f = None
                shard_id += 1
                counter = 0

if f is not None:
    f.close()