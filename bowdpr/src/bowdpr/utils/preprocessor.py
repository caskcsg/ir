import json
import csv
import datasets
from dataclasses import dataclass
from typing import List, Tuple, Union, Dict

from .data_utils import read_corpus, build_corpus_idx_to_row, process_tsv_file

@dataclass
class JsonlTrainPreProcessor:
    query_file: str
    collection_file: str
    save_text: bool = False
    save_score: bool = False

    columns = ['_id', 'title', 'text']
    title_field = 'title'
    text_field = 'text'

    def __post_init__(self):
        # Load query corpus
        self.query_dataset: datasets.Dataset = read_corpus(self.query_file)
        self.idx_to_query: Dict[str, int] = build_corpus_idx_to_row(self.query_dataset)
        # Load passage corpus
        self.passage_dataset: datasets.Dataset = read_corpus(self.collection_file)
        self.idx_to_passage: Dict[str, int] = build_corpus_idx_to_row(self.passage_dataset)

    @staticmethod
    def read_qrel(relevance_file) -> Dict[str, List[Tuple[str, str]]]:
        qrel = {}
        with open(relevance_file, encoding='utf8') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            # tsvreader = csv.reader(f, delimiter=" ")
            for row in tsvreader:
                if len(row) == 2:   # MS-MARCO Dev set format: [qid, pid]
                    topicid, docid = row
                    score = 0.
                elif len(row) == 3:   # SentenceTransformers format: [qid, pid, score]
                    topicid, docid, score = row
                elif len(row) == 4: # MS-MARCO Training set format: [qid, 0, pid, 1]
                    topicid, _, docid, score = row
                else:
                    raise NotImplementedError()
                
                if topicid in qrel:
                    qrel[topicid].append((docid, score))
                else:
                    qrel[topicid] = [(docid, score)]
        return qrel

    def get_query(self, query_id):
        ret = {'query_id': query_id}
        if self.save_text:
            ret['query'] = self.query_dataset[self.idx_to_query[query_id]]['text']
        return ret

    def get_passage(self, item: Tuple[str, float]):
        docid, score = item
        ret = {"docid": docid}

        if self.save_text:
            entry = self.passage_dataset[self.idx_to_passage[docid]]
            title = entry[self.title_field]
            if title is None:
                title = ""
            ret["title"] = title
            ret["text"] = entry[self.text_field]
        
        if self.save_score:
            ret["ce_score"] = score
        
        return ret

    def process_one(self, train):
        q, pp, nn = train[:3]
        train_example = self.get_query(q)
        train_example['positive_passages'] = [self.get_passage(p) for p in pp]
        train_example['negative_passages'] = [self.get_passage(n) for n in nn]

        return json.dumps(train_example, ensure_ascii=False)
