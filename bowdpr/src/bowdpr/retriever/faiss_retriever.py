import time
import glob
import pickle
import numpy as np
from tqdm import tqdm
from itertools import chain
from functools import partial
from multiprocessing import Pool
from argparse import ArgumentParser
from pathlib import Path

import faiss

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

class BaseFaissIPRetriever:
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatIP(embedding_dim)

    def add(self, p_reps: np.ndarray):
        self.index.add(p_reps)

    def search(self, q_reps: np.ndarray, k: int):
        return self.index.search(q_reps, k)

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int, quiet: bool=False):
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in tqdm(range(0, num_query, batch_size), disable=quiet):
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        return all_scores, all_indices


class FaissRetriever(BaseFaissIPRetriever):
    def __init__(self, embedding_dim: int, factory_str: str, init_reps: np.ndarray):
        self.index = faiss.index_factory(embedding_dim, factory_str)
        self.index.verbose = True
        if not self.index.is_trained:
            self.index.train(init_reps)


def str2bool(str):
    return True if str.lower() == 'true' else False

def _look_up_func(q_dd: list, p_lookup: dict):
    return [str(p_lookup[x]) for x in q_dd]

def search_queries(retriever, q_reps, p_lookup, args):
    num_queries = q_reps.shape[0]
    start_time = time.time()

    if args.batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.batch_size, args.quiet)
    else:
        all_scores, all_indices = retriever.search(q_reps, args.depth)
    
    end_time = time.time()
    qps = num_queries / (end_time - start_time)
    logger.info(f"Num of queries in total: {num_queries}")
    logger.info(f"Search time (second): {end_time - start_time}")
    logger.info(f"Queries per second: {qps}")

    # psg_indices = [[str(p_lookup[x]) for x in q_dd] for q_dd in all_indices]
    # Multiprocess mapping
    with Pool(10) as p:
        psg_indices = p.map(partial(_look_up_func, p_lookup=p_lookup), tqdm(all_indices, desc="Maping indices"), chunksize=1000)

    psg_indices = np.array(psg_indices)
    return all_scores, psg_indices

def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file, quiet: bool=False):
    Path(ranking_save_file).parent.mkdir(parents=True, exist_ok=True)
    with open(ranking_save_file, 'w') as f:
        for qid, q_doc_scores, q_doc_indices in tqdm(zip(q_lookup, corpus_scores, corpus_indices), desc="Writing ranks", disable=quiet):
            score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            for s, idx in score_list:
                f.write(f'{qid}\t{idx}\t{s}\n')

def pickle_load(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup

def main():
    parser = ArgumentParser()
    parser.add_argument('--query_reps', required=True)
    parser.add_argument('--passage_reps', required=True)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--depth', type=int, default=1000)
    parser.add_argument('--save_ranking_to', required=True)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--faiss_num_threads', type=int, default=64)

    args = parser.parse_args()

    # Init Faiss Retriever & Load Corpus
    index_files = glob.glob(args.passage_reps)
    logger.info(f'Pattern match found {len(index_files)} files; loading them into index.')

    p_reps_0, p_lookup_0 = pickle_load(index_files[0])
    p_reps_0 = p_reps_0.astype('float32')
    retriever = BaseFaissIPRetriever(embedding_dim=p_reps_0.shape[1])

    logger.info(f"Faiss num of threads: {args.faiss_num_threads}")
    faiss.omp_set_num_threads(args.faiss_num_threads)

    # Loading to MEM
    shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
    if len(index_files) > 1:
        shards = tqdm(shards, desc='Loading shards into index', total=len(index_files), disable=args.quiet)
    
    look_up = []
    for p_reps, p_lookup in shards:
        p_reps = p_reps.astype('float32')
        retriever.add(p_reps)
        look_up += p_lookup
    
    # Loading to GPU, Multi-GPU for Faiss
    logger.info(f'Using multi GPUs for Faiss')
    co = faiss.GpuMultipleClonerOptions()
    co.shard, co.useFloat16 = True, False
    retriever.index = faiss.index_cpu_to_all_gpus(
        retriever.index,
        co=co,
    )

    # Load queries
    query_files = glob.glob(args.query_reps)
    q_reps, q_lookup = pickle_load(query_files[0])
    for curr_q_reps, curr_q_lookup in map(pickle_load, query_files[1:]):
        q_reps = np.concatenate((q_reps, curr_q_reps))
        q_lookup += curr_q_lookup
    q_reps = q_reps.astype('float32')

    # Start Search
    logger.info('Index Search Start')
    all_scores, psg_indices = search_queries(retriever, q_reps, look_up, args)
    logger.info('Index Search Finished')

    write_ranking(psg_indices, all_scores, q_lookup, args.save_ranking_to)


if __name__ == '__main__':
    main()
