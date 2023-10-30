"""
Make data for Cot-MAE
Step 2: Generate Queries

This code shows how to generate using parallel GPU's for very long corpus.
Multiple GPU's can be used to generate faster!

We use torch.multiprocessing module and define multiple pools for each GPU.
Then we chunk our big corpus into multiple smaller corpus and generate simultaneously.

Important to use the code within the __main__ module!

Usage: CUDA_VISIBLE_DEVICES=0,1 python query_gen_multi_gpu.py
"""
import logging
import argparse
from tqdm import tqdm


from transformers import set_seed
from datasets import load_dataset

from generate import QueryGenerator as QGen
from auto_model import QGenModel



logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Arguments to generate query for a corpus.")

    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default="results/all-with_prefix-t5-base-v1", 
        help="Name or path of the query generator model."
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="data/debug/debug_temp.json", 
        help="Name of the huggingface dataset."
    )
    parser.add_argument(
        "--saved_filename", 
        type=str, 
        default="gen.json", 
        help="Saved filename."
    )
    parser.add_argument(
        "--ques_per_passage",
        type=int,
        default=5,
        help="Number of queries to be generated for each passage in the corpus."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/debug",
        help="Directionary to store the generated queries."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size during query generation."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fix random seed for reproducibility."
    )
    
    args = parser.parse_args()
    return args


#Important, you need to shield your code with if __name__. Otherwise, CUDA runs into issues when spawning new processes.
if __name__ == '__main__':

    args = get_args()

    set_seed(args.seed)

    #### Load the corpus dataset

    logger.info("Loading corpus...")

    dataset = load_dataset('json', data_files=args.dataset, split='train')

    corpus = {}

    for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
        corpus[idx] = {
            'docid': item['docid'],
            'text': item['anchor'],
            'title': '',
            'random_sampled': item['random_sampled'],
            'nearby': item['nearby'],
            'overlap': item['overlap'],
        }
    
    ###########################
    #### Query-Generation  ####
    ###########################

    #Define the model
    gen_prefix = "text2query: "
    model = QGenModel(args.model_name_or_path, gen_prefix=gen_prefix)

    #Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool()

    generator = QGen(model=model)

    #### Query-Generation using Nucleus Sampling (top_k=25, top_p=0.95) ####
    #### https://huggingface.co/blog/how-to-generate
    #### Prefix is required to seperate out synthetic queries and qrels from original
    # prefix = "gen-"

    #### Generating 5 questions per document for all documents in the corpus 
    #### Reminder the higher value might produce diverse questions but also duplicates
    ques_per_passage = args.ques_per_passage

    chunk_size = 5000    # chunks to split within each GPU
    batch_size = args.batch_size       # batch size within a single GPU 

    generator.generate_multi_process(
        corpus=corpus, 
        pool=pool, 
        output_dir=args.output_dir, 
        ques_per_passage=args.ques_per_passage, 
        saved_filename=args.saved_filename,
        batch_size=batch_size,
        chunk_size=chunk_size,
    )
    
    #Optional: Stop the proccesses in the pool
    model.stop_multi_process_pool(pool)

