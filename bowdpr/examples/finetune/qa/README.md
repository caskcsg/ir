# Fine-tune on NQ and TQA

## Download Wiki Corpus and Training datasets

1. Download Wiki Corpus hosted by DPR
```bash
mkdir data
cd data

# Download Jsonl corpus converted and hosted by Tevatron 
wget https://huggingface.co/datasets/Tevatron/wikipedia-nq-corpus/blob/main/corpus.jsonl.gz
gunzip corpus.jsonl.gz
```

2. Download Training datasets from DPR-NQ
```bash
mkdir nq

# Get train set
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz

python convert_hn_format.py --input_path biencoder-nq-train.json.gz --output_path nq/nq-train.jsonl
python convert_hn_format.py --input_path biencoder-nq-dev.json.gz --output_path nq/nq-dev.jsonl

# Get test set
wget "https://huggingface.co/datasets/Tevatron/wikipedia-nq/resolve/main/nq-test.jsonl.gz?download=true" -O "nq/nq-test.jsonl.gz"
gunzip nq/nq-test.jsonl.gz
```

3. Download Training datasets from DPR-TQA
```bash
mkdir trivia

# Get train set
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz

python convert_hn_format.py --input_path biencoder-trivia-train.json.gz --output_path trivia/trivia-train.jsonl
python convert_hn_format.py --input_path biencoder-trivia-dev.json.gz --output_path trivia/trivia-dev.jsonl

# Get test set
wget "https://huggingface.co/datasets/Tevatron/wikipedia-trivia/resolve/main/trivia-test.jsonl.gz?download=true" -O "trivia/trivia-test.jsonl.gz"
gunzip trivia/trivia-test.jsonl.gz
```

## Fine-tune and Test on NQ or TQA
Finetuning on QA datasets involves a two-stage pipeline
 - s1: BM25 negs
 - s2: BM25 negs + Mined negatives from s1

These pipelines are bootstraped in `train_dpr_nq.sh` and `train_dpr_trivia.sh`. The pre-trained checkpoint on Wikipedia and BookCorpus is released in [bowdpr/bowdpr_wiki](https://huggingface.co/bowdpr/bowdpr_wiki). Assume the download checkpoints are already placed in `examples/results/$MODEL_NAME/model` (You can set `$MODEL_NAME` to any name as you wish), please execute the fine-tuning pipelines by just run:

```bash
bash train_dpr_nq.sh $MODEL_NAME
```

Or
```bash
bash train_dpr_trivia.sh $MODEL_NAME
```

## Directly Test a Retreiver
We have released the fine-tuned [NQ](https://huggingface.co/bowdpr/bowdpr_wiki_nqft) and [Trivia](https://huggingface.co/bowdpr/bowdpr_wiki_triviaft) retriever to Huggingface. Please execute the following script to test the retrieval performances.

```bash
# Save the scores of retrieval results to this folder. Change to any temporary folder as you wish
mkdir -p results/nq
bash test_dpr_nq.sh bowdpr/bowdpr_wiki_nqft results/nq
```

```bash
mkdir -p results/trivia
bash test_dpr_nq.sh bowdpr/bowdpr_wiki_triviaft results/trivia
```
