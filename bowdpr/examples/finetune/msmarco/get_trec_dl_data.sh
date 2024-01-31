#!/bin/bash
BASE_DIR=$(dirname "$(dirname "$PWD")")

SCRIPT_DIR=$PWD

mkdir -p marco/downloads/trec_dl_2019
mkdir -p marco/downloads/trec_dl_2020

# get TREC DL 2019 test queries
cd $SCRIPT_DIR/marco/downloads/trec_dl_2019
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz
gunzip msmarco-test2019-queries.tsv.gz
wget https://trec.nist.gov/data/deep/2019qrels-pass.txt

# get TREC DL 2020 test queries
cd $SCRIPT_DIR/marco/downloads/trec_dl_2020
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz
gunzip msmarco-test2020-queries.tsv.gz
wget https://trec.nist.gov/data/deep/2020qrels-pass.txt

# Text format
cd $SCRIPT_DIR
cp marco/downloads/trec_dl_2019/2019qrels-pass.txt marco/text/trec_dl_2019
cp marco/downloads/trec_dl_2020/2020qrels-pass.txt marco/text/trec_dl_2020
python $SCRIPT_DIR/build_query_passage_text.py --file marco/downloads/trec_dl_2019/msmarco-test2019-queries.tsv --save_to marco/text/trec_dl_2019/test.query.jsonl --is_query
python $SCRIPT_DIR/build_query_passage_text.py --file marco/downloads/trec_dl_2020/msmarco-test2020-queries.tsv --save_to marco/text/trec_dl_2020/test.query.jsonl --is_query

# get trec_eval
cd $SCRIPT_DIR/marco/downloads
wget https://trec.nist.gov/trec_eval/trec_eval-9.0.7.tar.gz
tar -xzvf trec_eval-9.0.7.tar.gz
cd trec_eval-9.0.7

make
cp trec_eval ..
