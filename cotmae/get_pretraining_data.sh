#!/bin/bash

mkdir -p data
cd data
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz
gunzip msmarco-docs.tsv.gz
awk '{for(i=3;i<NF;i++)printf("%s ",$i);print $NF}' msmarco-docs.tsv > msmarco-docs.txt
cd ..

python make_cotmae_data.py --file=data/msmarco-docs.txt \
    --save_to=data/msmarco-docs.mlen128.json
