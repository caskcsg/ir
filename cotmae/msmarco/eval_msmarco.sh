#!/bin/bash
# @Author  :   Ma (Ma787639046@outlook.com)
# This script is based on open source tools mentioned bellow. Huge thanks to these great works: 
#    Tevatron (Gao, L., Ma, X., Lin, J., & Callan, J. (2022). Tevatron: An efficient and flexible toolkit for dense retrieval. arXiv preprint arXiv:2203.05765.)
#    Faiss (Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data 7, 3 (2019), 535–547.)
#    GNU parallel (O. Tange (2011): GNU Parallel - The Command-Line Power Tool,;login: The USENIX Magazine, February 2011:42-47.)
BASE_DIR=$(dirname "$PWD")

MODEL_NAME=$1
MODEL_TYPE=bert
TOKENIZER_NAME=bert-base-uncased
if grep -q "roberta" <<< "$MODEL_NAME"; then
  MODEL_TYPE=roberta
  TOKENIZER_NAME=roberta-base
fi
# LR=2e-5 for 8-gpus
LR=2e-5
SEED=42
SPLITS=8
TRAIN_N_PASSAGES=16
MODEL_PATH=$BASE_DIR/results/$MODEL_NAME/model
OUTPUT_DIR=$BASE_DIR/results/$MODEL_NAME/tevatron

set -x
##########################
# Fine-tuning Stage 1
##########################
rm -rf $OUTPUT_DIR/retriever_model_s1
python -m torch.distributed.launch --nproc_per_node=$SPLITS -m \
  tevatron.driver.train \
  --model_name_or_path $MODEL_PATH \
  --output_dir $OUTPUT_DIR/retriever_model_s1 \
  --save_strategy no \
  --train_dir marco/$MODEL_TYPE/train \
  --fp16 \
  --per_device_train_batch_size 8 \
  --learning_rate $LR \
  --num_train_epochs 3 \
  --train_n_passages $TRAIN_N_PASSAGES \
  --negatives_x_device \
  --seed $SEED \
  --dataloader_num_workers 2

##########################
# Test Stage 1 Retriever
##########################
bash test_retriever.sh $OUTPUT_DIR/retriever_model_s1 $OUTPUT_DIR s1_

############################################
# Mining Hard Negatives Using Stage 1 Retriever
############################################
rm -rf $OUTPUT_DIR/encoding/corpus
rm -rf $OUTPUT_DIR/encoding/query
mkdir -p $OUTPUT_DIR/encoding/corpus
mkdir -p $OUTPUT_DIR/encoding/query

parallel --ungroup --halt soon,fail=1 --xapply CUDA_VISIBLE_DEVICES={1} python -m tevatron.driver.encode \
  --output_dir None \
  --model_name_or_path $OUTPUT_DIR/retriever_model_s1 \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --encode_in_path marco/$MODEL_TYPE/corpus/split{2}.json \
  --encoded_save_path $OUTPUT_DIR/encoding/corpus/split{2}.pt \
  ::: $(seq -f "%01g" 0 $(($SPLITS-1))) \
  ::: $(seq -f "%02g" 0 $(($SPLITS-1)))

CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.encode \
  --output_dir None \
  --model_name_or_path $OUTPUT_DIR/retriever_model_s1 \
  --fp16 \
  --q_max_len 32 \
  --encode_is_qry \
  --per_device_eval_batch_size 128 \
  --encode_in_path marco/$MODEL_TYPE/query/train.query.json \
  --encoded_save_path $OUTPUT_DIR/encoding/query/train.pt

# Search
rm -rf $OUTPUT_DIR/train_ranks
mkdir -p $OUTPUT_DIR/train_ranks

parallel --ungroup --halt soon,fail=1 --xapply CUDA_VISIBLE_DEVICES={1} python -m tevatron.faiss_retriever \
  --query_reps $OUTPUT_DIR/encoding/query/train.pt \
  --passage_reps $OUTPUT_DIR/encoding/corpus/split{2}.pt \
  --batch_size 4000 \
  --save_ranking_to $OUTPUT_DIR/train_ranks/{2} \
  ::: $(seq -f "%01g" 0 $(($SPLITS-1))) \
  ::: $(seq -f "%02g" 0 $(($SPLITS-1)))

python -m tevatron.faiss_retriever.reducer \
  --score_dir $OUTPUT_DIR/train_ranks \
  --query $OUTPUT_DIR/encoding/query/train.pt \
  --save_ranking_to $OUTPUT_DIR/train.rank.tsv

# Build HN Train file
rm -rf $OUTPUT_DIR/train-hn
python build_train_hn.py \
  --tokenizer_name $TOKENIZER_NAME \
  --hn_file $OUTPUT_DIR/train.rank.tsv \
  --qrels marco/qrels.train.tsv \
  --queries marco/train.query.txt \
  --collection marco/corpus.tsv \
  --save_to $OUTPUT_DIR/train-hn \
  --seed $SEED \
  --prefix s1

# Copy BM25 negatives to `train-hn` dir
cp marco/$MODEL_TYPE/train/*.json $OUTPUT_DIR/train-hn/

############################################
# Fine-tuning Stage 2 Retriever
############################################
rm -rf $OUTPUT_DIR/retriever_model_s2
python -m torch.distributed.launch --nproc_per_node=$SPLITS -m tevatron.driver.train \
  --model_name_or_path $MODEL_PATH \
  --output_dir $OUTPUT_DIR/retriever_model_s2 \
  --save_strategy no \
  --train_dir $OUTPUT_DIR/train-hn \
  --fp16 \
  --per_device_train_batch_size 8 \
  --learning_rate $LR \
  --num_train_epochs 2 \
  --train_n_passages $TRAIN_N_PASSAGES \
  --negatives_x_device \
  --seed $SEED \
  --dataloader_num_workers 2

############################################
# Encode & Testing Stage 2 Retriever
############################################
bash test_retriever.sh $OUTPUT_DIR/retriever_model_s2 $OUTPUT_DIR

# Remove tevatron temp
python clean_temp.py --dir=$OUTPUT_DIR