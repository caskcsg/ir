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
Q_MLEN=32
P_MLEN=144
MODEL_PATH=$BASE_DIR/results/$MODEL_NAME/model
OUTPUT_DIR=$BASE_DIR/results/$MODEL_NAME/tevatron
TREC_DIR_S1=$BASE_DIR/results/$MODEL_NAME/trec_dl/s1
TREC_DIR_S2=$BASE_DIR/results/$MODEL_NAME/trec_dl/s2
LOG_DIR=$BASE_DIR/results/$MODEL_NAME/logs
mkdir -p $LOG_DIR

set -x
##########################
# Fine-tuning Stage 1
##########################
torchrun --nproc_per_node=$SPLITS -m \
  tevatron.driver.train \
  --model_name_or_path $MODEL_PATH \
  --output_dir $OUTPUT_DIR/retriever_model_s1 \
  --save_strategy no \
  --train_dir marco/$MODEL_TYPE/train \
  --fp16 \
  --warmup_ratio 0.1 \
  --per_device_train_batch_size 8 \
  --learning_rate $LR \
  --num_train_epochs 3 \
  --train_n_passages $TRAIN_N_PASSAGES \
  --negatives_x_device \
  --seed $SEED \
  --q_max_len $Q_MLEN \
  --p_max_len $P_MLEN \
  --dataloader_num_workers 2 \
  --report_to tensorboard \
  --optim adamw_torch_fused \
  |& tee $LOG_DIR/msmarco_ft_s1.log

##########################
# Test Stage 1 Retriever
##########################
bash test_retriever.sh $OUTPUT_DIR/retriever_model_s1 \
  $OUTPUT_DIR $MODEL_TYPE s1_ $OUTPUT_DIR/encoding/corpus_s1 \
  |& tee -a $LOG_DIR/msmarco_ft_s1.log

bash test_trec_dl.sh $OUTPUT_DIR/retriever_model_s1 $TREC_DIR_S1 $OUTPUT_DIR/encoding/corpus_s1 \
  |& tee $LOG_DIR/msmarco_ft_trec_s1.log

############################################
# Mining Hard Negatives Using Stage 1 Retriever
############################################
if [ ! -f "$OUTPUT_DIR/encoding/corpus_s1/split00.pt" ]; then
  rm -rf $OUTPUT_DIR/encoding/corpus_s1
  mkdir -p $OUTPUT_DIR/encoding/corpus_s1

  parallel --ungroup --halt soon,fail=1 --xapply CUDA_VISIBLE_DEVICES={1} python -m tevatron.driver.encode \
    --output_dir None \
    --model_name_or_path $OUTPUT_DIR/retriever_model_s1 \
    --fp16 \
    --p_max_len $P_MLEN \
    --per_device_eval_batch_size 2048 \
    --encode_in_path marco/$MODEL_TYPE/corpus/split{2}.json \
    --encoded_save_path $OUTPUT_DIR/encoding/corpus_s1/split{2}.pt \
    ::: $(seq -f "%01g" 0 $(($SPLITS-1))) \
    ::: $(seq -f "%02g" 0 $(($SPLITS-1))) \
    |& tee $LOG_DIR/msmarco_ft_mine_s1hn.log
fi

rm -rf $OUTPUT_DIR/encoding/query_s1
mkdir -p $OUTPUT_DIR/encoding/query_s1
CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.encode \
  --output_dir None \
  --model_name_or_path $OUTPUT_DIR/retriever_model_s1 \
  --fp16 \
  --q_max_len $Q_MLEN \
  --encode_is_qry \
  --per_device_eval_batch_size 2048 \
  --encode_in_path marco/$MODEL_TYPE/query/train.query.json \
  --encoded_save_path $OUTPUT_DIR/encoding/query_s1/train.pt \
  |& tee -a $LOG_DIR/msmarco_ft_mine_s1hn.log

# Search
rm -rf $OUTPUT_DIR/train_ranks
mkdir -p $OUTPUT_DIR/train_ranks

python -m tevatron.faiss_retriever \
  --query_reps "$OUTPUT_DIR/encoding/query_s1/train.pt" \
  --passage_reps "$OUTPUT_DIR/encoding/corpus_s1/*.pt" \
  --batch_size 4000 \
  --save_ranking_to $OUTPUT_DIR/train.rank.tsv \
  --save_text \
  --enable_multi_gpu \
  |& tee -a $LOG_DIR/msmarco_ft_mine_s1hn.log

# Build HN Train file
rm -rf $OUTPUT_DIR/train-hn
python build_train_hn.py \
  --tokenizer_name $OUTPUT_DIR/retriever_model_s1 \
  --hn_file $OUTPUT_DIR/train.rank.tsv \
  --qrels marco/qrels.train.tsv \
  --queries marco/train.query.txt \
  --collection marco/corpus.tsv \
  --save_to $OUTPUT_DIR/train-hn \
  --seed $SEED \
  --truncate $P_MLEN \
  --prefix s1 \
  --n_sample 128 \
  |& tee -a $LOG_DIR/msmarco_ft_mine_s1hn.log

# Copy BM25 negatives to `train-hn` dir (We do not use BM25 negs in query-as-context)
# cp marco/$MODEL_TYPE/train/*.json $OUTPUT_DIR/train-hn/

############################################
# Fine-tuning Stage 2 Retriever
############################################
torchrun --nproc_per_node=$SPLITS -m tevatron.driver.train \
  --model_name_or_path $MODEL_PATH \
  --output_dir $OUTPUT_DIR/retriever_model_s2 \
  --save_strategy no \
  --train_dir $OUTPUT_DIR/train-hn \
  --fp16 \
  --warmup_ratio 0.1 \
  --per_device_train_batch_size 8 \
  --learning_rate $LR \
  --num_train_epochs 3 \
  --train_n_passages $TRAIN_N_PASSAGES \
  --negatives_x_device \
  --seed $SEED \
  --q_max_len $Q_MLEN \
  --p_max_len $P_MLEN \
  --dataloader_num_workers 2 \
  --report_to tensorboard \
  --optim adamw_torch_fused \
  |& tee $LOG_DIR/msmarco_ft_s2.log

############################################
# Encode & Testing Stage 2 Retriever
############################################
bash test_retriever.sh $OUTPUT_DIR/retriever_model_s2 \
  $OUTPUT_DIR $MODEL_TYPE s2_  $OUTPUT_DIR/encoding/corpus_s2 \
  |& tee -a $LOG_DIR/msmarco_ft_s2.log

# Test TREC 2019 & 2020
bash test_trec_dl.sh $OUTPUT_DIR/retriever_model_s2 $TREC_DIR_S2 $OUTPUT_DIR/encoding/corpus_s2 \
  |& tee $LOG_DIR/msmarco_ft_trec_s2.log

# Remove tevatron temp
python clean_temp.py --dir=$OUTPUT_DIR