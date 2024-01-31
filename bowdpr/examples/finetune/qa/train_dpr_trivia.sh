#!/bin/bash
# @Author  :   Ma (Ma787639046@outlook.com)
BASE_DIR=$(dirname "$(dirname "$PWD")")

MODEL_NAME=$1
TRAIL_NAME=$MODEL_NAME
SEED=42
TRAIN_N_PASSAGES=16
export Q_MLEN=32
export P_MLEN=256
NPROC_PER_NODE=${GPU_NUM:-$(nvidia-smi -L | wc -l)}
TOTAL_BATCH_SIZE=$((64))
REAL_BATCH_SIZE_PER_GPU=$(($TOTAL_BATCH_SIZE/$NPROC_PER_NODE))
# REAL_BATCH_SIZE_PER_GPU=8   # debug

MODEL_PATH=$BASE_DIR/results/$MODEL_NAME/model
OUTPUT_DIR=$BASE_DIR/results/$TRAIL_NAME/trivia
LOG_DIR=$BASE_DIR/results/$TRAIL_NAME/logs
mkdir -p $LOG_DIR

# Global model kwargs
MODEL_KWARGS=" --pooling_strategy cls "
MODEL_KWARGS+=" --score_function dot "
export MODEL_KWARGS=$MODEL_KWARGS

##########################
# Common Fine-tuning Args
##########################
TRAIN_ARGS=" --do_train \
--save_strategy epoch \
--save_total_limit 1 \
--logging_steps 50 \
--remove_unused_columns False \
--bf16 \
--warmup_steps 1000 \
--per_device_train_batch_size $REAL_BATCH_SIZE_PER_GPU \
--learning_rate 5e-6 \
--min_lr_ratio 0.1 \
--lr_scheduler_type cosine \
--temperature 1.0 \
--train_n_passages $TRAIN_N_PASSAGES \
--seed $SEED \
--q_max_len $Q_MLEN \
--p_max_len $P_MLEN \
--dataloader_num_workers 8 \
--optim adamw_apex_fused \
"

# EVAL_ARGS=" --do_eval \
# --evaluation_strategy epoch \
# --bf16_full_eval \
# --per_device_eval_batch_size 128 \
# --dev_path data/trivia/trivia-dev.jsonl \
# --eval_n_passages 25 \
# --metric_for_best_model recall@20 \
# --load_best_model_at_end \
# "

set -e
set -x
##########################
# Fine-tuning Stage 1
##########################
if [ ! -f $OUTPUT_DIR/retriever_model_s1/pytorch_model.bin ]; then
  torchrun --nproc_per_node=$NPROC_PER_NODE -m \
    bowdpr.finetune.fit \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR/retriever_model_s1 \
    --corpus_path data/trivia/trivia-train.jsonl \
    --num_train_epochs 20 \
    $TRAIN_ARGS \
    $MODEL_KWARGS \
    --report_to tensorboard \
    --run_name ${TRAIL_NAME}-s1 \
    |& tee $LOG_DIR/trivia_ft_s1.log
fi

##########################
# Test Stage 1 Retriever
##########################
bash test_dpr_trivia.sh $OUTPUT_DIR/retriever_model_s1 \
  $OUTPUT_DIR s1_ $OUTPUT_DIR/encoding/corpus_s1 \
  |& tee -a $LOG_DIR/trivia_ft_s1.log

# Save encoding path: $OUTPUT_DIR/encoding/corpus_s1


############################################
# Mining Hard Negatives Using Stage 1 Retriever
############################################
if [ ! -f "$OUTPUT_DIR/encoding/corpus_s1/psg.0.pt" ]; then
  rm -rf $OUTPUT_DIR/encoding/corpus_s1

  torchrun --nproc_per_node ${NPROC_PER_NODE} \
  -m bowdpr.finetune.encode \
    --output_dir None \
    --model_name_or_path $OUTPUT_DIR/retriever_model_s1 \
    --bf16 \
    --p_max_len $P_MLEN \
    --per_device_eval_batch_size 2048 \
    --dataloader_num_workers 8 \
    --corpus_dir data/corpus.jsonl \
    --encoded_save_prefix $OUTPUT_DIR/encoding/corpus_s1/psg \
    $MODEL_KWARGS \
    |& tee $LOG_DIR/trivia_ft_mine_s1hn.log
fi

torchrun --nproc_per_node ${NPROC_PER_NODE} \
  -m bowdpr.finetune.encode \
  --output_dir None \
  --model_name_or_path $OUTPUT_DIR/retriever_model_s1 \
  --bf16 \
  --q_max_len $Q_MLEN \
  --encode_is_qry \
  --per_device_eval_batch_size 2048 \
  --dataloader_num_workers 8 \
  --corpus_path data/trivia/trivia-train.jsonl \
  --encoded_save_prefix $OUTPUT_DIR/encoding/query_s1/train \
  $MODEL_KWARGS \
  |& tee -a $LOG_DIR/trivia_ft_mine_s1hn.log

# Search
python -m bowdpr.retriever.faiss_retriever \
  --query_reps "$OUTPUT_DIR/encoding/query_s1/*.pt" \
  --passage_reps "$OUTPUT_DIR/encoding/corpus_s1/*.pt" \
  --batch_size 4000 \
  --save_ranking_to $OUTPUT_DIR/train.rank.tsv \
  --depth 100 \
  |& tee -a $LOG_DIR/trivia_ft_mine_s1hn.log

# Build HN Train file
mkdir -p $OUTPUT_DIR/train_hn
python annotate.py \
  --tsv_ranks_path $OUTPUT_DIR/train.rank.tsv \
  --query_collection data/trivia/trivia-train.jsonl \
  --passage_collection data/corpus.jsonl \
  --qrels_reference_collection data/trivia/trivia-train.jsonl \
  --output_path $OUTPUT_DIR/train_hn/mined_hn.jsonl \
  --save_text \
  --build_hn

# Copy BM25 negatives to `train_hn`
cp data/trivia/trivia-train.jsonl $OUTPUT_DIR/train_hn

##########################
# Fine-tuning Stage 2
##########################
if [ ! -f $OUTPUT_DIR/retriever_model_s2/pytorch_model.bin ]; then
  torchrun --nproc_per_node=$NPROC_PER_NODE -m \
    bowdpr.finetune.fit \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR/retriever_model_s2 \
    --corpus_path $OUTPUT_DIR/train_hn \
    --num_train_epochs 10 \
    $TRAIN_ARGS \
    $MODEL_KWARGS \
    --report_to tensorboard \
    --run_name ${TRAIL_NAME}-s2 \
    |& tee $LOG_DIR/trivia_ft_s2.log
fi

##########################
# Test Stage 1 Retriever
##########################
bash test_dpr_trivia.sh $OUTPUT_DIR/retriever_model_s2 \
  $OUTPUT_DIR s2_ \
  |& tee -a $LOG_DIR/trivia_ft_s2.log

# Remove temp
python clean_temp.py --dir=$OUTPUT_DIR
