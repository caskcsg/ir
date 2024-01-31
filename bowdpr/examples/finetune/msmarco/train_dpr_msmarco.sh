#!/bin/bash
# @Author  :   Ma (Ma787639046@outlook.com)
BASE_DIR=$(dirname "$(dirname "$PWD")")

MODEL_NAME=$1
TRAIL_NAME=$MODEL_NAME
SEED=42
TRAIN_N_PASSAGES=16
export Q_MLEN=32
export P_MLEN=144
NPROC_PER_NODE=${GPU_NUM:-$(nvidia-smi -L | wc -l)}
TOTAL_BATCH_SIZE=$((64))
REAL_BATCH_SIZE_PER_GPU=$(($TOTAL_BATCH_SIZE/$NPROC_PER_NODE))

MODEL_PATH=$BASE_DIR/results/$MODEL_NAME/model
OUTPUT_DIR=$BASE_DIR/results/$TRAIL_NAME/dpr
TREC_DIR_S1=$BASE_DIR/results/$TRAIL_NAME/trec_dl/s1
TREC_DIR_S2=$BASE_DIR/results/$TRAIL_NAME/trec_dl/s2
LOG_DIR=$BASE_DIR/results/$TRAIL_NAME/logs
mkdir -p $LOG_DIR

echo $MODEL_PATH
exit

# Global model kwargs
MODEL_KWARGS=" --pooling_strategy cls "
MODEL_KWARGS+=" --score_function dot "
export MODEL_KWARGS=$MODEL_KWARGS

##########################
# Common Fine-tuning Args
##########################
TRAIN_ARGS=" --do_train \
--save_strategy no \
--save_total_limit 1 \
--logging_steps 50 \
--remove_unused_columns False \
--bf16 \
--warmup_steps 1000 \
--per_device_train_batch_size $REAL_BATCH_SIZE_PER_GPU \
--learning_rate 2e-5 \
--min_lr_ratio 0.1 \
--lr_scheduler_type cosine \
--num_train_epochs 3 \
--temperature 1.0 \
--train_n_passages $TRAIN_N_PASSAGES \
--seed $SEED \
--q_max_len $Q_MLEN \
--p_max_len $P_MLEN \
--dataloader_num_workers 4 \
--optim adamw_apex_fused \
"

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
    --query_collection marco/text/query/train.query.jsonl \
    --passage_collection marco/text/corpus \
    --corpus_path marco/text/train \
    $TRAIN_ARGS \
    $MODEL_KWARGS \
    --report_to tensorboard \
    --run_name ${TRAIL_NAME}-s1 \
    |& tee $LOG_DIR/msmarco_ft_s1.log
fi

##########################
# Test Stage 1 Retriever
##########################
bash test_dpr_msmarco.sh $OUTPUT_DIR/retriever_model_s1 \
  $OUTPUT_DIR s1_ $OUTPUT_DIR/encoding/corpus_s1 \
  |& tee -a $LOG_DIR/msmarco_ft_s1.log

bash test_dpr_trec_dl.sh $OUTPUT_DIR/retriever_model_s1 $TREC_DIR_S1 $OUTPUT_DIR/encoding/corpus_s1 \
  |& tee $LOG_DIR/msmarco_ft_trec_s1.log

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
    --dataloader_num_workers 4 \
    --corpus_dir marco/text/corpus \
    --encoded_save_prefix $OUTPUT_DIR/encoding/corpus_s1/psg \
    $MODEL_KWARGS \
    |& tee $LOG_DIR/msmarco_ft_mine_s1hn.log
fi

rm -rf $OUTPUT_DIR/encoding/query_s1
torchrun --nproc_per_node ${NPROC_PER_NODE} \
  -m bowdpr.finetune.encode \
  --output_dir None \
  --model_name_or_path $OUTPUT_DIR/retriever_model_s1 \
  --bf16 \
  --q_max_len $Q_MLEN \
  --encode_is_qry \
  --per_device_eval_batch_size 2048 \
  --dataloader_num_workers 4 \
  --corpus_path marco/text/query/train.query.jsonl \
  --encoded_save_prefix $OUTPUT_DIR/encoding/query_s1/train \
  $MODEL_KWARGS \
  |& tee -a $LOG_DIR/msmarco_ft_mine_s1hn.log

# Search
python -m bowdpr.retriever.faiss_retriever \
  --query_reps "$OUTPUT_DIR/encoding/query_s1/*.pt" \
  --passage_reps "$OUTPUT_DIR/encoding/corpus_s1/*.pt" \
  --batch_size 4000 \
  --save_ranking_to $OUTPUT_DIR/train.rank.tsv \
  |& tee -a $LOG_DIR/msmarco_ft_mine_s1hn.log

# Build HN Train file
rm -rf $OUTPUT_DIR/train-hn
python -m bowdpr.utils.build_train_hn \
  --hn_file $OUTPUT_DIR/train.rank.tsv \
  --qrels marco/qrels.train.tsv \
  --queries marco/text/query/train.query.jsonl \
  --collection marco/text/corpus \
  --save_to $OUTPUT_DIR/train-hn \
  --seed $SEED \
  --prefix s1 \
  --n_sample 128 \
  --depth 2 200 \
  |& tee -a $LOG_DIR/msmarco_ft_mine_s1hn.log

# Copy BM25 negatives to `train-hn` dir
# cp marco/text/train/*.json $OUTPUT_DIR/train-hn/

############################################
# Fine-tuning Stage 2 Retriever
############################################
if [ ! -f $OUTPUT_DIR/retriever_model_s2/pytorch_model.bin ]; then
  torchrun --nproc_per_node=$NPROC_PER_NODE -m \
    bowdpr.finetune.fit \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR/retriever_model_s2 \
    --query_collection marco/text/query/train.query.jsonl \
    --passage_collection marco/text/corpus \
    --corpus_path $OUTPUT_DIR/train-hn \
    $TRAIN_ARGS \
    $MODEL_KWARGS \
    --report_to tensorboard \
    --run_name ${TRAIL_NAME}-s2 \
    |& tee $LOG_DIR/msmarco_ft_s2.log
fi

############################################
# Encode & Testing Stage 2 Retriever
############################################
bash test_dpr_msmarco.sh $OUTPUT_DIR/retriever_model_s2 \
  $OUTPUT_DIR s2_  $OUTPUT_DIR/encoding/corpus_s2 \
  |& tee -a $LOG_DIR/msmarco_ft_s2.log

# Test TREC 2019 & 2020
bash test_dpr_trec_dl.sh $OUTPUT_DIR/retriever_model_s2 $TREC_DIR_S2 $OUTPUT_DIR/encoding/corpus_s2 \
  |& tee $LOG_DIR/msmarco_ft_trec_s2.log

# Remove temp
python clean_temp.py --dir=$OUTPUT_DIR
