#!/bin/bash
BASE_DIR=$(dirname "$(dirname "$PWD")")


datename=$(date +%Y%m%d%H%M%S)
# datename=nq_test
MODEL_PATH=$1
OUTPUT_DIR=$2
PREFIX=$3
SAVE_ENCODED_CORPUS_PATH=$4
Q_MLEN=${Q_MLEN:-32}
P_MLEN=${P_MLEN:-256}
QUERY_PATH=data/trivia/trivia-test.jsonl
CORPUS_PATH=data/corpus.jsonl
NPROC_PER_NODE=${GPU_NUM:-$(nvidia-smi -L | wc -l)}
ENCODE_TEMP_PATH=${OUTPUT_DIR}/ENCODE_TEMP_${datename}

set -e
set -x
############################################
# Encode & Testing Stage for Retriever
############################################
# Encode and Search
# Encode corpus
torchrun --nproc_per_node ${NPROC_PER_NODE} \
  -m bowdpr.finetune.encode \
  --output_dir None \
  --model_name_or_path $MODEL_PATH \
  --bf16 \
  --p_max_len $P_MLEN \
  --per_device_eval_batch_size 1024 \
  --dataloader_num_workers 4 \
  --corpus_path $CORPUS_PATH \
  --encoded_save_prefix $ENCODE_TEMP_PATH/encoding/corpus/psg \
  $MODEL_KWARGS

# Encode query
torchrun --nproc_per_node ${NPROC_PER_NODE} \
  -m bowdpr.finetune.encode \
  --output_dir None \
  --model_name_or_path $MODEL_PATH \
  --bf16 \
  --q_max_len $Q_MLEN \
  --encode_is_qry \
  --per_device_eval_batch_size 1024 \
  --corpus_path $QUERY_PATH \
  --encoded_save_prefix $ENCODE_TEMP_PATH/encoding/query/qry \
  $MODEL_KWARGS

# Faiss retrieve
python -m bowdpr.retriever.faiss_retriever \
	--query_reps "$ENCODE_TEMP_PATH/encoding/query/*.pt" \
	--passage_reps "$ENCODE_TEMP_PATH/encoding/corpus/*.pt" \
	--save_ranking_to $ENCODE_TEMP_PATH/test.rank.tsv \
  --depth 100

# Annotate & score
python annotate.py \
  --tsv_ranks_path $ENCODE_TEMP_PATH/test.rank.tsv \
  --query_collection $QUERY_PATH \
  --passage_collection $CORPUS_PATH \
  --output_path $ENCODE_TEMP_PATH/test.jsonl

python evaluate_dpr_retrieval.py \
  --retrieval $ENCODE_TEMP_PATH/test.jsonl \
  --save $OUTPUT_DIR/${PREFIX}result.json

# mv $ENCODE_TEMP_PATH/test.jsonl $OUTPUT_DIR/${PREFIX}test.jsonl

if [ "$SAVE_ENCODED_CORPUS_PATH" != "" ]; then
  mkdir -p $SAVE_ENCODED_CORPUS_PATH
  mv ${ENCODE_TEMP_PATH}/encoding/corpus/* ${SAVE_ENCODED_CORPUS_PATH}/
fi

# Remove temp
rm -rf $ENCODE_TEMP_PATH
