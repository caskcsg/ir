#!/bin/bash
BASE_DIR=$(dirname "$(dirname "$PWD")")

datename=$(date +%Y%m%d%H%M%S)
MODEL_PATH=$1
OUTPUT_DIR=$2
PREFIX=$3
SAVE_ENCODED_CORPUS_PATH=$4
Q_MLEN=${Q_MLEN:-32}
P_MLEN=${P_MLEN:-144}
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
  --corpus_path marco/text/corpus \
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
  --corpus_path marco/text/query/dev.query.jsonl \
  --encoded_save_prefix $ENCODE_TEMP_PATH/encoding/query/qry \
  $MODEL_KWARGS

# Faiss retrieve
python -m bowdpr.retriever.faiss_retriever \
	--query_reps "$ENCODE_TEMP_PATH/encoding/query/*.pt" \
	--passage_reps "$ENCODE_TEMP_PATH/encoding/corpus/*.pt" \
	--save_ranking_to $ENCODE_TEMP_PATH/dev.rank.tsv \
  --depth 1000

# Score
python score_to_marco.py $ENCODE_TEMP_PATH/dev.rank.tsv
python msmarco_eval.py \
  --path_to_reference=marco/qrels.dev.tsv \
  --path_to_candidate=$ENCODE_TEMP_PATH/dev.rank.tsv.marco \
  --save_folder=$OUTPUT_DIR \
  --prefix=$PREFIX

# mv $ENCODE_TEMP_PATH/dev.rank.tsv $OUTPUT_DIR/${PREFIX}dev.rank.tsv

if [ "$SAVE_ENCODED_CORPUS_PATH" != "" ]; then
  mkdir -p $SAVE_ENCODED_CORPUS_PATH
  mv ${ENCODE_TEMP_PATH}/encoding/corpus/* ${SAVE_ENCODED_CORPUS_PATH}/
fi

# Remove temp
rm -rf $ENCODE_TEMP_PATH
