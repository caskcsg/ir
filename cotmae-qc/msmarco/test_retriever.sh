#!/bin/bash
BASE_DIR=$(dirname "$PWD")

datename=$(date +%Y%m%d%H%M%S)
MODEL_PATH=$1
OUTPUT_DIR=$2
MODEL_TYPE=$3
PREFIX=$4
SAVE_ENCODED_CORPUS_PATH=$5
SPLITS=8
Q_MLEN=32
P_MLEN=144
ENCODE_TEMP_PATH=${OUTPUT_DIR}/ENCODE_TEMP_${datename}

set -x
############################################
# Encode & Testing Stage for Retriever
############################################
# Encode and Search
# Encode corpus
mkdir -p $ENCODE_TEMP_PATH/encoding/corpus/
parallel --ungroup --halt soon,fail=1 --xapply CUDA_VISIBLE_DEVICES={1} python -m tevatron.driver.encode \
  --output_dir None \
  --model_name_or_path $MODEL_PATH \
  --fp16 \
  --p_max_len $P_MLEN \
  --per_device_eval_batch_size 2048 \
  --encode_in_path marco/$MODEL_TYPE/corpus/split{2}.json \
  --encoded_save_path $ENCODE_TEMP_PATH/encoding/corpus/split{2}.pt \
  ::: $(seq -f "%01g" 0 $(($SPLITS-1))) \
  ::: $(seq -f "%02g" 0 $(($SPLITS-1)))

# Encode query
mkdir -p $ENCODE_TEMP_PATH/encoding/query/
CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.encode \
  --output_dir None \
  --model_name_or_path $MODEL_PATH \
  --fp16 \
  --q_max_len $Q_MLEN \
  --encode_is_qry \
  --per_device_eval_batch_size 2048 \
  --encode_in_path marco/$MODEL_TYPE/query/dev.query.json \
  --encoded_save_path $ENCODE_TEMP_PATH/encoding/query/qry.pt

# Faiss retrieve
python -m tevatron.faiss_retriever \
	--query_reps "$ENCODE_TEMP_PATH/encoding/query/qry.pt" \
	--passage_reps "$ENCODE_TEMP_PATH/encoding/corpus/*.pt" \
	--depth 1000 \
	--batch_size -1 \
	--save_ranking_to $ENCODE_TEMP_PATH/dev.rank.tsv \
  --save_text \
  --enable_multi_gpu

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

# Remove tevatron temp
rm -rf $ENCODE_TEMP_PATH
