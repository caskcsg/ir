#!/bin/bash
BASE_DIR=$(dirname "$PWD")

datename=$(date +%Y%m%d%H%M%S)
MODEL_PATH=$1
OUTPUT_DIR=$2
PREFIX=$3
SPLITS=8
ENCODE_TEMP_PATH=${OUTPUT_DIR}/ENCODE_TEMP_${datename}

set -x
############################################
# Encode & Testing Stage for Retriever
############################################
# Encode and Search
# Encode corpus
mkdir -p $ENCODE_TEMP_PATH/encoding/corpus-s4/
parallel --ungroup --halt soon,fail=1 --xapply CUDA_VISIBLE_DEVICES={1} python -m tevatron.driver.encode \
  --output_dir None \
  --model_name_or_path $MODEL_PATH \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --encode_in_path marco/bert/corpus/split{2}.json \
  --encoded_save_path $ENCODE_TEMP_PATH/encoding/corpus-s4/split{2}.pt \
  ::: $(seq -f "%01g" 0 $(($SPLITS-1))) \
  ::: $(seq -f "%02g" 0 $(($SPLITS-1)))

# Encode query
mkdir -p $ENCODE_TEMP_PATH/encoding/query-s4/
CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.encode \
  --output_dir None \
  --model_name_or_path $MODEL_PATH \
  --fp16 \
  --q_max_len 32 \
  --encode_is_qry \
  --per_device_eval_batch_size 128 \
  --encode_in_path marco/bert/query/dev.query.json \
  --encoded_save_path $ENCODE_TEMP_PATH/encoding/query-s4/qry.pt

# Faiss retrieve
mkdir -p $ENCODE_TEMP_PATH/dev_ranks
parallel --ungroup --halt soon,fail=1 --xapply CUDA_VISIBLE_DEVICES={1} python -m tevatron.faiss_retriever \
	--query_reps $ENCODE_TEMP_PATH/encoding/query-s4/qry.pt \
	--passage_reps $ENCODE_TEMP_PATH/encoding/corpus-s4/split{2}.pt \
	--depth 1000 \
	--batch_size -1 \
	--save_ranking_to $ENCODE_TEMP_PATH/dev_ranks/{2} \
  ::: $(seq -f "%01g" 0 $(($SPLITS-1))) \
  ::: $(seq -f "%02g" 0 $(($SPLITS-1)))

# Reduce
python -m tevatron.faiss_retriever.reducer \
  --score_dir $ENCODE_TEMP_PATH/dev_ranks \
  --query $ENCODE_TEMP_PATH/encoding/query-s4/qry.pt \
  --save_ranking_to $ENCODE_TEMP_PATH/dev.rank.tsv

# Score
python score_to_marco.py $ENCODE_TEMP_PATH/dev.rank.tsv
python msmarco_eval.py \
  --path_to_reference=marco/qrels.dev.tsv \
  --path_to_candidate=$ENCODE_TEMP_PATH/dev.rank.tsv.marco \
  --save_folder=$OUTPUT_DIR \
  --prefix=$PREFIX

mv $ENCODE_TEMP_PATH/dev.rank.tsv $OUTPUT_DIR/${PREFIX}dev.rank.tsv

# Remove tevatron temp
rm -rf $ENCODE_TEMP_PATH
