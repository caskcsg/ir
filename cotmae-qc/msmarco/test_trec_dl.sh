#!/bin/bash
BASE_DIR=$(dirname "$PWD")

datename=$(date +%Y%m%d%H%M%S)

MODEL_PATH=$1
OUTPUT_DIR=$2
REUSE_ENCODED_CORPUS_PATH=$3

SPLITS=8
Q_MLEN=32
P_MLEN=144
ENCODE_TEMP_PATH=${OUTPUT_DIR}/ENCODE_TEMP_${datename}
if [ ! -d $MODEL_PATH ]; then
  echo "${MODEL_PATH} does not exists, please run tevatron first"
  exit
fi
mkdir -p $OUTPUT_DIR

set -x
############################################
# Encode & Testing Stage for Retriever
############################################
# Encode and Search
# Encode corpus
mkdir -p $ENCODE_TEMP_PATH/encoding/corpus/
if [ "$REUSE_ENCODED_CORPUS_PATH" != "" ] && [ -f "$REUSE_ENCODED_CORPUS_PATH/split00.pt" ]; 
then
  echo "Re-Using Corpus in $REUSE_ENCODED_CORPUS_PATH"
  cp -r $REUSE_ENCODED_CORPUS_PATH/* $ENCODE_TEMP_PATH/encoding/corpus/
else
  parallel --ungroup --halt soon,fail=1 --xapply CUDA_VISIBLE_DEVICES={1} python -m tevatron.driver.encode \
    --output_dir None \
    --model_name_or_path $MODEL_PATH \
    --fp16 \
    --p_max_len $P_MLEN \
    --per_device_eval_batch_size 2048 \
    --encode_in_path marco/bert/corpus/split{2}.json \
    --encoded_save_path $ENCODE_TEMP_PATH/encoding/corpus/split{2}.pt \
    ::: $(seq -f "%01g" 0 $(($SPLITS-1))) \
    ::: $(seq -f "%02g" 0 $(($SPLITS-1)))
fi

for PREFIX in 2019 2020; do
  # Encode query
  mkdir -p $ENCODE_TEMP_PATH/encoding/query/
  CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.encode \
    --output_dir None \
    --model_name_or_path $MODEL_PATH \
    --fp16 \
    --q_max_len $Q_MLEN \
    --encode_is_qry \
    --per_device_eval_batch_size 2048 \
    --encode_in_path marco/bert/trec_dl_${PREFIX}/test.query.json \
    --encoded_save_path $ENCODE_TEMP_PATH/encoding/query/qry_${PREFIX}.pt

  # Faiss retrieve
  mkdir -p $ENCODE_TEMP_PATH/test_ranks_${PREFIX}
  python -m tevatron.faiss_retriever \
    --query_reps "$ENCODE_TEMP_PATH/encoding/query/qry_${PREFIX}.pt" \
    --passage_reps "$ENCODE_TEMP_PATH/encoding/corpus/*.pt" \
    --depth 1000 \
    --batch_size -1 \
    --save_ranking_to $ENCODE_TEMP_PATH/test.rank.${PREFIX}.tsv \
    --save_text \
    --enable_multi_gpu

  # Score
  python -m tevatron.utils.format.convert_result_to_trec \
        --input $ENCODE_TEMP_PATH/test.rank.${PREFIX}.tsv \
        --output $ENCODE_TEMP_PATH/test.rank.${PREFIX}.trec.tsv

  marco/trec_eval -l 2 -m ndcg_cut.10 -c marco/bert/trec_dl_${PREFIX}/${PREFIX}qrels-pass.txt \
    $ENCODE_TEMP_PATH/test.rank.${PREFIX}.trec.tsv \
    |& tee $OUTPUT_DIR/${PREFIX}_results.txt
done

# Remove tevatron temp
rm -rf $ENCODE_TEMP_PATH
