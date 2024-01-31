#!/bin/bash
BASE_DIR=$(dirname "$(dirname "$PWD")")


datename=$(date +%Y%m%d%H%M%S)

MODEL_PATH=$1
OUTPUT_DIR=$2
REUSE_ENCODED_CORPUS_PATH=$3

SPLITS=8
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
if [ "$REUSE_ENCODED_CORPUS_PATH" != "" ] && [ -f "$REUSE_ENCODED_CORPUS_PATH/psg.0.pt" ]; 
then
  echo "Re-Using Corpus in $REUSE_ENCODED_CORPUS_PATH"
  mkdir -p $ENCODE_TEMP_PATH/encoding/corpus/
  cp -r $REUSE_ENCODED_CORPUS_PATH/* $ENCODE_TEMP_PATH/encoding/corpus/
else
  torchrun --nproc_per_node ${NPROC_PER_NODE} \
  -m bowdpr.finetune.encode \
    --output_dir None \
    --model_name_or_path $MODEL_PATH \
    --bf16 \
    --p_max_len $P_MLEN \
    --per_device_eval_batch_size 1024 \
    --corpus_path marco/text/corpus \
    --encoded_save_prefix $ENCODE_TEMP_PATH/encoding/corpus/psg \
    $MODEL_KWARGS
fi

for PREFIX in 2019 2020; do
  # Encode query
  torchrun --nproc_per_node ${NPROC_PER_NODE} \
  -m bowdpr.finetune.encode \
    --output_dir None \
    --model_name_or_path $MODEL_PATH \
    --bf16 \
    --q_max_len $Q_MLEN \
    --encode_is_qry \
    --per_device_eval_batch_size 1024 \
    --corpus_path marco/text/trec_dl_${PREFIX}/test.query.jsonl \
    --encoded_save_prefix $ENCODE_TEMP_PATH/encoding/query_${PREFIX}/qry \
    $MODEL_KWARGS

  # Faiss retrieve
  python -m bowdpr.retriever.faiss_retriever \
    --query_reps "$ENCODE_TEMP_PATH/encoding/query_${PREFIX}/*.pt" \
    --passage_reps "$ENCODE_TEMP_PATH/encoding/corpus/*.pt" \
    --depth 1000 \
    --save_ranking_to $ENCODE_TEMP_PATH/test.rank.${PREFIX}.tsv
  
  # # Save scores
  # cp $ENCODE_TEMP_PATH/test.rank.${PREFIX}.tsv $OUTPUT_DIR/test.rank.${PREFIX}.tsv

  # Score
  python convert_result_to_trec.py \
        --input $ENCODE_TEMP_PATH/test.rank.${PREFIX}.tsv \
        --output $ENCODE_TEMP_PATH/test.rank.${PREFIX}.trec.tsv

  marco/trec_eval -l 2 -m ndcg_cut.10 -c marco/bert/trec_dl_${PREFIX}/${PREFIX}qrels-pass.txt \
    $ENCODE_TEMP_PATH/test.rank.${PREFIX}.trec.tsv \
    |& tee $OUTPUT_DIR/${PREFIX}_results.txt
done

# Remove temp
rm -rf $ENCODE_TEMP_PATH
