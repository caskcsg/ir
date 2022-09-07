#!/bin/bash

# Model name & Output Path
MODEL_NAME=${0%.*}   # Use filename as model's output dir name
OUTPUT_DIR=results/$MODEL_NAME

if [ ! -d $OUTPUT_DIR/model ]; then
  mkdir -p $OUTPUT_DIR/model
  echo "makedir $OUTPUT_DIR/model"
fi

if [ ! -d $OUTPUT_DIR/logs ]; then
  mkdir -p $OUTPUT_DIR/logs
  echo "makedir $OUTPUT_DIR/logs"
fi

if [ ! -d $OUTPUT_DIR/tfboard/$MODEL_NAME ]; then
  mkdir -p $OUTPUT_DIR/tfboard/$MODEL_NAME
  echo "makedir $OUTPUT_DIR/tfboard/$MODEL_NAME"
fi

TOTAL_STEPS=800000
TOTAL_BATCH_SIZE=1024
TOTAL_BATCH_SIZE=$((TOTAL_BATCH_SIZE/2))  # one text generates two spans: 'anchor', 'contextual_span'
BATCH_SIZE_PER_GPU=64
GRAD_ACCU=1

python -m torch.distributed.launch \
  --nproc_per_node 8 \
  run_pretraining.py \
  --model_name_or_path bert-base-uncased \
  --output_dir $OUTPUT_DIR/model \
  --do_train \
  --logging_steps 20 \
  --save_steps 100000 \
  --save_total_limit 4 \
  --fp16 \
  --logging_dir $OUTPUT_DIR/tfboard/$MODEL_NAME \
  --warmup_ratio 0.1 \
  --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
  --gradient_accumulation_steps $GRAD_ACCU \
  --learning_rate 1e-4 \
  --max_steps $TOTAL_STEPS \
  --overwrite_output_dir \
  --dataloader_drop_last \
  --dataloader_num_workers 16 \
  --max_seq_length 128 \
  --train_path data/msmarco-docs.mlen128.json \
  --weight_decay 0.01 \
  --data_type mixed \
  --encoder_mask_ratio 0.30 \
  --decoder_mask_ratio 0.45 \
  --use_decoder_head \
  --enable_head_mlm \
  --n_head_layers 2 \
  |& tee $OUTPUT_DIR/logs/run_pretraining.log

if [ -f "$OUTPUT_DIR/model/pytorch_model.bin" ]; then
  cd msmarco
  bash eval_msmarco.sh $MODEL_NAME
fi