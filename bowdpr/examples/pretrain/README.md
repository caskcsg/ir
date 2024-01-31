# Pre-training with Bag-of-Word Prediction for Dense Passage Retrieval

## Data Prepare
Execute the following command below to download and truncate the wikipedia and bookcorpus corpus.

```bash
mkdir data
python prepare_pretrain_data.py --save_to data/wikibook.ml512.jsonl
```

## Pre-training with Bag-of-Word Prediction
To pre-train on wikipedia and bookcorpus, please execute the following command.

```bash
torchrun -m bowdpr.pretrain.fit  \
--do_train \
--model_name_or_path bert-base-uncased \
--output_dir ../examples/$MODEL_NAME/model \
--train_path data/wikibook.ml512.jsonl \
--dataloader_drop_last \
--dataloader_num_workers 8 \
--max_seq_length 512 \
--mlm_probability 0.30 \
--logging_steps 50 \
--max_steps 140000 \
--save_steps 20000 \
--warmup_steps 4000 \
--report_to tensorboard \
--optim adamw_apex_fused \
--bf16 \
--lr_scheduler_type cosine \
--min_lr_ratio 0.1 \
--learning_rate 3e-4 \
--per_device_train_batch_size 64 \
--gradient_accumulation_steps 4 \
--weight_decay 0.01 \
--enable_dec_bow_loss \
--bow_factor_cosine_decay_to 0.1
```

Pre-training with MS-MARCO corpus will lead to better retrieval performances on MS-MARCO passage ranking dataset. Execute the command below to start the pre-training.

```bash
# Concat all MS-MARCO Jsonl corpus
cat ../finetune/msmarco/marco/text/corpus/*.jsonl > data/msmarco_psg.jsonl

torchrun -m bowdpr.pretrain.fit  \
--do_train \
--model_name_or_path $Path_to_above_pretrained_ckpt \
--output_dir ../examples/$MODEL_NAME/model \
--train_path data/msmarco_psg.jsonl \
--dataloader_drop_last \
--dataloader_num_workers 8 \
--max_seq_length 144 \
--mlm_probability 0.30 \
--logging_steps 50 \
--max_steps 80000 \
--save_steps 20000 \
--warmup_steps 4000 \
--report_to tensorboard \
--optim adamw_apex_fused \
--bf16 \
--lr_scheduler_type cosine \
--min_lr_ratio 0.1 \
--learning_rate 3e-4 \
--per_device_train_batch_size 128 \
--gradient_accumulation_steps 2 \
--weight_decay 0.01 \
--enable_dec_bow_loss \
--bow_factor_cosine_decay_to 0.1
```

# Pre-trained Checkpoints Release
We have release the above pre-trained checkpoints to [bowdpr/bowdpr_wiki](https://huggingface.co/bowdpr/bowdpr_wiki) and [bowdpr/bowdpr_marco](https://huggingface.co/bowdpr/bowdpr_marco).
