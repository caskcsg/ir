# Query-as-context Pre-training for Dense Passage Retrieval
Codebase for **[EMNLP 2023 Main Conference Paper]** [Query-as-context Pre-training for Dense Passage Retrieval](https://arxiv.org/abs/2212.09598)

## Dependencies
Please refer to [PyTorch Homepage](https://pytorch.org/) to install a pytorch version suitable for your system.

Dependencies can be installed by running codes below.
```bash
apt-get install parallel
pip install -r requirments.txt
```

We use [Tevatron](https://github.com/texttron/tevatron) toolkit for finetuning. You can install it by following its guidelines.


## Data processing
The core idea of our work is utilizing queries as better contextual related spans for context-based pre-training of dense retrievers. There are 3 steps for data processing and query generation.

### Step 1: Download & Generate Text Splits
You can download MS-MARCO Document Corpus from [here](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz).

If finish downloading, please unzip and extract textual corpus using the following commands:

```bash
gunzip msmarco-docs.tsv.gz
awk '{for(i=3;i<NF;i++)printf("%s ",$i);print $NF}' msmarco-docs.tsv > msmarco-docs.txt
```

We utilize data process scripts from CoT-MAE for cutting pre-training corpus to proper maxlen. Please use the commands below for generating text splits:

```bash
cd qg
python make_contextual_splits.py \
    --file=$PATH_TO/msmarco-docs.txt \
    --save_to=$PATH_TO/msmarco_text.json \
    --maxlen=144
```


### Step 2: Generate Queries with multi-GPU
DocT5query is used for query generation. Please refer to following commands.

```bash
python query_gen.py \
    --model_name_or_path=doc2query/all-with_prefix-t5-base-v1 \
    --dataset=$PATH_TO/msmarco_text.json \
    --output_dir=$SAVE_FOLDER \
    --saved_filename=msmarco_text.q20.json \
    --ques_per_passage=20 \
    --batch_size=64
```

It is highly recommended to shard the `msmarco_text.json` to multiple splits, then run query generation on multiple nodes for parallel processing.

### Step 3: Tokenize Json Text
Tokenize all truncated corpus and generated queries with the following commands.

```bash
python tokenize_json_file.py \
    --file=$PATH_TO/msmarco_text.q20.json \
    --save_to=$PATH_TO/msmarco_qg.mlen144.s15.json \
    --save_queries_as_list
```

`msmarco_qg.mlen144.s15.json` is the final preprocessed pre-training corpus, which is used in following pre-training archetictures.


## coCondenser + Query-as-context
We adapt the contrastive pre-training archeticture from coCondenser.

`GradCache` is used for enlarging effective contrastive batch size. Please install the dependency of `GradCache` by following commands:

```bash
pip install -e cocondenser/GradCache
```

Then you can use the following commands to pre-train coCondenser with query contexts.

```bash
torchrun cocondenser/run_co_pre_training.py \
    --model_name_or_path Luyu/condenser \
    --train_path $PATH_TO/msmarco_qg.mlen144.s15.json \
    --output_dir $OUTPUT_DIR/model \
    --overwrite_output_dir \
    --dataloader_drop_last \
    --dataloader_num_workers 16 \
    --do_train \
    --logging_steps 200 \
    --save_steps 10000 \
    --save_total_limit 10 \
    --fp16 \
    --logging_dir $TENSORBOARD_DIR \
    --report_to tensorboard \
    --optim adamw_torch_fused \
    --warmup_ratio 0.1 \
    --model_type bert \
    --data_type query \
    --learning_rate 1e-4 \
    --max_steps 120000 \
    --per_device_train_batch_size 256 \
    --max_seq_length 144 \
    --weight_decay 0.01 \
    --n_head_layers 2 \
    --skip_from 6 \
    --late_mlm \
    --cache_chunk_size 64
```

Training with large batch size is essential for effective contrastive pre-training. The global batch size used in out setting is 2048.


## CoT-MAE + Query-as-context
We adapt the bottlenecked pre-training archeticture from CoT-MAE. Please use the following commands to pre-train CoT-MAE with query contexts.

```bash
torchrun cotmae/run_pretraining.py \
    --model_name_or_path bert-base-uncased \
    --train_path $PATH_TO/msmarco_qg.mlen144.s15.json \
    --output_dir $OUTPUT_DIR/model \
    --overwrite_output_dir \
    --dataloader_drop_last \
    --dataloader_num_workers 16 \
    --do_train \
    --logging_steps 200 \
    --save_steps 10000 \
    --save_total_limit 10 \
    --fp16 \
    --logging_dir $TENSORBOARD_DIR \
    --report_to tensorboard \
    --optim adamw_torch_fused \
    --warmup_ratio 0.1 \
    --data_type query \
    --learning_rate 4e-4 \
    --max_steps 50000 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 32 \
    --max_seq_length 144 \
    --weight_decay 0.01 \
    --bert_mask_ratio 0.30 \
    --use_dec_head \
    --n_dec_head_layers 1 \
    --enable_dec_head_loss \
    --dec_head_coef 1.0 \
    --attn_window 3
```

The total batch size `# of GPUs * BATCH_SIZE_PER_GPU * GRAD_ACC` is 16k. You can speed up pre-training with multi-nodes environments.

## Finetuning
### Finetuning on MS-Marco Passage ranking task with Tevatron
[Tevatron](https://github.com/texttron/tevatron) toolkit for finetuning. Please refer to [msmarco/eval_msmarco.sh](./msmarco/eval_msmarco.sh) for more details. You can run a two stage finetuning by just running this scripts.
```bash
cd msmarco
# Get and process data when running Tevatron for the first time
bash get_data.sh
# Assume the pre-trained model is located in ./results/$MODEL_NAME/model
bash eval_msmarco.sh $MODEL_NAME
```

Pre-training with query-as-context boost the performance of both contrastive and bottlenecked baselines. Scores on MS-MARCO and TREC-DL is listed as follows.

#### Scores with contrastive method:

|                                           | MS-MARCO |      |      | TREC-DL 19 | TREC-DL 20  |
|-------------------------------------------|----------|------|------|------------|-------------|
|                                           | MRR@10   | R@50 | R@1k | NDCG@10    | NDCG@10     |
| coCondenser (paper)                       | 38.2     | 86.5 | 98.4 | 71.7       | 68.4        |
| coCondenser (120K) - retriever 1          | 37       | 86   | 98.5 | 68.2       | 68.8        |
| **w/ query-as-context (120K) - retriever 1** | **37.4**     | **87.3** | **98.6** | 68.1       | **69.2**        |
| coCondenser (120K) - retriever 2          | 38.8     | 87.8 | 98.8 | 71.1       | 68.4        |
| **w/ query-as-context (120K) - retriever 2** | **39.4**     | **88.6** | **99.0**   | **73.1**       | **71.8**        |

#### Scores with bottlenecked method:

|                                         | MS-MARCO |      |      | TREC-DL 19 | TREC-DL 20  |
|-----------------------------------------|----------|------|------|------------|-------------|
|                                         | MRR@10   | R@50 | R@1k | NDCG@10    | NDCG@10     |
| CoT-MAE  (paper)                        | 39.4     | 87   | 98.7 | 70.9       | 70.4        |
| CoT-MAE (50K) - retriever 1             | 37.2     | 85.7 | 98.2 | 65.7       | 66.5        |
| **w/ query-as-context (50K) - retriever 1** | **38.6**     | **87.7** | **98.6** | **67.7**       | **67.8**        |
| CoT-MAE (50K) - retriever 2             | 38.8     | 87.3 | 98.6 | 70.7       | 69.7        |
| **w/ query-as-context (50K) - retriever 2** | **40.2**     | **88.8** | **98.8** | **71.5**       | **72.7**        |

### Test on BEIR
Please refer to [BEIR repo](https://github.com/beir-cellar/beir) for fine-tuning and testing on BEIR benchmarks. Specifically, you can find the example training script at [here](https://github.com/beir-cellar/beir/blob/main/examples/retrieval/training/train_msmarco_v3.py). 

## Bugs or Questions
If you have any questions, please email to Guangyuan Ma (Ma787639046@outlook.com) and Xing Wu (wuxing@iie.ac.cn).

## Citation
If you find our work useful, please consider to cite our paper.

```
@misc{wu2023queryascontext,
      title={Query-as-context Pre-training for Dense Passage Retrieval}, 
      author={Xing Wu and Guangyuan Ma and Wanhui Qian and Zijia Lin and Songlin Hu},
      year={2023},
      eprint={2212.09598},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

