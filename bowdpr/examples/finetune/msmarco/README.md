# Fine-tune on MS-MARCO Passage Ranking Task

## Download MS-MARCO Collections
```bash
bash get_data.sh
```

## Fine-tune on MS-MARCO
Finetuning on MS-MARCO dataset involves a two-stage pipeline
 - s1: BM25 negs
 - s2: Mined negatives from s1

These pipelines are bootstraped in `train_dpr_msmarco.sh`. The pre-trained checkpoint on MS-MARCO Passage Corpus is released in [bowdpr/bowdpr_marco](https://huggingface.co/bowdpr/bowdpr_marco). Assume the checkpoints are already placed in `examples/results/$MODEL_NAME/model` (You can set `$MODEL_NAME` to any name as you wish), please execute the fine-tuning pipelines by just run:

```bash
bash train_dpr_msmarco.sh $MODEL_NAME
```

## Directly Test a Retreiver
We have released the fine-tuned [MS-MARCO](https://huggingface.co/bowdpr/bowdpr_marco_ft) retriever to Huggingface. Please execute the following script to test the retrieval performances.

```bash
# Save the scores of retrieval results to this folder. Change to any temporary folder as you wish
mkdir -p results/msmarco
bash test_dpr_nq.sh bowdpr/bowdpr_marco_ft results/msmarco
```
