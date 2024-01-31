# Pre-training with Bag-of-Word Prediction for Dense Passage Retrieval
Codebase for Paper [Drop your Decoder: Pre-training with Bag-of-Word Prediction for Dense Passage Retrieval](https://arxiv.org/abs/2401.11248). We develop an encoder-only pre-training schema for dense retrieval, named Bag-of-Word Prediction, for directly compressing the lexicon information into dense representations. 

## Get Started
Please install [Faiss](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) by following their guidelines. Then you can easily set up the environment by cloning this repo, and runing the following command.

```bash
pip install -e .
```

## Model Release
We have released multiple models pre-trained with Bag-of-Word Prediction. 

| Model               | Description                                     |
|---------------------|-------------------------------------------------|
| [bowdpr/bowdpr_wiki](https://huggingface.co/bowdpr/bowdpr_wiki)  | Pre-trained Model on Wikipedia and BookCorpus.  |
| [bowdpr/bowdpr_marco](https://huggingface.co/bowdpr/bowdpr_marco) | Pre-trained Model on MS-MARCO Passages.         |

The released fine-tuned retrievers are listed as follows.

| Model                       | Description                                                                |
|-----------------------------|----------------------------------------------------------------------------|
| [bowdpr/bowdpr_marco_ft](https://huggingface.co/bowdpr/bowdpr_marco_ft)      | Retriever initialized from bowdpr/bowdpr_marco and fine-tuned on MS-MARCO  |
| [bowdpr/bowdpr_wiki_nqft](https://huggingface.co/bowdpr/bowdpr_wiki_nqft)     | Retriever initialized from bowdpr/bowdpr_wiki and fine-tuned on NQ         |
| [bowdpr/bowdpr_wiki_triviaft](https://huggingface.co/bowdpr/bowdpr_wiki_triviaft) | Retriever initialized from bowdpr/bowdpr_wiki and fine-tuned on Trivia     |



## Pre-training Efficiency
Our model achieves considerable pre-training speedup comparing to previous MAE-style pre-training methods. Speed test is conducted with a batch size of 64, max sequence length of 512 and dataloader number of workers of 8.

|                   |                 | Data Process |         | Additional Decoder |             | Training Speed    |               |
|-------------------|-----------------|--------------|---------|--------------------|-------------|-------------------|---------------|
| Model             | Archeticture    | Complexity   | Time(s) | Complexity         | GPU Time(s) | Sample per second | Degeneration  |
| Pure MLM Pre-train | Encoder         | O(n)         | 0.0476  | -                  | -           | 269.708           | -             |
| Auto-Encoding     | Encoder-Decoder | O(n)         | 0.0940  | O(n^2)             | 0.0013      | 222.658           | 17.4%         |
| Auto-Regression   | Encoder-Decoder | O(n)         | 0.0636  | O(n^2)             | 0.0030      | 215.136           | 20.2%         |
| Enhanced Decoding | Encoder-Decoder | O(n^2)       | 5.6261  | O(n^2)             | 0.0012      | 85.797            | 68.2%         |
| **BoW Prediction**    | **Encoder**         | O(n)         | **0.0533**  | -                  | **0.0002**      | **266.359**           | **1.2%**          |

## Retrieval Performances
Our model achieves state-of-the-art retrieval performances on multiple retrieval benchmarks, without using any special masking, context sampling, data augmentation, or task-ensembling techniques.

|                | MS-MARCO |       |       | Natural Question |       |       | Trivia QA |       |        |
|----------------|----------|-------|-------|------------------|-------|-------|-----------|-------|--------|
| Model          | MRR@10   | R@50  | R@1k  | R@5              | R@20  | R@100 | R@5       | R@20  | R@100  |
| RetroMAE       | 39.3     | 87.0  | 98.5  | 74.4             | 84.4  | 89.4  | 78.9      | 84.5  | 88.0   |
| **BoW Prediction** | **40.1**     | **88.7**  | **98.9**  | **75.3**             | **84.6**  | **90.4**  | **79.4**      | **84.9**  | **88.0**   |

## Training
Please refer to examples below for reproducing our works.
1. [Pre-training on Wikipedia & BookCorpus or MS-MARCO Passages](examples/pretrain/README.md)
2. [Fine-tuning on MS-MARCO Passage Ranking Task](examples/finetune/msmarco/README.md)
3. [Fine-tuning on Natural Questions or TriviaQA](examples/finetune/qa/README.md)
4. [Fine-tuning on BEIR](examples/finetune/beir/README.md)


## Bugs or Questions
If you encounter any bugs or questions, please feel free to open an issue and contact me.

## Cite
If you are interested in our work, please consider citing our paper.

```
@misc{ma2024bow_pred,
      title={Drop your Decoder: Pre-training with Bag-of-Word Prediction for Dense Passage Retrieval}, 
      author={Guangyuan Ma and Xing Wu and Zijia Lin and Songlin Hu},
      year={2024},
      eprint={2401.11248},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}

```

