#!/bin/python3
# Metrics used for prediction of huggingface trainer
import numpy as np
from transformers.trainer_utils import EvalPrediction

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def mrr_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::k]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def recall_score(y_true, y_score, k=5):
    order = np.argsort(y_score)
    y_rank = order[np.arange(len(order)), y_true]       # Get [row_idx, y_true] value of each row
    r = np.mean(y_rank < k)
    return r

def ctr_score(y_true, y_score, k=1):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.mean(y_true)

def compute_metrics_warpper(metric_for_best_model: str):
    # Split metric_type for @
    if "@" in metric_for_best_model:
        metric_type = metric_for_best_model.split("@")[0]
        k = int(metric_for_best_model.split("@")[1])
    else:
        metric_type = metric_for_best_model
        k = 5      # Default to 5

    if metric_type == "dcg":
        metric_func = dcg_score
    elif metric_type == "ndcg":
        metric_func = ndcg_score
    elif metric_type == "mrr":
        metric_func = mrr_score
    elif metric_type == "ctr":
        metric_func = ctr_score
    elif metric_type == "recall":
        metric_func = recall_score
    else:
        raise NotImplementedError()
    
    def compute_metrics(prediction: EvalPrediction) -> dict:
        scores: np.array = prediction.predictions   # [num_examples, train_n_passage]
        labels: np.array = prediction.label_ids     # Should be num_examples of `0`
        metric = metric_func(y_true=labels, y_score=scores, k=k)
        return {metric_for_best_model: metric}

    return compute_metrics
