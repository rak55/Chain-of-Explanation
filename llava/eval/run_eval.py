import argparse
import json
import collections
from nltk.translate.bleu_score import sentence_bleu
from eval_metrics.evaluate_metrics import (
    calculate_exactmatch,
    calculate_f1score,
)
from tabulate import tabulate
from eval_metrics.glossary import normalize_word

import warnings
from datasets import load_dataset

warnings.simplefilter("ignore")


def parse_option():
    parser = argparse.ArgumentParser(
        "Evaluation for LLaVA Generated Outputs", add_help=False
    )
    parser.add_argument("--dataset", type=str, default="flaviagiammarino/vqa-rad")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        help="path to prediction file",
    )
    args = parser.parse_args()
    return args


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            data.append(json.loads(line))
    return data


def evaluate(ds, pred):
    closed_scores = collections.defaultdict(list)
    exact_scores = collections.defaultdict(list)
    f1_scores = collections.defaultdict(list)
    open_hit_scores = collections.defaultdict(list)
    closed_answers = {"yes", "no"}

    for idx in range(len(ds)):
        ex = ds[idx]

        gt_value = ex["answer"].lower()
        pred_value = pred[idx]["pred"].lower()

        gt_value = normalize_word(gt_value)
        pred_value = normalize_word(pred_value)

        if gt_value in closed_answers:
            # for close-ended question (Yes/No)
            # closed_scores['q_id'].append(pred_item['question_id'])
            if "yes" in pred_value or "no" in pred_value:
                if gt_value in pred_value:
                    closed_scores["hit"].append(1)
                else:
                    closed_scores["hit"].append(0)
            else:
                closed_scores["hit"].append(0)
        else:
            # for open-ended question
            if gt_value in pred_value:
                hit = 1.0
            else:
                hit = 0.0
            open_hit_scores["hit"].append(hit)

            # open_hit_scores['hit'].append(calculate_appearance_with_normalization(pred_value, gt_value, candidate))
            # open_hit_scores['q_id'].append(pred_item['question_id'])

            exact_scores["hit"].append(calculate_exactmatch(pred_value, gt_value))
            # exact_scores['q_id'].append(pred_item['question_id'])

            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            f1_scores["f1"].append(f1_score)
            f1_scores["precision"].append(precision)
            f1_scores["recall"].append(recall)
            # f1_scores['q_id'].append(pred_item['question_id'])

            # if isinstance(f1_scores['hit'][-1], str):
            #     # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    # f1_score = sum(f1_scores["f1"]) / len(f1_scores["f1"])
    # precision = sum(f1_scores["precision"]) / len(f1_scores["precision"])
    recall = sum(f1_scores["recall"]) / len(f1_scores["recall"])

    # open_hit_score = sum(open_hit_scores["hit"]) / len(open_hit_scores["hit"])
    closed_score = (
        sum(closed_scores["hit"]) / len(closed_scores["hit"])
        if len(closed_scores["hit"]) != 0
        else 0.0
    )

    num_open, num_close = len(closed_scores["hit"]), len(open_hit_scores["hit"])
    print(f"num_open {num_open} || num_close {num_close}")

    return tabulate(
        [
            ["Closed Accuracy", closed_score * 100],
            # ["open accuracy", open_hit_score * 100],
            # ["f1 score", f1_score * 100],
            # ["precision", precision * 100],
            ["Open Recall", recall * 100],
        ],
        headers=["Metric", "Performance"],
    )


if __name__ == "__main__":
    args = parse_option()

    print(f"Loading dataset: {args.dataset} ({args.split})")
    dataset = load_dataset(args.dataset)
    data_split = dataset[args.split]

    pred = load_jsonl(args.pred)

    print(f"Dataset Size: {len(data_split)} || Pred Size: {len(pred)}")
    assert len(data_split) == len(
        pred
    ), "please make sure preds and dataset are exactly matched"

    pred = {p["id"]: p for p in pred}

    # perform evaluation
    results = evaluate(data_split, pred)
    print(results)
