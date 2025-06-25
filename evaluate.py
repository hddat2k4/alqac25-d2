import json
import argparse
from collections import defaultdict

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def article_key(article):
    return f"{article['law_id']}::{article['article_id']}"

def compute_metrics(retrieved_file, ground_truth_file):
    retrieved_data = load_json(retrieved_file)
    ground_truth_data = load_json(ground_truth_file)

    # Tạo dictionary để tra cứu ground truth theo question_id
    ground_truth_map = {
        item["question_id"]: set(article_key(a) for a in item["relevant_articles"])
        for item in ground_truth_data
    }

    precisions, recalls, f2s = [], [], []

    for item in retrieved_data:
        qid = item["question_id"]
        retrieved = set(article_key(a) for a in item["retrieved_articles"])
        relevant = ground_truth_map.get(qid, set())

        true_positives = retrieved & relevant

        precision = len(true_positives) / len(retrieved) if retrieved else 0.0
        recall = len(true_positives) / len(relevant) if relevant else 0.0

        if precision + recall == 0:
            f2 = 0.0
        else:
            f2 = (5 * precision * recall) / (4 * precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f2s.append(f2)

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f2 = sum(f2s) / len(f2s)

    return {
        "Average Precision": round(avg_precision, 4),
        "Average Recall": round(avg_recall, 4),
        "Average F2": round(avg_f2, 4)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate F2, Precision, Recall")
    parser.add_argument("--retrieved_file", required=True, help="Path to retrieved_articles.json")
    parser.add_argument("--ground_truth_file", required=True, help="Path to ground_truth.json")

    args = parser.parse_args()
    metrics = compute_metrics(args.retrieved_file, args.ground_truth_file)

    print("\n📊 Evaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
