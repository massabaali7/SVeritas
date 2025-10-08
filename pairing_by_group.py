import json
import argparse
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import pandas as pd
import ast
import os 
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq

def compute_eer(labels, scores):
        """sklearn style compute eer
        """
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        threshold = interp1d(fpr, thresholds)(eer)
        return eer, threshold 
def compute_minDCF(labels, scores, p_target=0.01, c_miss=1, c_fa=1):
        """MinDCF
        Computes the minimum of the detection cost function.  The comments refer to
        equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
        """
        scores = np.array(scores)
        labels = np.array(labels)
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1.0 - tpr

        min_c_det = float("inf")
        min_c_det_threshold = thresholds[0]
        for i in range(0, len(fnr)):
            c_det = c_miss * fnr[i] * p_target + c_fa * fpr[i] * (1 - p_target)
            if c_det < min_c_det:
                min_c_det = c_det
                min_c_det_threshold = thresholds[i]
        c_def = min(c_miss * p_target, c_fa * (1 - p_target))
        min_dcf = min_c_det / c_def
        return min_dcf, min_c_det_threshold
def count_unique_speakers(grouped_data):
    speaker_counts = {}
    for group_value, items in grouped_data.items():
        speakers = set(item.split('/')[0] for item in items)
        speaker_counts[group_value] = len(speakers)
    return speaker_counts

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def safe_parse_metadict(meta_str):
    try:
        return ast.literal_eval(meta_str)
    except Exception as e:
        print(f"Failed to parse metadict: {e}")
        return {}
# drop prefer not to answer
def group_by_attribute(data, attributes):
    grouped = defaultdict(list)
    speaker_groups = defaultdict(set)
    for item in data:
        meta_dict = safe_parse_metadict(data[item]['metadict'])
        spk_id = item.split('/')[0]
        key = tuple(meta_dict.get(attr, "unknown") for attr in attributes)
        grouped[key].append(item)
        speaker_groups[key].add(spk_id)

    return grouped, speaker_groups

def compute_pairwise_cosine(group, model_name="wavLMBasePlus", path = "./path_to_embeddings/EARS/"):
    embeddings = []
    ids = []
    for item in group: 
        spk_id, filename = item.split("/")
        vector = np.load(os.path.join(path, spk_id, model_name, filename.split('.wav')[0] +".npy"))
        embeddings.append(vector.squeeze()) #.squeeze(0))
        ids.append(item)
    # ids = [item['id'] for item in group]
    # embeddings = [item['embedding'] for item in group]
    embeddings = np.array(embeddings)
    similarity_matrix = cosine_similarity(embeddings,embeddings)
    pairs = []
    all_scores = []
    all_labels = []
    for i, j in combinations(range(len(group)), 2):
        #similarity_matrix = cosine_similarity(embeddings[i],embeddings[j])
        spk_id_i = ids[i].split('/')[0]
        spk_id_j = ids[j].split('/')[0]
        label = int(spk_id_i == spk_id_j)  # 1 if same speaker, 0 if different
        score = float(similarity_matrix[i][j])
        pairs.append({
            "id_1": ids[i],
            "id_2": ids[j],

            "cosine_similarity": score,
            "label": label
        })
        all_scores.append(score)
        all_labels.append(label)

    eer, threshold = compute_eer(all_labels,all_scores)
    print(f"EER:{eer*100}%")
    return pairs

def main(json_file, group_attribute, model_name, output_file, path_to_embeddings ):
    data = load_json(json_file)
    group_attributes = [attr.strip() for attr in group_attribute.split(",")]

    grouped, speaker_counts = group_by_attribute(data, group_attributes)
    print("\n Unique speaker counts by", group_attribute)
    for attr_value, count in speaker_counts.items():
        print(f"{attr_value}: {len(count)} unique speakers")
    output_file= model_name+"_pairwise_results.csv"
    all_pairs = []
    for group_value, group_items in grouped.items():
        try:
            if group_value[0] != 'prefer not to answer':
                print(f"Processing group: {group_value} ({len(group_items)} items)")
                pairs = compute_pairwise_cosine(group_items,model_name=model_name,path = path_to_embeddings )
                for pair in pairs:
                    pair[group_attribute] = group_value
                all_pairs.extend(pairs)
        except RuntimeError as e:
            print("error")
            continue

    # df = pd.DataFrame(all_pairs)
    # df.to_csv(output_file, index=False)
    # print(f"Saved {len(df)} pairs to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--group_by", type=str, required=True, help="Attribute to group by (e.g., gender, language, age)")
    parser.add_argument("--output_file", type=str, default="pairwise_results.csv", help="Output CSV file path")
    parser.add_argument("--model_name", type=str, default="wavLM", help="Model Name")
    parser.add_argument("--embeddings_path", type=str, default="./embedding/dataset_name", help="Directory of your embeddings")
    args = parser.parse_args()

    main(args.json_file, args.group_by, args.model_name, args.output_file, args.embeddings_path)
