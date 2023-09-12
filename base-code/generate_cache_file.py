from pathlib import Path
from collections import OrderedDict
import argparse

import pandas as pd
import numpy as np


methods = [
    "chinese_whispers",
    "correlation_clustering",
    "wsbm",
    "spectral_clustering",
]
idx = 1
result = f"results_{idx}"

datasets = {
    1: "dwug_data_annotated_only",
    2: "dwug_old_data_annotated_only",
    3: "dwug_new_data_annotated_only",
}


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number", type=int)
args = parser.parse_args()


p = "../outputs/experiment-results/{result}/{method}/{folder}/{folder}_{dataset}.csv"
for m in methods:
    for index_dataset in [1, 2, 3]:
        for index in range(1, args.number + 1):
            result = f"results_{index}"

            fields_results = OrderedDict()
            fields_results["adjusted_rand_score"] = []
            fields_results["parameters"] = []
            fields_results["model_hyperparameters"] = []
            fields_results["adjusted_rand_score_old"] = []
            fields_results["adjusted_rand_score_new"] = []

            fields_cache = OrderedDict()
            fields_cache["adjusted_rand_score"] = []
            fields_cache["parameters"] = []
            fields_cache["status"] = []
            fields_cache["gold_id"] = []
            fields_cache["predicted_clusters"] = []
            fields_cache["adjusted_rand_score_old"] = []
            fields_cache["adjusted_rand_score_new"] = []

            if index_dataset != 1:
                del fields_cache["adjusted_rand_score_old"]
                del fields_cache["adjusted_rand_score_new"]
                del fields_results["adjusted_rand_score_old"]
                del fields_results["adjusted_rand_score_new"]

            pd_cache = pd.DataFrame(fields_cache)
            pd_results = pd.DataFrame(fields_results)

            pd_cache.to_csv(
                p.format(
                    result=result,
                    method=m,
                    folder="cache",
                    dataset=f"{datasets[index_dataset]}",
                ),
                index=False,
                header=True,
            )
            pd_results.to_csv(
                p.format(
                    result=result,
                    method=m,
                    folder="results",
                    dataset=f"{datasets[index_dataset]}",
                ),
                index=False,
                header=True,
            )


pd_cache = pd.DataFrame(
    {
        "adjusted_rand_score": [],
        "silhouette": [],
        "calinski_harabasz": [],
        "eigengap": [],
        "adjusted_rand_score_old": [],
        "ari_silhouette_old": [],
        "ari_calinski_old": [],
        "ari_eigengap_old": [],
        "adjusted_rand_score_new": [],
        "ari_silhouette_new": [],
        "ari_calinski_new": [],
        "ari_eigengap_new": [],
        "ari_silhouette": [],
        "ari_calinski": [],
        "ari_eigengap": [],
        "number_cluster_selected_by_silhouette": [],
        "number_cluster_selected_by_calinski": [],
        "number_cluster_selected_by_eigengap": [],
        "no_clusters_results": [],
        "no_clusters_eigengap": [],
        "score_path": [],
        "parameters": [],
        "gold_id": [],
        "predicted_clusters": [],
    }
)

pd_cache_no = pd.DataFrame(
    {
        "adjusted_rand_score": [],
        "silhouette": [],
        "calinski_harabasz": [],
        "eigengap": [],
        "ari_silhouette": [],
        "ari_calinski": [],
        "ari_eigengap": [],
        "number_cluster_selected_by_silhouette": [],
        "number_cluster_selected_by_calinski": [],
        "number_cluster_selected_by_eigengap": [],
        "no_clusters_results": [],
        "no_clusters_eigengap": [],
        "score_path": [],
        "parameters": [],
        "gold_id": [],
        "predicted_clusters": [],
    }
)

pd_results = pd.DataFrame(
    {
        "adjusted_rand_score": [],
        "silhouette": [],
        "calinski_harabasz": [],
        "eigengap": [],
        "adjusted_rand_score_old": [],
        "ari_silhouette_old": [],
        "ari_calinski_old": [],
        "ari_eigengap_old": [],
        "adjusted_rand_score_new": [],
        "ari_silhouette_new": [],
        "ari_calinski_new": [],
        "ari_eigengap_new": [],
        "ari_silhouette": [],
        "ari_calinski": [],
        "ari_eigengap": [],
        "no_clusters_2": [],
        "no_clusters_3": [],
        "no_clusters_4": [],
        "no_clusters_5": [],
        "score_path": [],
        "parameters": [],
    }
)

pd_results_no = pd.DataFrame(
    {
        "adjusted_rand_score": [],
        "silhouette": [],
        "calinski_harabasz": [],
        "eigengap": [],
        "ari_silhouette": [],
        "ari_calinski": [],
        "ari_eigengap": [],
        "no_clusters_2": [],
        "no_clusters_3": [],
        "no_clusters_4": [],
        "no_clusters_5": [],
        "score_path": [],
        "parameters": [],
    }
)

p = "../outputs/experiment-results/{result}/spectral_clustering/{folder}/{folder}_{dataset}.csv"
for index_dataset in [1, 2, 3]:
    for index in range(1, args.number + 1):
        result = f"results_{index}"

        if index_dataset == 1:
            pd_cache.to_csv(
                p.format(
                    result=result,
                    folder="cache",
                    dataset=datasets[index_dataset],
                ),
                header=True,
                index=False,
            )
            pd_results.to_csv(
                p.format(
                    result=result,
                    folder="results",
                    dataset=datasets[index_dataset],
                ),
                header=True,
                index=False,
            )
        else:
            pd_cache_no.to_csv(
                p.format(
                    result=result,
                    folder="cache",
                    dataset=datasets[index_dataset],
                ),
                header=True,
                index=False,
            )
            pd_results_no.to_csv(
                p.format(
                    result=result,
                    folder="results",
                    dataset=datasets[index_dataset],
                ),
                header=True,
                index=False,
            )
