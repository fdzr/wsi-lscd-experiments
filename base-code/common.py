import ast
from glob import glob
import json
import logging
import random
import re
import warnings
from pprint import pprint
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.csgraph import laplacian
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

from utils_ import _save_graph, _save_results, load_senses_and_scores
from clustering import _adjacency_matrix_to_nxgraph


warnings.filterwarnings("ignore")

seeds = {
    "chinese_whispers": {1: 90, 2: 95, 3: 100, 4: 105, 5: 110},
    "spectral_clustering": {1: 2100, 2: 1600, 3: 100, 4: 600, 5: 1100},
    "correlation_clustering": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
    "wsbm": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
}


def get_adj_matrix(
    data: pd.DataFrame,
    id_to_int: dict,
    n_sentences: int,
    scaler: object,
    fits_scaler: bool = False,
    threshold: int = 0.5,
    binarize: bool = False,
    meta_information: dict = {},
) -> np.ndarray:
    """
    Returns a adjacency matric of size n_sentences * n_sentences

    Parameters:
    data: a pandas dataframe containing the sentence pairs and their scores
    id_to_int: a dictionary mapping a sentence identifier to a distinct int less than n_sentences
    threshold: no edge will be added between sentences having a score less than threshold
    binarize: if set to True, all edge weights will be converted to 1, else the edge weights will be
    scaled using min-max scaling
    """

    information = {}
    information.update(meta_information)
    information["binarize"] = binarize
    information["thresholds"] = []
    information["thresholds"].append(threshold)

    adj_matrix = np.zeros((n_sentences, n_sentences), dtype="float")
    adj_matrix_no_thresholds = np.zeros(
        (n_sentences, n_sentences), dtype="float"
    )
    if not binarize:
        if scaler is not None:
            if fits_scaler:
                data["score"] = scaler.fit_transform(
                    data["score"].to_numpy().reshape(-1, 1)
                )
            else:
                data["score"] = scaler.transform(
                    data["score"].to_numpy().reshape(-1, 1)
                )

            threshold = scaler.transform([[threshold]]).item()

    information["thresholds"].append(threshold)

    for i, row in data.iterrows():
        x = id_to_int[row["sentence1"] + f"{row['start1']}-{row['end1']}"]
        y = id_to_int[row["sentence2"] + f"{row['start2']}-{row['end2']}"]
        adj_matrix_no_thresholds[x, y] = adj_matrix_no_thresholds[y, x] = (
            1 if binarize else row["score"]
        )

        if row["score"] >= threshold:
            # binarization creates discrete graphs
            adj_matrix[x, y] = adj_matrix[y, x] = (
                1 if binarize else row["score"]
            )

    information.pop("logging", None)
    if "portion_dataset" in information:
        information.pop("portion_dataset", None)

    if "new_data" in information and "old_data" in information:
        information.pop("new_data", None)
        information.pop("old_data", None)

    _save_graph(**information, graphs=[adj_matrix, adj_matrix_no_thresholds])
    return adj_matrix


def eigengapHeuristic(adj_matrix, max_n_clusters):
    L = laplacian(adj_matrix, normed=True)
    n_components = adj_matrix.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1]

    return [x + 1 for x in index_largest_gap if x <= max_n_clusters - 1]


def get_wsi_data(
    sentences_path: str,
    senses_path: str,
    pairs_path: str,
    pair_regex: str,
    score_paths: dict,
    debug=False,
):
    sentences = pd.read_csv(f"summer-wsi/{sentences_path}", sep="\t")
    context_to_contextId = {
        x["context"]: x["context_id"] for _, x in sentences.iterrows()
    }
    senses = pd.read_csv(f"summer-wsi/{senses_path}", sep="\t")
    words = set(senses["word"])
    if debug:
        print("Words:", words)
    scores = {x: pd.DataFrame() for x in score_paths}

    for path in glob(pairs_path):
        word = re.search(pair_regex, path).group(1)
        formatted_word = word.replace("u#U0308", "ü").replace("#U00df", "ß")
        if formatted_word not in words:
            if debug:
                print(f"Word '{formatted_word}' not found!")
            continue

        with open(path, "r", encoding="utf8") as f_pairs:
            pairs = pd.DataFrame(json.load(f_pairs))
            for score_model in score_paths:
                with open(
                    glob(score_paths[score_model].format(word))[0],
                    "r",
                    encoding="utf8",
                ) as f_scores:
                    word_scores = pd.merge(
                        pairs,
                        pd.DataFrame(json.load(f_scores)),
                        how="inner",
                        on="id",
                    )

                scores[score_model] = scores[score_model].append(word_scores)

    for score_model in score_paths:
        scores[score_model]["score"] = scores[score_model]["score"].apply(
            lambda x: sum(map(float, x)) / len(x)
        )
        scores[score_model].rename(columns={"lemma": "word"}, inplace=True)
        if debug:
            print("Model:", score_model)
            print("No of pairs:", len(scores[score_model]))
        scores[score_model]["sentence1_id"] = scores[score_model][
            "sentence1"
        ].apply(lambda x: context_to_contextId.get(x))
        scores[score_model]["sentence2_id"] = scores[score_model][
            "sentence2"
        ].apply(lambda x: context_to_contextId.get(x))
        scores[score_model].dropna(
            subset=["sentence1_id", "sentence2_id"], inplace=True
        )
        if debug:
            print("No of pairs after filtering:", len(scores[score_model]))

    return senses, scores


def get_dwug_data(score_paths: list, annotated_only=False, debug=False):
    sentences_path = "datasets_unlabeled/se20lscd_v2/de/unlabeled-old+new.tsv"
    senses_path = "datasets/se20lscd_v2/de/sense-old+new.tsv"
    return get_wsi_data(
        senses_path if annotated_only else sentences_path,
        senses_path,
        "Data/Serge/german/*/dev.*.data",
        r"dev\.(.*)\.data",
        score_paths,
        debug,
    )


def get_dwug_data_annotated_only(score_paths: list, debug=False):
    return get_dwug_data(score_paths, True, debug)


def get_dwug_old_data(score_paths: list, annotated_only=False, debug=False):
    sentences_path = "datasets_unlabeled/se20lscd_v2/de/unlabeled-old.tsv"
    senses_path = "datasets/se20lscd_v2/de/sense-old.tsv"
    return get_wsi_data(
        senses_path if annotated_only else sentences_path,
        senses_path,
        "Data/Serge/german/*/dev.*.data",
        r"dev\.(.*)\.data",
        score_paths,
        debug,
    )


def get_dwug_old_data_annotated_only(score_paths: list, debug=False):
    return get_dwug_old_data(score_paths, True, debug)


def get_dwug_new_data(score_paths: list, annotated_only=False, debug=False):
    sentences_path = "datasets_unlabeled/se20lscd_v2/de/unlabeled-new.tsv"
    senses_path = "datasets/se20lscd_v2/de/sense-new.tsv"
    return get_wsi_data(
        senses_path if annotated_only else sentences_path,
        senses_path,
        "Data/Serge/german/*/dev.*.data",
        r"dev\.(.*)\.data",
        score_paths,
        debug,
    )


def get_dwug_new_data_annotated_only(score_paths: list, debug=False):
    return get_dwug_new_data(score_paths, True, debug)


def get_btsrnc_data(score_paths: list, debug=False):
    senses, scores = get_wsi_data(
        "datasets/russe_bts-rnc/ru/train.tsv",
        "datasets/russe_bts-rnc/ru/train.tsv",
        "Data/russe_bts-rnc/ru/train/*/*.data",
        r"train/.+/(.*)\.data",
        score_paths,
        debug,
    )

    for x in scores:
        scores[x]["end1"] = scores[x]["end1"].apply(lambda y: str(int(y) + 1))
        scores[x]["end2"] = scores[x]["end2"].apply(lambda y: str(int(y) + 1))
    return senses, scores


def get_predictions(
    senses,
    scores,
    binarize,
    thresholds,
    scaling,
    word_level_threshold,
    model_hyperparameters,
    get_clusters,
    max_n_clusters,
    meta_information: dict = None,
):
    words = list(senses["word"].unique())
    scores_data = []
    best_score_data_per_word = []
    cluster_rand_scores = {x: 0 for x in range(2, max_n_clusters + 1)}
    total_ari_calinski = 0.0
    total_ari_silhouette = 0.0
    (
        total_adj_rand,
        total_silhouette_rand,
        total_calinski_rand,
        total_eigengap_rand,
    ) = (
        0,
        0,
        0,
        0,
    )
    skip_words = 0
    cache_name = meta_information.pop("cache_name", None)
    portion_dataset = meta_information.pop("portion_dataset")
    skip_iterations = 0
    no_experiment = meta_information.pop("no_experiment", None)

    if portion_dataset == "dwug_data_annotated_only":
        total_adj_rand_old = 0
        total_silhouette_rand_old = 0
        total_calinski_rand_old = 0
        total_eigengap_rand_old = 0
        total_adj_rand_new = 0
        total_silhouette_rand_new = 0
        total_calinski_rand_new = 0
        total_eigengap_rand_new = 0
        cluster_rand_scores_old = {x: 0 for x in range(2, max_n_clusters + 1)}
        cluster_rand_scores_new = {x: 0 for x in range(2, max_n_clusters + 1)}
        ari_eigengap_old = None
        ari_eigengap_new = None
        total_ari_silhouette_old = 0.0
        total_ari_calinski_old = 0.0
        total_ari_eigengap_old = 0.0
        total_ari_silhouette_new = 0.0
        total_ari_calinski_new = 0.0
        total_ari_eigengap_new = 0.0

    for idx, word in enumerate(words):
        word_senses = senses[senses["word"] == word]
        if word_senses.shape[0] < max_n_clusters + 1:
            skip_words += 1
            continue
        word_scores = scores[scores["word"] == word]
        contexts = sorted(
            list(
                set(
                    word_scores.apply(
                        lambda x: x["sentence1"]
                        + f"{x['start1']}-{x['end1']}",
                        axis=1,
                    )
                ).union(
                    set(
                        word_scores.apply(
                            lambda x: x["sentence2"]
                            + f"{x['start2']}-{x['end2']}",
                            axis=1,
                        )
                    )
                )
            )
        )

        random.Random(idx).shuffle(contexts)
        n_sentences = len(contexts)
        id_to_int = {x: i for i, x in enumerate(contexts)}
        word_senses["scores_exist?"] = word_senses.apply(
            lambda x: x["context"] + x["positions"] in id_to_int, axis=1
        )
        word_senses = word_senses[word_senses["scores_exist?"] == True]

        information_hyperparameters = {}
        information_hyperparameters.update(meta_information)
        information_hyperparameters["word"] = word

        adj_matrix = get_adj_matrix(
            word_scores,
            id_to_int,
            n_sentences,
            scaling,
            threshold=thresholds[word]
            if word_level_threshold
            else thresholds["all"],
            binarize=binarize,
            meta_information=information_hyperparameters,
        )

        G = _adjacency_matrix_to_nxgraph(adj_matrix)
        if len(G.nodes) == len(list(nx.connected_components(G))):
            skip_iterations += 1
            continue

        hyperparameters_for_printing = {
            "quantile": information_hyperparameters["quantile"],
            "dataset": information_hyperparameters["dataset"],
            "binarize": binarize,
        }
        information_hyperparameters["logging"].info(
            f"Processing word: {word}, hyperparameters: {str(hyperparameters_for_printing)}"
        )

        max_rand_score = -100
        max_silhoutte_score = -100
        max_silhoutte_rand_score = -100
        max_calinski_harabasz = -100
        max_calinski_harabasz_rand_score = -100
        eigengap_rand_score = None
        eigengap_n_clusters = eigengapHeuristic(adj_matrix, max_n_clusters)
        predicted_clusters = {}
        cluster_rand_scores_cache = {
            x: 0 for x in range(2, max_n_clusters + 1)
        }
        cluster_calinski = {x: 0 for x in range(2, max_n_clusters + 1)}
        cluster_silhouette = {x: 0 for x in range(2, max_n_clusters + 1)}

        if portion_dataset == "dwug_data_annotated_only":
            max_rand_score_old = -100
            max_silhouette_score_old = -100
            max_silhouette_rand_score_old = -100
            max_calinski_harabasz_old = -100
            max_calinski_harabasz_rand_score_old = -100
            eigengap_rand_score_old = None

            max_rand_score_new = -100
            max_silhouette_score_old = -100
            max_silhouette_rand_score_new = -100
            max_calinski_harabasz_new = -100
            max_calinski_harabasz_rand_score_new = -100
            eigengap_rand_score_new = None
            predicted_clusters_old = {}
            predicted_clusters_new = {}
            cluster_rand_scores_old_cache = {
                x: 0 for x in range(2, max_n_clusters + 1)
            }
            cluster_rand_scores_new_cache = {
                x: 0 for x in range(2, max_n_clusters + 1)
            }

        for n_clusters in range(1, max_n_clusters + 1):
            clusters = get_clusters(
                adj_matrix,
                n_clusters,
                model_hyperparameters,
                idx + seeds[meta_information["method"]][no_experiment],
            )
            pred_clusters = word_senses.apply(
                lambda x: int(
                    clusters[id_to_int[x["context"] + x["positions"]]]
                ),
                axis=1,
            )
            predicted_clusters[n_clusters] = pred_clusters.to_dict()
            rand = metrics.adjusted_rand_score(
                word_senses["gold_sense_id"], pred_clusters
            )

            if portion_dataset == "dwug_data_annotated_only":
                (
                    word_sense_old_data,
                    word_sense_new_data,
                ) = load_senses_and_scores(
                    meta_information["old_data"],
                    meta_information["new_data"],
                    word,
                    id_to_int,
                    meta_information["dataset"],
                )

                pred_clusters_old = word_sense_old_data.apply(
                    lambda x: int(
                        clusters[id_to_int[x["context"] + x["positions"]]]
                    ),
                    axis=1,
                )
                predicted_clusters_old[
                    n_clusters
                ] = pred_clusters_old.to_dict()
                rand_old = metrics.adjusted_rand_score(
                    word_sense_old_data["gold_sense_id"], pred_clusters_old
                )

                pred_clusters_new = word_sense_new_data.apply(
                    lambda x: int(
                        clusters[id_to_int[x["context"] + x["positions"]]]
                    ),
                    axis=1,
                )
                predicted_clusters_new[
                    n_clusters
                ] = pred_clusters_new.to_dict()
                rand_new = metrics.adjusted_rand_score(
                    word_sense_new_data["gold_sense_id"], pred_clusters_new
                )

            if n_clusters != 1:
                cluster_rand_scores[n_clusters] += rand
                cluster_rand_scores_cache[n_clusters] += rand
                if portion_dataset == "dwug_data_annotated_only":
                    cluster_rand_scores_old[n_clusters] += rand_old
                    cluster_rand_scores_new[n_clusters] += rand_new
                    cluster_rand_scores_old_cache[n_clusters] += rand_old
                    cluster_rand_scores_new_cache[n_clusters] += rand_new

                # silouhette & calinski_harabasz is not defined for 1 cluster,
                # hence we need a default value, here -1
                try:
                    silhouette = metrics.silhouette_score(adj_matrix, clusters)
                    calinski_harabasz = metrics.calinski_harabasz_score(
                        adj_matrix, clusters
                    )
                    cluster_calinski[n_clusters] += calinski_harabasz
                    cluster_silhouette[n_clusters] += silhouette
                except ValueError:
                    silhouette = -1
                    calinski_harabasz = -1
                    cluster_calinski[n_clusters] = -1
                    cluster_silhouette[n_clusters] = -1

                if silhouette > max_silhoutte_score:
                    max_silhoutte_score = silhouette
                    max_silhoutte_rand_score = rand
                if calinski_harabasz > max_calinski_harabasz:
                    max_calinski_harabasz = calinski_harabasz
                    max_calinski_harabasz_rand_score = rand
            else:
                silhouette, calinski_harabasz = None, None

            if rand > max_rand_score:
                max_rand_score = rand
            if eigengap_n_clusters[0] == n_clusters:
                eigengap_rand_score = rand
                if portion_dataset == "dwug_data_annotated_only":
                    ari_eigengap_old = rand_old
                    ari_eigengap_new = rand_new

            if portion_dataset == "dwug_data_annotated_only":
                if rand_old > max_rand_score_old:
                    max_rand_score_old = rand_old
                if rand_new > max_rand_score_new:
                    max_rand_score_new = rand_new

            scores_data.append(
                {
                    "word": word,
                    "n_clusters": n_clusters,
                    "adjusted_rand_score": rand,
                    "silhouette": silhouette,
                    "calinski_harabasz": calinski_harabasz,
                    "eigengap_idx": eigengap_n_clusters.index(n_clusters) + 1,
                }
            )
        list_cluster_calinski = list(cluster_calinski.values())
        list_cluster_silhouette = list(cluster_silhouette.values())
        index_ari_calinski = (
            list_cluster_calinski.index(max(list_cluster_calinski)) + 2
        )
        index_ari_silhouette = (
            list_cluster_silhouette.index(max(list_cluster_silhouette)) + 2
        )

        param = {
            "adjusted_rand_score": max_rand_score,
            "silhouette": max_silhoutte_rand_score,
            "calinski_harabasz": max_calinski_harabasz_rand_score,
            "eigengap": eigengap_rand_score,
            "adjusted_rand_score_old": -3.0,
            "ari_silhouette_old": -3.0,
            "ari_calinski_old": -3.0,
            "ari_eigengap_old": -3.0,
            "adjusted_rand_score_new": -3.0,
            "ari_silhouette_new": -3.0,
            "ari_calinski_new": -3.0,
            "ari_eigengap_new": -3.0,
            "ari_silhouette": cluster_rand_scores_cache[index_ari_silhouette],
            "ari_calinski": cluster_rand_scores_cache[index_ari_calinski],
            "ari_eigengap": eigengap_rand_score,
            "number_cluster_selected_by_silhouette": index_ari_silhouette,
            "number_cluster_selected_by_calinski": index_ari_calinski,
            "number_cluster_selected_by_eigengap": eigengap_n_clusters[0],
            "no_clusters_results": cluster_rand_scores_cache,
            "no_clusters_eigengap": eigengap_n_clusters[0],
        }

        if portion_dataset == "dwug_data_annotated_only":
            param["adjusted_rand_score_old"] = max_rand_score_old
            param["ari_silhouette_old"] = cluster_rand_scores_old_cache[
                index_ari_silhouette
            ]
            param["ari_calinski_old"] = cluster_rand_scores_old_cache[
                index_ari_calinski
            ]
            param["ari_eigengap_old"] = ari_eigengap_old
            param["adjusted_rand_score_new"] = max_rand_score_new
            param["ari_silhouette_new"] = cluster_rand_scores_new_cache[
                index_ari_silhouette
            ]
            param["ari_calinski_new"] = cluster_rand_scores_new_cache[
                index_ari_calinski
            ]
            param["ari_eigengap_new"] = ari_eigengap_new
        else:
            param.pop("adjusted_rand_score_old", None)
            param.pop("adjusted_rand_score_new", None)
            param.pop("ari_silhouette_old", None)
            param.pop("ari_calinski_old", None)
            param.pop("ari_eigengap_old", None)
            param.pop("ari_silhouette_new", None)
            param.pop("ari_calinski_new", None)
            param.pop("ari_eigengap_new", None)

        df_cache = pd.DataFrame([param])
        df_cache["score_path"] = [information_hyperparameters["dataset"]]
        df_cache["parameters"] = [
            {
                "word": word,
                "binarize": binarize,
                "quantile": information_hyperparameters["quantile"],
                "word_level_threshold": word_level_threshold,
            }
        ]
        df_cache["gold_id"] = [str(word_senses["gold_sense_id"].to_dict())]
        df_cache["predicted_clusters"] = [str(predicted_clusters)]
        df_cache.to_csv(f"{cache_name}", mode="a", header=False, index=False)

        information_hyperparameters["logging"].info("Done")
        information_hyperparameters["logging"].info("\n\n")

        total_adj_rand += max_rand_score
        total_silhouette_rand += max_silhoutte_rand_score
        total_calinski_rand += max_calinski_harabasz_rand_score
        total_eigengap_rand += eigengap_rand_score
        total_ari_silhouette += cluster_rand_scores_cache[index_ari_silhouette]
        total_ari_calinski += cluster_rand_scores_cache[index_ari_calinski]
        param["word"] = word
        best_score_data_per_word.append(param)

        if portion_dataset == "dwug_data_annotated_only":
            total_adj_rand_old += max_rand_score_old
            total_silhouette_rand_old += max_silhouette_rand_score_old
            total_calinski_rand_old += max_calinski_harabasz_rand_score_old
            total_eigengap_rand_old += (
                0.0
                if eigengap_rand_score_old is None
                else eigengap_rand_score_old
            )

            total_adj_rand_new += max_rand_score_new
            total_silhouette_rand_new += max_silhouette_rand_score_new
            total_calinski_rand_new += max_calinski_harabasz_rand_score_new
            total_eigengap_rand_new += (
                0.0
                if eigengap_rand_score_new is None
                else eigengap_rand_score_new
            )
            total_ari_silhouette_old += cluster_rand_scores_old_cache[
                index_ari_silhouette
            ]
            total_ari_calinski_old += cluster_rand_scores_old_cache[
                index_ari_calinski
            ]
            total_ari_eigengap_old += ari_eigengap_old
            total_ari_silhouette_new += cluster_rand_scores_new_cache[
                index_ari_silhouette
            ]
            total_ari_calinski_new += cluster_rand_scores_new_cache[
                index_ari_calinski
            ]
            total_ari_eigengap_new += ari_eigengap_new

    total_adj_rand /= len(words) - skip_words - skip_iterations
    total_silhouette_rand /= len(words) - skip_words - skip_iterations
    total_calinski_rand /= len(words) - skip_words - skip_iterations
    total_eigengap_rand /= len(words) - skip_words - skip_iterations
    total_ari_silhouette /= len(words) - skip_words - skip_iterations
    total_ari_calinski /= len(words) - skip_words - skip_iterations
    cluster_rand_scores = {
        k: v / len(words) for k, v in cluster_rand_scores.items()
    }

    if portion_dataset == "dwug_data_annotated_only":
        total_adj_rand_old /= len(words) - skip_words - skip_iterations
        total_silhouette_rand_old /= len(words) - skip_words - skip_iterations
        total_calinski_rand_old /= len(words) - skip_words - skip_iterations
        total_eigengap_rand_old /= len(words) - skip_words - skip_iterations
        cluster_rand_scores_old = {
            k: v / len(words) for k, v in cluster_rand_scores_old.items()
        }

        total_adj_rand_new /= len(words) - skip_words - skip_iterations
        total_silhouette_rand_new = len(words) - skip_words - skip_iterations
        total_calinski_rand_new = len(words) - skip_words - skip_iterations
        total_eigengap_rand_new = len(words) - skip_words - skip_iterations
        cluster_rand_scores_new = {
            k: v / len(words) for k, v in cluster_rand_scores_new.items()
        }
        total_ari_silhouette_old /= len(words) - skip_words - skip_iterations
        total_ari_calinski_old /= len(words) - skip_words - skip_iterations
        total_ari_eigengap_old /= len(words) - skip_words - skip_iterations
        total_ari_silhouette_new /= len(words) - skip_words - skip_iterations
        total_ari_calinski_new /= len(words) - skip_words - skip_iterations
        total_ari_eigengap_new /= len(words) - skip_words - skip_iterations

        answer = (
            scores_data,
            {
                "adjusted_rand_score": total_adj_rand,
                "silhouette": total_silhouette_rand,
                "calinski_harabasz": total_calinski_rand,
                "eigengap": total_eigengap_rand,
                "ari_silhouette": total_ari_silhouette,
                "ari_calinski": total_ari_calinski,
                "ari_eigengap": total_eigengap_rand,
            },
            cluster_rand_scores,
            best_score_data_per_word,
            {
                "adjusted_rand_score_old": total_adj_rand_old,
                "ari_silhouette_old": total_ari_silhouette_old,
                "ari_calinski_old": total_ari_calinski_old,
                "ari_eigengap_old": total_ari_eigengap_old,
            },
            cluster_rand_scores_old,
            {
                "adjusted_rand_score_new": total_adj_rand_new,
                "ari_silhouette_new": total_ari_silhouette_new,
                "ari_calinski_new": total_ari_calinski_new,
                "ari_eigengap_new": total_ari_eigengap_new,
            },
            cluster_rand_scores_new,
        )
    else:
        answer = (
            scores_data,
            {
                "adjusted_rand_score": total_adj_rand,
                "silhouette": total_silhouette_rand,
                "calinski_harabasz": total_calinski_rand,
                "eigengap": total_eigengap_rand,
                "ari_silhouette": total_ari_silhouette,
                "ari_calinski": total_ari_calinski,
                "ari_eigengap": total_eigengap_rand,
            },
            cluster_rand_scores,
            best_score_data_per_word,
        )

    return answer


def get_predictions_without_nclusters(
    senses,
    scores,
    binarize,
    thresholds,
    scaling,
    word_level_threshold,
    model_hyperparameters,
    get_clusters,
    meta_information: dict = None,
):
    words = list(senses["word"].unique())
    scores_data = []
    total_adj_rand = 0
    total_adj_rand_old_data = 0
    total_adj_rand_new_data = 0
    skip_words = 0
    is_already_processed = False
    is_not_processed = None
    ocurred_exception = False
    skip_iterations = 0

    cache_name = meta_information.pop("cache_name", None)
    cache_items = meta_information.pop("cache_items", None)
    cache_rand_score = meta_information.pop("cache_rand_score", None)
    cache_status = meta_information.pop("cache_status", None)
    portion_dataset = meta_information.pop("portion_dataset", None)
    no_experiment = meta_information.pop("no_experiment", None)

    for idx, word in enumerate(words):
        param = {
            "word": word,
            "quantile": meta_information["quantile"],
            "dataset": meta_information["dataset"],
            "binarize": binarize,
        }

        if meta_information["method"] == "wsbm":
            param.update(model_hyperparameters)

        for index, item in enumerate(cache_items):
            if ast.literal_eval(item) == param:
                if cache_status[index] == "done":
                    is_already_processed = True
                    total_adj_rand += float(cache_rand_score[index])
                    break
                else:
                    is_not_processed = True
                    break

        if is_already_processed is True:
            is_already_processed = False
            continue

        word_senses = senses[senses["word"] == word]
        word_scores = scores[scores["word"] == word]
        contexts = sorted(
            list(
                set(
                    word_scores.apply(
                        lambda x: x["sentence1"]
                        + f"{x['start1']}-{x['end1']}",
                        axis=1,
                    )
                ).union(
                    set(
                        word_scores.apply(
                            lambda x: x["sentence2"]
                            + f"{x['start2']}-{x['end2']}",
                            axis=1,
                        )
                    )
                )
            )
        )

        random.Random(idx).shuffle(contexts)
        n_sentences = len(contexts)
        id_to_int = {x: i for i, x in enumerate(contexts)}
        word_senses["scores_exist?"] = word_senses.apply(
            lambda x: x["context"] + x["positions"] in id_to_int, axis=1
        )
        word_senses = word_senses[word_senses["scores_exist?"] == True]

        information_hyperparameters = {}
        information_hyperparameters.update(meta_information)
        information_hyperparameters["word"] = word

        adj_matrix = get_adj_matrix(
            word_scores,
            id_to_int,
            n_sentences,
            scaling,
            threshold=thresholds[word]
            if word_level_threshold
            else thresholds["all"],
            binarize=binarize,
            meta_information=information_hyperparameters,
        )

        G = _adjacency_matrix_to_nxgraph(adj_matrix)
        if len(G.nodes) == len(list(nx.connected_components(G))):
            skip_iterations += 1
            continue

        hyperparameters_for_printing = {
            "quantile": information_hyperparameters["quantile"],
            "dataset": information_hyperparameters["dataset"],
            "binarize": binarize,
        }
        information_hyperparameters["logging"].info(
            f"Processing word: {word}, hyperparameters: {str(hyperparameters_for_printing)}"
        )

        if is_not_processed is None:
            df = pd.DataFrame(
                {
                    "adjusted_rand_score": [-1.0],
                    "parameters": [param],
                    "status": ["processing"],
                    "gold_id": [None],
                    "predicted_clusters": [None],
                }
            )
            df.to_csv(f"{cache_name}", mode="a", index=False, header=False)
        else:
            is_not_processed = None

        try:
            clusters = get_clusters(
                adj_matrix,
                model_hyperparameters,
                idx + seeds[meta_information["method"]][no_experiment],
            )
        except Exception as e:
            ocurred_exception = True
            print(e)

        df = pd.read_csv(f"{cache_name}")
        n_row = len(df["parameters"].to_numpy())
        df.loc[n_row - 1, "status"] = "done"

        if ocurred_exception is False:
            pred_clusters = word_senses.apply(
                lambda x: int(
                    clusters[id_to_int[x["context"] + x["positions"]]]
                ),
                axis=1,
            )
            rand = metrics.adjusted_rand_score(
                word_senses["gold_sense_id"], pred_clusters
            )
            df.loc[n_row - 1, "gold_id"] = str(
                word_senses["gold_sense_id"].to_dict()
            )
            df.loc[n_row - 1, "predicted_clusters"] = str(
                pred_clusters.to_dict()
            )

            if portion_dataset == "dwug_data_annotated_only":
                word_sense_old_data, word_score_old_data = meta_information[
                    "old_data"
                ]
                word_sense_new_data, word_score_new_data = meta_information[
                    "new_data"
                ]

                word_score_old_data = word_score_old_data[
                    meta_information["dataset"]
                ]
                word_score_new_data = word_score_new_data[
                    meta_information["dataset"]
                ]

                word_sense_old_data = word_sense_old_data[
                    word_sense_old_data["word"] == word
                ]
                word_score_old_data = word_score_old_data[
                    word_score_old_data["word"] == word
                ]
                word_sense_new_data = word_sense_new_data[
                    word_sense_new_data["word"] == word
                ]
                word_score_new_data = word_score_new_data[
                    word_score_new_data["word"] == word
                ]

                word_sense_old_data[
                    "scores_exist?"
                ] = word_sense_old_data.apply(
                    lambda x: x["context"] + x["positions"] in id_to_int,
                    axis=1,
                )
                word_sense_old_data = word_sense_old_data[
                    word_sense_old_data["scores_exist?"] == True
                ]
                word_sense_new_data[
                    "scores_exist?"
                ] = word_sense_new_data.apply(
                    lambda x: x["context"] + x["positions"] in id_to_int,
                    axis=1,
                )
                word_sense_new_data = word_sense_new_data[
                    word_sense_new_data["scores_exist?"] == True
                ]

                pred_clusters_old_data = word_sense_old_data.apply(
                    lambda x: int(
                        clusters[id_to_int[x["context"] + x["positions"]]]
                    ),
                    axis=1,
                )
                pred_clusters_new_data = word_sense_new_data.apply(
                    lambda x: int(
                        clusters[id_to_int[x["context"] + x["positions"]]]
                    ),
                    axis=1,
                )

                rand_old_data = metrics.adjusted_rand_score(
                    word_sense_old_data["gold_sense_id"],
                    pred_clusters_old_data,
                )
                rand_new_data = metrics.adjusted_rand_score(
                    word_sense_new_data["gold_sense_id"],
                    pred_clusters_new_data,
                )

                df.loc[n_row - 1, "adjusted_rand_score_old"] = rand_old_data
                df.loc[n_row - 1, "adjusted_rand_score_new"] = rand_new_data

        else:
            rand = 0.0
            ocurred_exception = False

        information_hyperparameters["logging"].info("Done")
        information_hyperparameters["logging"].info("\n\n")

        df.loc[n_row - 1, "adjusted_rand_score"] = rand
        df.to_csv(f"{cache_name}", index=False)

        scores_data.append({"word": word, "adjusted_rand_score": rand})

        total_adj_rand += rand
        if portion_dataset == "dwug_data_annotated_only":
            total_adj_rand_old_data += rand_old_data
            total_adj_rand_new_data += rand_new_data

    total_adj_rand /= len(words) - skip_iterations
    adj_rand_score = {"adjusted_rand_score": total_adj_rand}

    if portion_dataset == "dwug_data_annotated_only":
        total_adj_rand_old_data /= len(words) - skip_iterations
        total_adj_rand_new_data /= len(words) - skip_iterations

        adj_rand_score["adjusted_rand_score_old"] = total_adj_rand_old_data
        adj_rand_score["adjusted_rand_score_new"] = total_adj_rand_new_data

    return scores_data, adj_rand_score


def get_scaler(values: np.array, f: object = MinMaxScaler) -> object:
    f = f()
    scaling = f.fit(values)
    return scaling


def get_scaling(scores) -> dict:
    scalers = dict()
    for score_path in scores:
        scalers[score_path] = get_scaler(
            scores[score_path]["score"].to_numpy().reshape(-1, 1)
        )

    return scalers


def get_thresholds(scores: pd.DataFrame) -> dict:
    thresholds = dict()
    words = None

    for score_path in scores:
        if words is None:
            words = list(scores[score_path]["word"].unique())
        thresholds[score_path] = dict()
        for quantile in range(10):
            if not quantile:
                thresholds[score_path][0] = {"all": 0.5}
            else:
                thresholds[score_path][quantile] = {
                    "all": np.quantile(
                        scores[score_path]["score"], quantile / 10
                    )
                }
                for idx, word in enumerate(words):
                    thresholds[score_path][quantile][word] = np.quantile(
                        scores[score_path][scores[score_path]["word"] == word][
                            "score"
                        ],
                        quantile / 10,
                    )

    return thresholds


def merge_dicts(dict1, dict2):
    d = {}
    d.update(dict1)
    d.update(dict2)
    return d


def grid_search(
    get_data,
    get_clusters,
    score_paths: dict,
    model_hyperparameter_combinations: list,
    max_n_clusters: int,
    method: str = None,
    logger_message: dict = None,
    cache: dict = None,
    run_experiments: int = 1,
    dataset: str = None,
):
    senses, scores = get_data(score_paths)
    if dataset == "dwug_data_annotated_only":
        senses_old_data, scores_old_data = get_dwug_old_data_annotated_only(
            score_paths
        )
        senses_new_data, scores_new_data = get_dwug_new_data_annotated_only(
            score_paths
        )

    thresholds = get_thresholds(scores)
    scalers = get_scaling(scores)
    # parameters = pd.read_csv(cache_name)
    # cache_items = parameters['Parameters'].to_numpy()
    # cache_rand_score = parameters['Rand_score'].to_numpy()
    # cache_status = parameters['Status'].to_numpy()

    hyperparameter_combinations = []
    for binarize in [True, False]:
        for percentile in range(10):
            for word_level_threshold in [False]:  # removed True
                if not percentile and word_level_threshold:
                    continue
                for score_path in score_paths:
                    for (
                        model_hyperparameters
                    ) in model_hyperparameter_combinations:
                        hyperparameter_combinations.append(
                            {
                                "binarize": binarize,
                                "percentile": percentile,
                                "word_level_threshold": word_level_threshold,
                                "score_path": score_path,
                                "model_hyperparameters": model_hyperparameters,
                            }
                        )

    n = len(hyperparameter_combinations)
    logging = None

    if logger_message is not None and logger_message["logging"]:
        logging = logger_message["logging"]
        logging.info(f"======================{dataset}====================")

    for no_experiment in range(1, run_experiments + 1):
        max_scores = None
        best_combination = None
        max_cluster_scores = None
        best_cluster_combination = None
        best_results_per_word_combination = {}
        df = pd.DataFrame()

        for idx, hyperparameters in enumerate(hyperparameter_combinations):
            extra_hyperparameters = {
                "dataset": hyperparameters["score_path"],
                "quantile": hyperparameters["percentile"],
                "method": method,
                "logging": logging,
                "cache_name": cache["name"].format(
                    result=f"results_{no_experiment}", method=method
                ),
                "portion_dataset": dataset,
                "no_experiment": no_experiment
                # "cache_items": cache_items,
                # "cache_rand_score": cache_rand_score,
                # "cache_status": cache_status
            }

            if dataset == "dwug_data_annotated_only":
                extra_hyperparameters["old_data"] = (
                    senses_old_data,
                    scores_old_data,
                )
                extra_hyperparameters["new_data"] = (
                    senses_new_data,
                    scores_new_data,
                )

                # combination_data,
                # combination_scores,
                # combination_cluster_rand_scores,
                # best_results_per_word,

            answer = get_predictions(
                senses,
                scores[hyperparameters["score_path"]],
                hyperparameters["binarize"],
                thresholds[hyperparameters["score_path"]][
                    hyperparameters["percentile"]
                ],
                scalers[hyperparameters["score_path"]],
                hyperparameters["word_level_threshold"],
                hyperparameters["model_hyperparameters"],
                get_clusters,
                max_n_clusters,
                extra_hyperparameters,
            )

            combination_data = answer[0]
            combination_scores = answer[1]
            combination_cluster_rand_scores = answer[2]
            best_results_per_word = answer[3]

            if dataset == "dwug_data_annotated_only":
                combination_scores_old = answer[4]
                combination_cluster_rand_scores_old = answer[5]
                combination_scores_new = answer[6]
                combination_cluster_rand_scores_new = answer[7]

            if logging is not None:
                logging.info(
                    f"{idx + 1}/{len(hyperparameter_combinations)} steps executed"
                )

            aux = {**hyperparameters}
            if hyperparameters["word_level_threshold"] is False:
                aux["threshold"] = thresholds[hyperparameters["score_path"]][
                    hyperparameters["percentile"]
                ]["all"]
            if hyperparameters["binarize"] is False:
                aux["min_value_scaling"] = scalers[
                    hyperparameters["score_path"]
                ].data_min_
                aux["max_value_scaling"] = scalers[
                    hyperparameters["score_path"]
                ].data_max_
            print(combination_scores, aux)

            if dataset == "dwug_data_annotated_only":
                combination_scores = {
                    **combination_scores,
                    **combination_scores_old,
                    **combination_scores_new,
                }
            df_results = pd.DataFrame([combination_scores])
            df_results["no_clusters_2"] = [combination_cluster_rand_scores[2]]
            df_results["no_clusters_3"] = [combination_cluster_rand_scores[3]]
            df_results["no_clusters_4"] = [combination_cluster_rand_scores[4]]
            df_results["no_clusters_5"] = [combination_cluster_rand_scores[5]]
            df_results["score_path"] = [hyperparameters["score_path"]]
            aux.pop("model_hyperparameters")
            if "min_value_scaling" in aux:
                aux.pop("min_value_scaling")
                aux.pop("max_value_scaling")
            if "threshold" in aux:
                aux.pop("threshold")

            df_results["parameters"] = [aux]
            result_name = cache["result_name"].format(
                result=f"results_{no_experiment}", method=method
            )
            df_results.to_csv(
                f"{result_name}", mode="a", header=False, index=False
            )

            if idx % 10 == 0 or idx == n - 1:
                print("Saving. Total results:", idx + 1, "/", n)

            if max_scores == None:
                max_scores = combination_scores
                best_combination = {
                    x: hyperparameters for x in combination_scores
                }
            else:
                for x in combination_scores:
                    if combination_scores[x] > max_scores[x]:
                        max_scores[x] = combination_scores[x]
                        best_combination[x] = hyperparameters

            if max_cluster_scores == None:
                max_cluster_scores = combination_cluster_rand_scores
                best_cluster_combination = {
                    n: hyperparameters for n in combination_cluster_rand_scores
                }
            else:
                for x in combination_cluster_rand_scores:
                    if (
                        combination_cluster_rand_scores[x]
                        > max_cluster_scores[x]
                    ):
                        max_cluster_scores[
                            x
                        ] = combination_cluster_rand_scores[x]
                        best_cluster_combination[x] = hyperparameters

            if (
                not hyperparameters["score_path"]
                in best_results_per_word_combination
            ):
                best_results_per_word_combination[
                    hyperparameters["score_path"]
                ] = best_results_per_word
            else:
                for item in best_results_per_word:
                    for combination in best_results_per_word_combination[
                        hyperparameters["score_path"]
                    ]:
                        if item["word"] == combination["word"]:
                            if (
                                combination["adjusted_rand_score"]
                                < item["adjusted_rand_score"]
                            ):
                                combination["adjusted_rand_score"] = item[
                                    "adjusted_rand_score"
                                ]

                            if combination["silhouette"] < item["silhouette"]:
                                combination["silhouette"] = item["silhouette"]

                            if (
                                combination["calinski_harabasz"]
                                < item["calinski_harabasz"]
                            ):
                                combination["calinski_harabasz"] = item[
                                    "calinski_harabasz"
                                ]

                            if combination["eigengap"] < item["eigengap"]:
                                combination["eigengap"] = item["eigengap"]

                            break

        best_n_cluster_score = max(max_cluster_scores.values())
        best_n_cluster = [
            x
            for x in max_cluster_scores
            if max_cluster_scores[x] == best_n_cluster_score
        ][0]
        max_scores["fixed"] = best_n_cluster_score
        best_combination["fixed"] = {
            "best_n_cluster": best_n_cluster,
            **best_cluster_combination[best_n_cluster],
        }
        print(max_scores)
        print(best_combination)

    # return max_scores, best_combination


def grid_search_without_nclusters(
    get_data,
    get_clusters,
    score_paths: dict,
    model_hyperparameter_combinations: list,
    include_binarize: bool = False,
    file_name_results_per_word: str = "results_words",
    method: str = None,
    logger_message: dict = None,
    cache: dict = None,
    run_experiments: int = 1,
    dataset: str = None,
):
    senses, scores = get_data(score_paths)
    if dataset == "dwug_data_annotated_only":
        senses_old_data, scores_old_data = get_dwug_old_data_annotated_only(
            score_paths
        )
        senses_new_data, scores_new_data = get_dwug_new_data_annotated_only(
            score_paths
        )

    thresholds = get_thresholds(scores)
    scalers = get_scaling(scores)
    try:
        parameters = pd.read_csv(cache["name"])
        cache_items = parameters["parameters"].to_numpy()
        cache_rand_score = parameters["adjusted_rand_score"].to_numpy()
        cache_status = parameters["status"].to_numpy()
    except Exception:
        cache_items = []
        cache_rand_score = []
        cache_status = []

    hyperparameter_combinations = []
    for binarize in (
        [True]
        if method == "wsbm"
        else [True, False]
        if include_binarize
        else [False]
    ):
        for percentile in range(10):
            for word_level_threshold in [False]:
                if not percentile and word_level_threshold:
                    continue
                for score_path in score_paths:
                    if isinstance(model_hyperparameter_combinations, dict):
                        combinations = model_hyperparameter_combinations[
                            score_path
                        ]
                    else:
                        combinations = model_hyperparameter_combinations

                    for model_hyperparameters in combinations:
                        hyperparameter_combinations.append(
                            {
                                "binarize": binarize,
                                "percentile": percentile,
                                "word_level_threshold": word_level_threshold,
                                "score_path": score_path,
                                "model_hyperparameters": model_hyperparameters,
                            }
                        )

    n = len(hyperparameter_combinations)
    logging = None

    if logger_message is not None and logger_message["logging"]:
        logging = logger_message["logging"]
        logging.info(f"======================{dataset}====================")

    for no_experiment in range(1, run_experiments + 1):
        max_scores = None
        best_combination = None
        best_results_per_word_combination = {}

        for idx, hyperparameters in enumerate(hyperparameter_combinations):
            extra_hyperparameters = {
                "dataset": hyperparameters["score_path"],
                "quantile": hyperparameters["percentile"],
                "method": method,
                "logging": logging,
                "cache_name": cache["name"].format(
                    result=f"results_{no_experiment}", method=method
                ),
                "cache_items": cache_items,
                "cache_rand_score": cache_rand_score,
                "cache_status": cache_status,
                "portion_dataset": dataset,
                "no_experiment": no_experiment,
            }

            if dataset == "dwug_data_annotated_only":
                extra_hyperparameters["old_data"] = (
                    senses_old_data,
                    scores_old_data,
                )
                extra_hyperparameters["new_data"] = (
                    senses_new_data,
                    scores_new_data,
                )

            (
                combination_data,
                combination_scores,
            ) = get_predictions_without_nclusters(
                senses,
                scores[hyperparameters["score_path"]],
                hyperparameters["binarize"],
                thresholds[hyperparameters["score_path"]][
                    hyperparameters["percentile"]
                ],
                scalers[hyperparameters["score_path"]],
                hyperparameters["word_level_threshold"],
                hyperparameters["model_hyperparameters"],
                get_clusters,
                extra_hyperparameters,
            )

            if dataset == "dwug_data_annotated_only":
                results_adjusted_rand_score_old = combination_scores[
                    "adjusted_rand_score_old"
                ]
                results_adjusted_rand_score_new = combination_scores[
                    "adjusted_rand_score_new"
                ]
                combination_scores = {
                    "adjusted_rand_score": combination_scores[
                        "adjusted_rand_score"
                    ]
                }

            if logging is not None:
                logging.info(
                    f"{idx + 1}/{len(hyperparameter_combinations)} steps executed"
                )

            aux = {**hyperparameters}
            if hyperparameters["word_level_threshold"] is False:
                aux["threshold"] = thresholds[hyperparameters["score_path"]][
                    hyperparameters["percentile"]
                ]["all"]
            if hyperparameters["binarize"] is False:
                aux["min_value_scaling"] = scalers[
                    hyperparameters["score_path"]
                ].data_min_
                aux["max_value_scaling"] = scalers[
                    hyperparameters["score_path"]
                ].data_max_
            print(combination_scores, aux)

            df_results = pd.DataFrame([combination_scores])

            if "min_value_scaling" in aux:
                aux.pop("min_value_scaling")
                aux.pop("max_value_scaling")
            if "threshold" in aux:
                aux.pop("threshold")

            df_results["parameters"] = [aux]
            df_results["model_hyperparameters"] = [
                aux.pop("model_hyperparameters")
            ]
            result_name = cache["result_name"].format(
                result=f"results_{no_experiment}", method=method
            )
            if dataset == "dwug_data_annotated_only":
                df_results["adjusted_rand_score_old"] = [
                    results_adjusted_rand_score_old
                ]
                df_results["adjusted_rand_score_new"] = [
                    results_adjusted_rand_score_new
                ]
            df_results.to_csv(
                f"{result_name}", mode="a", header=False, index=False
            )

            if idx % 10 == 0 or idx == n - 1:
                print("Saving. Total results:", idx + 1, "/", n)

            if max_scores == None:
                max_scores = combination_scores
                best_combination = {
                    x: hyperparameters for x in combination_scores
                }
            else:
                for x in combination_scores:
                    if combination_scores[x] > max_scores[x]:
                        max_scores[x] = combination_scores[x]
                        best_combination[x] = hyperparameters

            if (
                hyperparameters["score_path"]
                not in best_results_per_word_combination
            ):
                best_results_per_word_combination[
                    hyperparameters["score_path"]
                ] = combination_data
            else:
                for item in best_results_per_word_combination[
                    hyperparameters["score_path"]
                ]:
                    for word, score in combination_data:
                        if (
                            word.strip() == item["word"].strip()
                            and score > item["score"]
                        ):
                            item[score] = score
                            break

        print(max_scores)
        print(best_combination)

    # return max_scores, best_combination
