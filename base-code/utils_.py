import os
import pickle
from typing import List

import numpy as np
import pandas as pd


def _save_graph(
    method: str = None,
    dataset: str = None,
    word: str = None,
    binarize: bool = None,
    quantile: int = None,
    thresholds: List = [],
    graphs: np.array = [],
):
    if len(thresholds) > 0 and len(thresholds) != 2:
        raise "Threshold list only take two values"

    if os.path.exists("graph_analysis") is False:
        os.mkdir("graph_analysis")

    if os.path.exists(f"graph_analysis/{method}") is False:
        os.mkdir(f"graph_analysis/{method}")

    if os.path.exists(f"graph_analysis/{method}/{dataset}") is False:
        os.mkdir(f"graph_analysis/{method}/{dataset}")

    if os.path.exists(f"graph_analysis/{method}/{dataset}/{word}") is False:
        os.mkdir(f"graph_analysis/{method}/{dataset}/{word}")

    if (
        os.path.exists(f"graph_analysis/{method}/{dataset}/{word}/binarize_{binarize}")
        is False
    ):
        os.mkdir(f"graph_analysis/{method}/{dataset}/{word}/binarize_{binarize}")

    if (
        os.path.exists(
            f"graph_analysis/{method}/{dataset}/{word}/binarize_{binarize}/quantile_{quantile}"
        )
        is False
    ):
        os.mkdir(
            f"graph_analysis/{method}/{dataset}/{word}/binarize_{binarize}/quantile_{quantile}"
        )

    np.save(
        f"graph_analysis/{method}/{dataset}/{word}/binarize_{binarize}/quantile_{quantile}/graph_with_thresholds.npy",
        graphs[0],
    )
    np.save(
        f"graph_analysis/{method}/{dataset}/{word}/binarize_{binarize}/quantile_{quantile}/graph_no_thresholds.npy",
        graphs[1],
    )
    with open(
        f"graph_analysis/{method}/{dataset}/{word}/binarize_{binarize}/quantile_{quantile}/thresholds.txt",
        "w",
    ) as f_out:
        f_out.write(f"Initial threshold: {thresholds[0]}\n")
        f_out.write(f"Final threshold: {thresholds[1]}\n")


def _save_results(results: dict, path_name_file: str) -> None:
    path = path_name_file.split("/")
    name_file = path[-1]
    path_results = "/".join(path[:-1])

    for dataset in results.keys():
        if os.path.exists(f"{path_results}/{dataset}") is False:
            os.mkdir(f"{path_results}/{dataset}")

        try:
            with open(f"{path_results}/{dataset}/{name_file}", "wb") as f_out:
                pickle.dump(results[dataset], f_out)
        except Exception as e:
            print(e)
            print("ERROR SAVING THE BEST RESULTS PER WORD")


def load_senses_and_scores(
    old_data: tuple[pd.DataFrame, dict[str, pd.DataFrame]],
    new_data: tuple[pd.DataFrame, dict[str, pd.DataFrame]],
    word: str,
    id_to_int: dict[str, int],
    path_dataset: str,
):
    word_sense_old_data, word_score_old_data = old_data
    word_sense_new_data, word_score_new_data = new_data
    word_score_old_data = word_score_old_data[path_dataset]
    word_score_new_data = word_score_new_data[path_dataset]

    word_sense_old_data = word_sense_old_data[word_sense_old_data["word"] == word]
    word_score_old_data = word_score_old_data[word_score_old_data["word"] == word]
    word_sense_new_data = word_sense_new_data[word_sense_new_data["word"] == word]
    word_score_new_data = word_score_new_data[word_score_new_data["word"] == word]

    word_sense_old_data["score_exists?"] = word_sense_old_data.apply(
        lambda x: x["context"] + x["positions"] in id_to_int, axis=1
    )
    word_sense_old_data = word_sense_old_data[
        word_sense_old_data["score_exists?"] == True
    ]
    word_sense_new_data["score_exists?"] = word_sense_new_data.apply(
        lambda x: x["context"] + x["positions"] in id_to_int, axis=1
    )
    word_sense_new_data = word_sense_new_data[
        word_sense_new_data["score_exists?"] == True
    ]

    return word_sense_old_data, word_sense_new_data
