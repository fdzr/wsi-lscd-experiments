from pathlib import Path
from ast import literal_eval as F
import sys
from typing import List
from pprint import pprint
import collections
import json
from datetime import datetime
import logging

import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.stats import spearmanr

from common import (
    get_dwug_data_annotated_only,
    get_dwug_old_data_annotated_only,
    get_dwug_new_data_annotated_only,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

DEBUG = False
INPUT_EXPERIMENTS = (
    "../cache_v2/results_{run}/{method}/{folder}/{folder}_{dataset}.csv"
    if DEBUG is True
    else "../outputs/experiment-results/results_{run}/{method}/{folder}/{folder}_{dataset}.csv"
)
OUTPUT = "../outputs"
OUTPUT_EXPERIMENTS = f"{OUTPUT}/experiment-results/"
OUTPUT_TO_AVG_PREDICTED_CLUSTERS = f"{OUTPUT}/avg_predicted_clusters/"

INPUT_FREQ_CLUSTERS = "../outputs/output_freq_clusters/run_{run}/{method}.csv"
INPUT_TO_AVG_PREDICTED_CLUSTERS = (
    "../outputs/avg_predicted_clusters/{dataset}/{method}.csv"
)


score_paths = {
    "rusemshift-finetune": "Data/l1ndotn_schemas/rusemshift/finetune/german/{0}/dev.*.scores",
    "rusemshift-train": "Data/l1ndotn_schemas/rusemshift/train/german/{0}/dev.*.scores",
    "ru-ru": "Data/l1ndotn_schemas/ru-ru/german/{0}/dev.*.scores",
    "en-en": "Data/l1ndotn_schemas/en-en/german/{0}/dev.*.scores",
}
METHODS = [
    "chinese_whispers",
    "correlation_clustering",
    "wsbm",
    "spectral_clustering",
]
DATASETS = [
    "dwug_data_annotated_only",
    "dwug_old_data_annotated_only",
    "dwug_new_data_annotated_only",
]

path_gold_data = "dwug_de/misc/dwug_de_sense/stats/maj_3/stats_groupings.csv"
gold_data = pd.read_csv(path_gold_data, sep="\t")
WORDS = gold_data.lemma.to_list()
NUMBER_OF_WORDS = len(WORDS)


def load_config_file(path=None):
    with open("config/config.json" if path is None else path, "r") as f_in:
        config = json.load(f_in)

    return config


CONFIG = load_config_file()


def load_gold_lscd_data() -> dict[str, float]:
    data = gold_data[["lemma", "change_graded"]]
    data = data.set_index("lemma")["change_graded"].to_dict()
    return data


def save_file(
    data: pd.DataFrame,
    path_to_save: str,
    header: bool = True,
    index: bool = False,
):
    logging.info("saving the results")
    data.to_csv(path_to_save, header=header, index=index)
    logging.info("saved...")


def load_predictions_set_words(
    data: pd.DataFrame,
) -> dict[str, dict[int, int]]:
    query = data.apply(
        lambda x: (F(x["parameters"])["word"], F(x["predicted_clusters"])),
        axis=1,
    )
    query = list(query)
    answer = {}
    for item in query:
        answer[item[0]] = item[1]

    return answer


def dataframe_index_to_context_id(senses: pd.DataFrame) -> dict[int, str]:
    indexes = senses.index.to_list()
    number_to_id = {}
    for id in indexes:
        number_to_id[id] = senses.loc[[id]].context_id.item()

    return number_to_id


def load_ids(senses: pd.DataFrame) -> List[str]:
    ids = set(senses.context_id.to_list())
    return ids


def file_exists(path: str):
    q = Path(path)
    return q.exists()


def create_and_save_csv(results: dict[str, float], method: str):
    data_to_save = {}
    try:
        config = CONFIG[method]
    except Exception:
        config = CONFIG["non-spectral_clustering"]

    for field in config["create_and_save_csv"]:
        if method == "spectral_clustering":
            data_to_save[field] = results[field.split("_", 1)[1]]
        else:
            data_to_save[field] = results[field.split("_")[1]]

    data = pd.DataFrame(data_to_save)
    logging.info(f"Saving the computed spearman for the method {method}")
    data.to_csv(
        f"{OUTPUT}/outputs_spearman/{method}_spearman.csv",
        header=True,
        index=False,
    )
    logging.info("saved...")


def save_analysis_freq_clusters(
    data: pd.DataFrame,
    freq_clusters_and_jsd,
    method,
    path_to_save: str,
):
    if method != "spectral_clustering":
        jsd = []
        freq1 = []
        freq2 = []
        information_data = pd.DataFrame(
            {
                "jsd": [],
                "freq_clusters_1": [],
                "freq_clusters_2": [],
                "parameters": [],
            }
        )

        for p in data.parameters.to_list():
            word = F(p)["word"]
            jsd.append(freq_clusters_and_jsd[word].jsd)
            freq1.append(freq_clusters_and_jsd[word].cluster_to_freq1)
            freq2.append(freq_clusters_and_jsd[word].cluster_to_freq2)

        information_data["jsd"] = jsd
        information_data["freq_clusters_1"] = freq1
        information_data["freq_clusters_2"] = freq2
        information_data["parameters"] = data.parameters.to_list()

    else:
        information_data = pd.DataFrame(
            {
                "jsd_silhouette": [],
                "jsd_calinski": [],
                "jsd_eigengap": [],
                "freq_clusters_1_silhouette": [],
                "freq_clusters_2_silhouette": [],
                "freq_clusters_1_calinski": [],
                "freq_clusters_2_calinski": [],
                "freq_clusters_1_eigengap": [],
                "freq_clusters_2_eigengap": [],
                "parameters": [],
            }
        )
        results = {
            "jsd_silhouette": [],
            "jsd_calinski": [],
            "jsd_eigengap": [],
            "freq_clusters_1_silhouette": [],
            "freq_clusters_2_silhouette": [],
            "freq_clusters_1_calinski": [],
            "freq_clusters_2_calinski": [],
            "freq_clusters_1_eigengap": [],
            "freq_clusters_2_eigengap": [],
        }

        for index in range(data.shape[0]):
            row = data.iloc[[index]]
            word = F(row.parameters.item())["word"]

            for validation_method in ["silhouette", "calinski", "eigengap"]:
                results[f"jsd_{validation_method}"].append(
                    freq_clusters_and_jsd[word][
                        int(
                            row[
                                f"number_cluster_selected_by_{validation_method}"
                            ].item()
                        )
                    ].jsd
                )
                results[f"freq_clusters_1_{validation_method}"].append(
                    freq_clusters_and_jsd[word][
                        row[
                            f"number_cluster_selected_by_{validation_method}"
                        ].item()
                    ].cluster_to_freq1
                )
                results[f"freq_clusters_2_{validation_method}"].append(
                    freq_clusters_and_jsd[word][
                        row[
                            f"number_cluster_selected_by_{validation_method}"
                        ].item()
                    ].cluster_to_freq2
                )

        for validation_method in ["silhouette", "calinski", "eigengap"]:
            information_data[f"jsd_{validation_method}"] = results[
                f"jsd_{validation_method}"
            ]
            information_data[f"freq_clusters_1_{validation_method}"] = results[
                f"freq_clusters_1_{validation_method}"
            ]
            information_data[f"freq_clusters_2_{validation_method}"] = results[
                f"freq_clusters_2_{validation_method}"
            ]

        information_data["parameters"] = data.parameters.to_list()

    exists_file = file_exists(path_to_save)
    information_data.to_csv(
        path_to_save,
        header=False if exists_file else True,
        index=False,
        mode="a",
    )


def compute_spearman(graded_lscd: dict[str, float], data=None) -> float:
    gold_data = load_gold_lscd_data()
    vector1 = [gold_data[word] for word in graded_lscd.keys()]
    if data is None:
        vector2 = [graded_lscd[word].jsd for word in graded_lscd.keys()]
    else:
        vector2 = [
            graded_lscd[word][data[word]].jsd for word in graded_lscd.keys()
        ]

    res = spearmanr(vector1, vector2)[0]

    return res


def compute_graded_lscd(
    predictions: dict[str, dict[int, int]],
    senses_whole: pd.DataFrame,
    senses_old: pd.DataFrame,
    senses_new: pd.DataFrame,
    method: str,
):
    answer = collections.OrderedDict()
    Results = collections.namedtuple(
        "Results", ["jsd", "cluster_to_freq1", "cluster_to_freq2"]
    )
    for word in predictions.keys():
        new_senses_whole = senses_whole[senses_whole["word"] == word]
        new_senses_old = senses_old[senses_old["word"] == word]
        new_senses_new = senses_new[senses_new["word"] == word]

        whole_data_id = dataframe_index_to_context_id(new_senses_whole)
        old_data_id = load_ids(new_senses_old)
        new_data_id = load_ids(new_senses_new)

        cluster_to_freq1 = {}
        cluster_to_freq2 = {}

        if method != "spectral_clustering":
            for id, cluster in predictions[word].items():
                if cluster not in cluster_to_freq1:
                    cluster_to_freq1[cluster] = 0
                if cluster not in cluster_to_freq2:
                    cluster_to_freq2[cluster] = 0

                if whole_data_id[id] in old_data_id:
                    cluster_to_freq1[cluster] += 1
                if whole_data_id[id] in new_data_id:
                    cluster_to_freq2[cluster] += 1

            c1 = np.array(list(cluster_to_freq1.values()))
            c2 = np.array(list(cluster_to_freq2.values()))
            val = distance.jensenshannon(c1, c2, base=2.0)
            answer[word] = Results(
                jsd=val,
                cluster_to_freq1=cluster_to_freq1,
                cluster_to_freq2=cluster_to_freq2,
            )

        else:
            validation_methods_answer = {}

            for no_clusters, mas_predictions in predictions[word].items():
                for id, cluster in mas_predictions.items():
                    if cluster not in cluster_to_freq1:
                        cluster_to_freq1[cluster] = 0
                    if cluster not in cluster_to_freq2:
                        cluster_to_freq2[cluster] = 0

                    if whole_data_id[id] in old_data_id:
                        cluster_to_freq1[cluster] += 1
                    if whole_data_id[id] in new_data_id:
                        cluster_to_freq2[cluster] += 1

                c1 = np.array(list(cluster_to_freq1.values()))
                c2 = np.array(list(cluster_to_freq2.values()))
                val = distance.jensenshannon(c1, c2, base=2.0)

                result = Results(
                    jsd=val,
                    cluster_to_freq1=cluster_to_freq1.copy(),
                    cluster_to_freq2=cluster_to_freq2.copy(),
                )
                validation_methods_answer[no_clusters] = result

                cluster_to_freq1.clear()
                cluster_to_freq2.clear()

            answer[word] = validation_methods_answer

    return answer


def extract_jsd_for_validation_methods(data: pd.DataFrame):
    rows = data.shape[0]
    silhouette = {
        F(data.iloc[[index]].parameters.item())["word"]: data.iloc[[index]][
            "number_cluster_selected_by_silhouette"
        ].item()
        for index in range(data.shape[0])
    }
    calinski = {
        F(data.iloc[[index]]["parameters"].item())["word"]: data.iloc[[index]][
            "number_cluster_selected_by_calinski"
        ].item()
        for index in range(data.shape[0])
    }
    eigengap = {
        F(data.iloc[[index]]["parameters"].item())["word"]: data.iloc[[index]][
            "number_cluster_selected_by_eigengap"
        ].item()
        for index in range(data.shape[0])
    }

    return silhouette, calinski, eigengap


def get_spearman(
    data: pd.DataFrame,
    senses_whole: pd.DataFrame,
    senses_old: pd.DataFrame,
    senses_new: pd.DataFrame,
    method: str,
    number_run: int,
) -> dict[int, float]:
    number_rows = data.shape[0]
    answer = {}
    logging.info("computing spearman by set of words [batches of 24 rows]")
    if method == "spectral_clustering":
        answer_silhouette = {}
        answer_calinski = {}
        answer_eigengap = {}

    for index in range(int(number_rows / NUMBER_OF_WORDS)):
        start = index * NUMBER_OF_WORDS
        end = start + NUMBER_OF_WORDS
        logging.info(f"loading batch-[{index}]")
        query_data = data.loc[start : end - 1, :]
        logging.info("loaded...")

        logging.info("loading cluster predictions...")
        predictions_per_word = load_predictions_set_words(query_data)
        logging.info("loaded...")

        logging.info("computing jsd and storing the cluster frequency")
        freq_clusters = compute_graded_lscd(
            predictions_per_word, senses_whole, senses_old, senses_new, method
        )
        logging.info("loaded and stored...")

        save_analysis_freq_clusters(
            query_data,
            freq_clusters,
            method,
            f"{OUTPUT}/output_freq_clusters/run_{number_run}/{method}.csv",
        )

        if method == "spectral_clustering":
            (
                silhouette,
                calinski,
                eigengap,
            ) = extract_jsd_for_validation_methods(query_data)

            logging.info("computing spearman for [silhouette]")
            answer_spearman_silhouette = compute_spearman(
                freq_clusters.copy(), silhouette
            )
            logging.info("computed...")

            logging.info("computing spearman for [calinski]")
            answer_spearman_calinski = compute_spearman(
                freq_clusters.copy(), calinski
            )
            logging.info("computed...")

            logging.info("computing spearman for [eigengap]")
            answer_spearman_eigengap = compute_spearman(
                freq_clusters.copy(), eigengap
            )
            logging.info("computed")

            answer_silhouette[index] = answer_spearman_silhouette
            answer_calinski[index] = answer_spearman_calinski
            answer_eigengap[index] = answer_spearman_eigengap
        else:
            logging.info("computing spearman")
            answer_spearman = compute_spearman(
                freq_clusters.copy(),
            )
            logging.info("computed...")

            answer[index] = answer_spearman

    if method == "spectral_clustering":
        return answer_silhouette, answer_calinski, answer_eigengap

    return answer


def compute_spearman_each_run():
    logging.info(
        "Starting to compute spearman and cluster frequencies for [cw, cc, wsbm, sc]"
    )
    path = INPUT_EXPERIMENTS
    results = {}

    logging.info("loading senses of the [full dataset]")
    senses_whole, scores_wholes = get_dwug_data_annotated_only(score_paths)
    logging.info("loaded...")

    logging.info("loading the senses of the [old dataset]")
    senses_old, scores_old = get_dwug_old_data_annotated_only(score_paths)
    logging.info("loaded...")

    logging.info("loading the senses of the [new dataset]")
    senses_new, scores_new = get_dwug_new_data_annotated_only(score_paths)
    logging.info("loaded...")

    for m in METHODS:
        results[m] = {}
        for index in range(1, 6):
            path_cache = path.format(
                run=index,
                method=m,
                folder="cache",
                dataset="dwug_data_annotated_only",
            )

            logging.info(f"Loading run-{index} of the experiments of {m}")
            cache_of_runs = pd.read_csv(path_cache)
            logging.info("loaded...")

            answer = get_spearman(
                cache_of_runs, senses_whole, senses_old, senses_new, m, index
            )
            if isinstance(answer, tuple):
                answer_silhouette = answer[0]
                answer_calinski = answer[1]
                answer_eigengap = answer[2]
                results[m][f"r{index}_silhouette"] = answer_silhouette
                results[m][f"r{index}_calinski"] = answer_calinski
                results[m][f"r{index}_eigengap"] = answer_eigengap
            else:
                results[m][f"r{index}"] = answer

        logging.info(f"saving spearman results for [{m}]")
        create_and_save_csv(results[m], m)
        logging.info("saved...")


def get_abs_difference(experiments: pd.DataFrame, field: str):
    assert isinstance(experiments, pd.DataFrame) is True

    gold_id_values = [
        len(set(F(item).values())) for item in experiments["gold_id"]
    ]

    answer = [
        abs(item[0] - item[1])
        for item in zip(gold_id_values, experiments[field].to_list())
    ]

    return answer


def compute_number_predicted_clusters():
    logging.info(
        "Computing the number of predicted clusters for [sc, cc, wsbm]"
    )
    new_values_per_dataset = {}
    for d in DATASETS:
        new_values_per_dataset[d] = {}
        logging.info(f"Dataset {d}")
        for m in ["chinese_whispers", "correlation_clustering", "wsbm"]:
            data_to_concatenate = []
            logging.info(f"Processing method [{m}]")

            for index in range(1, 6):
                logging.info(f"Loading run-[{index}] of the experiments")
                pd.read_csv(
                    INPUT_EXPERIMENTS.format(
                        run=index, method=m, folder="cache", dataset=d
                    )
                )

                experiment = pd.read_csv(
                    INPUT_EXPERIMENTS.format(
                        run=index, method=m, folder="cache", dataset=d
                    )
                )
                logging.info("loaded...")

                logging.info("Computing the number of clusters")
                experiment["number_clusters_predicted"] = experiment.apply(
                    lambda row: len(
                        set(list(F(row["predicted_clusters"]).values()))
                    ),
                    axis=1,
                )
                logging.info("computed")

                logging.info("Computing the [abs_difference]")
                new_experiment = experiment.assign(
                    abs_difference=lambda rows: get_abs_difference(
                        rows, "number_clusters_predicted"
                    )
                )
                logging.info("computed...")

                new_experiment.rename(
                    columns={
                        "number_clusters_predicted": f"number_clusters_predicted_r{index}",
                        "abs_difference": f"abs_difference_r{index}",
                    },
                    inplace=True,
                )
                data_to_concatenate.append(
                    new_experiment[
                        [
                            f"number_clusters_predicted_r{index}",
                            f"abs_difference_r{index}",
                        ]
                    ]
                )

            logging.info(
                f"concatenating the runs of the number of predicted clusters [{m}]"
            )
            new_values_per_dataset[d][m] = pd.concat(
                data_to_concatenate, axis=1
            )
            logging.info("concatenated...")

            save_file(
                new_values_per_dataset[d][m],
                f"{OUTPUT_TO_AVG_PREDICTED_CLUSTERS}/{d}/{m}.csv",
            )


def compute_number_predicted_clusters_for_sc():
    logging.info("Computing the number of predicted clusters for [sc]")
    new_values_per_dataset = {}
    for d in DATASETS:
        logging.info(f"Processing the dataset [{d}]")
        results_to_concatenate = []

        for index in range(1, 6):
            logging.info(f"loading the run-[{index}] of the experiments")
            path_to_experiments = INPUT_EXPERIMENTS.format(
                run=index,
                method="spectral_clustering",
                folder="cache",
                dataset=d,
            )
            experiments_result = pd.read_csv(path_to_experiments)
            logging.info("loaded...")

            logging.info(
                "computing the [abs_difference] for [silhouette, calinski, eigengap]"
            )
            new_experiments_results = experiments_result.assign(
                abs_difference_silhouette=lambda rows: get_abs_difference(
                    rows, "number_cluster_selected_by_silhouette"
                ),
                abs_difference_calinski=lambda rows: get_abs_difference(
                    rows, "number_cluster_selected_by_calinski"
                ),
                abs_difference_eigengap=lambda rows: get_abs_difference(
                    rows, "number_cluster_selected_by_eigengap"
                ),
            )
            logging.info("computed...")

            logging.info("renaming columns")
            new_experiments_results.rename(
                columns={
                    "number_cluster_selected_by_silhouette": f"number_cluster_selected_by_silhouette_r{index}",
                    "number_cluster_selected_by_calinski": f"number_cluster_selected_by_calinski_r{index}",
                    "number_cluster_selected_by_eigengap": f"number_cluster_selected_by_eigengap_r{index}",
                    "abs_difference_silhouette": f"abs_difference_silhouette_r{index}",
                    "abs_difference_calinski": f"abs_difference_calinski_r{index}",
                    "abs_difference_eigengap": f"abs_difference_eigengap_r{index}",
                },
                inplace=True,
            )
            logging.info("renamed...")

            logging.info("selecting a subset of columns")
            new_experiments_results = new_experiments_results[
                [
                    # f"number_cluster_selected_by_silhouette_r{index}",
                    # f"number_cluster_selected_by_calinski_r{index}",
                    # f"number_cluster_selected_by_eigengap_r{index}",
                    f"abs_difference_silhouette_r{index}",
                    f"abs_difference_calinski_r{index}",
                    f"abs_difference_eigengap_r{index}",
                ]
            ]
            logging.info("selected...")
            results_to_concatenate.append(new_experiments_results)

        logging.info("concatenating the experiments")
        data = pd.concat(results_to_concatenate, axis=1)
        logging.info("concatenated...")

        new_values_per_dataset[d] = data
        save_file(
            new_values_per_dataset[d],
            f"{OUTPUT_TO_AVG_PREDICTED_CLUSTERS}/{d}/spectral_clustering.csv",
        )


def concatenate_all_runs_experiments(
    dataset="dwug_data_annotated_only",
) -> dict[str, pd.DataFrame]:
    logging.info(
        f"""concatenating the 5 runs of the experiments for every method
            in the {dataset} dataset
        """
    )
    config_method = load_config_file("config/methods.json")
    data_per_method: dict[str, pd.DataFrame] = {}

    for m in METHODS:
        logging.info(f"Processing method [{m}]")

        data_to_concatenate = []
        columns_to_rename = {}
        if m not in config_method:
            fields = config_method["non_spectral_clustering"][
                "concatenate_all_runs_experiments"
            ]
        else:
            fields = config_method["spectral_clustering"][
                "concatenate_all_runs_experiments"
            ]

        if dataset != "dwug_data_annotated_only":
            fields = fields["no_full_dataset"]

        for index in range(1, 6):
            logging.info(f"processing run-[{index}]")
            path = INPUT_EXPERIMENTS.format(
                run=index,
                method=m,
                folder="cache",
                dataset=dataset,
            )

            logging.info("loading data")
            data = pd.read_csv(path)
            logging.info("loaded...")

            if "drop" in fields:
                data.drop(columns=fields["drop"], inplace=True)
            if "rename_before" in fields:
                data.rename(columns=fields["rename_before"], inplace=True)

            for field in fields["fields"]:
                columns_to_rename[field] = f"{field}_r{index}"

            data.rename(columns=columns_to_rename, inplace=True)
            data_to_concatenate.append(data)

        logging.info("concatenating data")
        data_per_method[m] = pd.concat(data_to_concatenate, axis=1)
        logging.info("concatenated...")

        if "drop_after" in fields:
            data_per_method[m].drop(columns=fields["drop_after"], inplace=True)

    return data_per_method


def concatenate_clusters_jsd_files():
    logging.info("concatenating clusters jsd files")
    CONFIG = load_config_file("config/methods.json")
    data_to_concatenate = []
    data_per_method = {}
    for m in METHODS:
        logging.info(f"processing method [{m}]")

        for index in range(1, 6):
            rename_columns = {}
            config = (
                CONFIG["non_spectral_clustering"][
                    "concatenate_clusters_jsd_files"
                ]
                if m not in CONFIG
                else CONFIG[m]["concatenate_clusters_jsd_files"]
            )

            if m != "spectral_clustering":
                for field in config["rename_columns"]:
                    rename_columns[field] = f"{field}_r{index}"
            else:
                for cluster_validation in [
                    "silhouette",
                    "calinski",
                    "eigengap",
                ]:
                    for field in config["rename_columns"]:
                        rename_columns[
                            f"{field}_{cluster_validation}"
                        ] = f"{field}_{cluster_validation}_r{index}"

            path = INPUT_FREQ_CLUSTERS.format(run=index, method=m)

            logging.info("loading file...")
            data = pd.read_csv(path)
            logging.info("loaded...")

            logging.info(f"renaming fields for the run-[{index}]")
            data.rename(
                columns=rename_columns,
                inplace=True,
            )
            logging.info("renamed...")

            data.drop(columns=config["drop"], inplace=True)
            data_to_concatenate.append(data)

        data_per_method[m] = pd.concat(data_to_concatenate, axis=1)
        data_to_concatenate.clear()

    return data_per_method


def load_predicted_clusters(dataset="dwug_data_annotated_only"):
    logging.info("loading the predicted cluster files")
    data_per_method = {}
    for m in METHODS:
        path = INPUT_TO_AVG_PREDICTED_CLUSTERS.format(
            dataset=dataset, method=m
        )
        data = pd.read_csv(path)
        data_per_method[m] = data
    logging.info("loaded...")

    return data_per_method


def build_big_file_results():
    logging.info("build big file results")
    predicted_clusters = load_predicted_clusters()
    all_experiments = concatenate_all_runs_experiments()
    predicted_jsd = concatenate_clusters_jsd_files()
    data_per_method = {}

    for m in METHODS:
        logging.info(f"processing method [{m}]")

        data_per_method[m] = pd.concat(
            [all_experiments[m], predicted_clusters[m], predicted_jsd[m]],
            axis=1,
        )

        save_file(
            data_per_method[m],
            f"{OUTPUT}/results_whole_dataset/{m}.csv",
        )


def build_big_file_for_non_full_dataset(dataset, portion):
    logging.info(f"build bid file results for {dataset} dataset")
    predicted_clusters = load_predicted_clusters(dataset)
    all_experiments = concatenate_all_runs_experiments(dataset)
    data_per_method = {}

    for m in METHODS:
        logging.info(f"processing method [{m}]")

        data_per_method[m] = pd.concat(
            [all_experiments[m], predicted_clusters[m]],
            axis=1,
        )

        save_file(
            data_per_method[m],
            f"{OUTPUT}/results_{portion}_dataset/{m}.csv",
        )


def clear_records():
    data = pd.DataFrame(
        {
            "jsd": [],
            "freq_clusters_1": [],
            "freq_clusters_2": [],
            "parameters": [],
        }
    )

    for i in range(1, 6):
        for m in METHODS:
            data.to_csv(
                f"output_freq_clusters/run_{i}/{m}.csv",
                index=False,
                header=True,
            )


if __name__ == "__main__":
    start_time = datetime.now()
    compute_spearman_each_run()
    compute_number_predicted_clusters()
    compute_number_predicted_clusters_for_sc()
    build_big_file_results()
    build_big_file_for_non_full_dataset("dwug_old_data_annotated_only", "old")
    build_big_file_for_non_full_dataset("dwug_new_data_annotated_only", "new")
    print(f"Elapsed time: {datetime.now() - start_time}")
