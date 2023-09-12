from ast import literal_eval as F
import json
from typing import TextIO, Union
import os
import sys

import pandas as pd
from scipy.stats import spearmanr


CLUSTER_VALIDATION_METHODS = ["silhouette", "calinski", "eigengap"]
NO_EXPERIMENTS = 5
INPUT_EXPERIMENTS = "../../outputs/"


def splitted_data(data: pd.DataFrame, number_of_words: int):
    n = data.shape[0]
    for index in range(int(n / number_of_words)):
        start = index * number_of_words
        end = start + number_of_words
        yield data.loc[start : end - 1, :]


def load_config_file():
    with open("config/5-fold-config.json", "r") as f_in:
        data = json.load(f_in)

    return data


CONFIG = load_config_file()


def load_gold_data_semeval():
    data = pd.read_csv(
        "../dwug_de/misc/dwug_de_sense/stats/maj_3/stats_groupings.csv",
        sep="\t",
    )
    return data


def load_gold_change_graded_semeval():
    data = load_gold_data_semeval()
    gold_change_graded = (
        data[["lemma", "change_graded"]]
        .set_index("lemma")["change_graded"]
        .to_dict()
    )

    return gold_change_graded


def create_n_consecutive_fields(field: str, number: int):
    fields = []
    for index in range(1, number + 1):
        fields.append(f"{field}{index}")

    return fields


def select_words(list_of_words, index_of_words):
    selected_words = []
    for index in index_of_words:
        selected_words.append(list_of_words[index])

    return selected_words


def prepare_data_for_spr(
    data, list_of_words, index_of_words, sc_with_field: dict[str, str] = None
):
    selected_words = select_words(list_of_words, index_of_words)
    gold_data = load_gold_change_graded_semeval()
    vector1 = [gold_data[word] for word in selected_words]
    if sc_with_field is None:
        vector2 = [round(float(val), 2) for val in data["avg_jsd"].to_list()]
    else:
        vector2 = [
            round(float(val), 2)
            for val in data[f"avg_jsd_{sc_with_field['field']}"].to_list()
        ]

    return vector1, vector2


def get_avg_ari_subset_words_sc(data, config):
    subset_data = data.copy()
    try:
        no_experiments = int(os.environ["NUMBER_EXPERIMENTS"])
    except Exception:
        no_experiments = NO_EXPERIMENTS

    for validation_method in CLUSTER_VALIDATION_METHODS:
        subset_data[f"avg_ari_{validation_method}"] = subset_data[
            create_n_consecutive_fields(
                config["spectral_clustering"][f"avg_ari_{validation_method}"][
                    "field"
                ],
                no_experiments,
            )
        ].mean(axis=1)
        subset_data[f"avg_ari_{validation_method}_old"] = subset_data[
            create_n_consecutive_fields(
                config["spectral_clustering"][
                    f"avg_ari_{validation_method}_old"
                ]["field"],
                no_experiments,
            )
        ].mean(axis=1)
        subset_data[f"avg_ari_{validation_method}_new"] = subset_data[
            create_n_consecutive_fields(
                config["spectral_clustering"][
                    f"avg_ari_{validation_method}_new"
                ]["field"],
                no_experiments,
            )
        ].mean(axis=1)

    return subset_data


def get_avg_jsd_subset_words_sc(data, config):
    subset_data = data.copy()
    try:
        no_experiments = int(os.environ["NUMBER_EXPERIMENTS"])
    except Exception:
        no_experiments = NO_EXPERIMENTS

    for validation_method in CLUSTER_VALIDATION_METHODS:
        subset_data[f"avg_jsd_{validation_method}"] = subset_data[
            create_n_consecutive_fields(
                config["spectral_clustering"][f"avg_jsd_{validation_method}"][
                    "field"
                ],
                no_experiments,
            )
        ].mean(axis=1)

    return subset_data


def get_avg_number_clusters_predicted(data, method):
    config = CONFIG["5-fold-cv"]
    config = config[
        "non_spectral_clustering" if method not in config else method
    ]["include_extra_information"]
    fields = config.keys()

    try:
        no_experiments = int(os.environ["NUMBER_EXPERIMENTS"])
    except Exception:
        no_experiments = NO_EXPERIMENTS

    for field in fields:
        if field.startswith("avg"):
            data[field] = data[
                create_n_consecutive_fields(
                    config[field]["field"], no_experiments
                )
            ].mean(axis=1)

    return data


def get_extra_fields_to_report(data, method, parameters):
    config = CONFIG["5-fold-cv"]
    config = config[
        "non_spectral_clustering" if method not in config else method
    ]["include_extra_information"]
    fields = config.keys()

    for field in fields:
        if field.startswith("avg"):
            parameters[field] = data[field].mean(axis=0)

    return parameters


def eval_spectral_clustering_method_changing_parameters(
    data,
    test_set,
    best_configuration_for_ari_training,
    best_configuration_for_jsd_training_set,
    target_words,
):
    results_ari = {}
    results_spr_lscd = {}

    for validation_method in CLUSTER_VALIDATION_METHODS:
        start = (
            best_configuration_for_ari_training[f"index_{validation_method}"]
            * 24
        )
        end = start + 24
        subset_test_for_spr_lscd = (
            data.loc[start : end - 1, :].iloc[test_set].copy()
        )
        gold_change_graded, jsd = prepare_data_for_spr(
            subset_test_for_spr_lscd,
            target_words,
            test_set,
            {"field": validation_method},
        )
        spr, _ = spearmanr(gold_change_graded, jsd)
        results_spr_lscd[f"spr_lscd_{validation_method}"] = spr

        start = (
            best_configuration_for_jsd_training_set[
                f"index_{validation_method}"
            ]
            * 24
        )
        end = start + end
        subset_test_for_ari = (
            data.loc[start : end - 1, :].iloc[test_set].copy()
        )
        results_ari[f"avg_ari_{validation_method}"] = subset_test_for_ari[
            f"avg_ari_{validation_method}"
        ].mean(axis=0)

        results_ari[f"avg_ari_{validation_method}_old"] = 0.0
        results_ari[f"avg_ari_{validation_method}_new"] = 0.0

    return results_ari, results_spr_lscd


def train_spectral_clustering_method(
    data: pd.DataFrame,
    config,
    training_set: list[int],
    list_of_words: list[str],
) -> tuple[dict, dict]:
    best_combination_for_ari = {
        "parameters_silhouette": {},
        "parameters_calinski": {},
        "parameters_eigengap": {},
        "parameters_silhouette_old": {},
        "parameters_calinski_old": {},
        "parameters_eigengap_old": {},
        "parameters_silhouette_new": {},
        "parameters_calinski_new": {},
        "parameters_eigengap_new": {},
        "ari_silhouette": 0.0,
        "ari_calinski": 0.0,
        "ari_eigengap": 0.0,
        "ari_silhouette_old": 0.0,
        "ari_calinski_old": 0.0,
        "ari_eigengap_old": 0.0,
        "ari_silhouette_new": 0.0,
        "ari_calinski_new": 0.0,
        "ari_eigengap_new": 0.0,
        "index_silhouette": -1,
        "index_calinski": -1,
        "index_eigengap": -1,
        "index_silhouette_old": -1,
        "index_calinski_old": -1,
        "index_eigengap_old": -1,
        "index_silhouette_new": -1,
        "index_calinski_new": -1,
        "index_eigengap_new": -1,
    }
    best_combination_for_spr_lscd = {
        "parameters_silhouette": [],
        "parameters_calinski": [],
        "parameters_eigengap": [],
        "spr_lscd_silhouette": 0.0,
        "spr_lscd_calinski": 0.0,
        "spr_lscd_eigengap": 0.0,
        "index_silhouette": -1,
        "index_calinski": -1,
        "index_eigengap": -1,
    }
    number_of_subset_of_words = -1
    data = get_avg_ari_subset_words_sc(data, config)
    data = get_avg_jsd_subset_words_sc(data, config)

    for set_of_words in splitted_data(data, 24):
        number_of_subset_of_words += 1
        subset_training = set_of_words.iloc[training_set].copy()
        parameters = F(subset_training["parameters_r1"].to_list()[0])
        parameters.pop("word", None)
        parameters["score_path"] = subset_training["score_path_r1"].to_list()[
            0
        ]
        extra_fields_to_report = get_extra_fields_to_report(
            subset_training, "spectral_clustering", parameters.copy()
        )
        parameters.pop("word_level_threshold", None)

        for method in CLUSTER_VALIDATION_METHODS:
            avg = subset_training[f"avg_ari_{method}"].mean(axis=0)
            if avg > best_combination_for_ari[f"ari_{method}"]:
                best_combination_for_ari[f"ari_{method}"] = avg
                best_combination_for_ari[
                    f"index_{method}"
                ] = number_of_subset_of_words
                parameters[
                    f"avg_number_cluster_selected_by_{method}"
                ] = extra_fields_to_report[
                    f"avg_number_cluster_selected_by_{method}"
                ]
                parameters[
                    f"avg_abs_difference_{method}"
                ] = extra_fields_to_report[f"avg_abs_difference_{method}"]
                best_combination_for_ari[
                    f"parameters_{method}"
                ] = parameters.copy()
                parameters.pop(
                    f"avg_number_cluster_selected_by_{method}", None
                )
                parameters.pop(f"avg_abs_difference_{method}", None)

            avg = subset_training[f"avg_ari_{method}_old"].mean(axis=0)
            if avg > best_combination_for_ari[f"ari_{method}_old"]:
                best_combination_for_ari[f"ari_{method}_old"] = avg
                best_combination_for_ari[
                    f"index_{method}_old"
                ] = number_of_subset_of_words
                best_combination_for_ari[
                    f"parameters_{method}_old"
                ] = parameters

            avg = subset_training[f"avg_ari_{method}_new"].mean(axis=0)
            if avg > best_combination_for_ari[f"ari_{method}_new"]:
                best_combination_for_ari[f"ari_{method}_new"] = avg
                best_combination_for_ari[
                    f"index_{method}_new"
                ] = number_of_subset_of_words
                best_combination_for_ari[
                    f"parameters_{method}_new"
                ] = parameters

            gold_data_graded_change, avg_jsd = prepare_data_for_spr(
                subset_training, list_of_words, training_set, {"field": method}
            )
            spr, _ = spearmanr(gold_data_graded_change, avg_jsd)
            if spr > best_combination_for_spr_lscd[f"spr_lscd_{method}"]:
                best_combination_for_spr_lscd[f"spr_lscd_{method}"] = spr
                best_combination_for_spr_lscd[
                    f"index_{method}"
                ] = number_of_subset_of_words
                best_combination_for_spr_lscd[
                    f"parameters_{method}"
                ] = parameters

    return best_combination_for_ari, best_combination_for_spr_lscd


def eval_spectral_clustering(
    data: pd.DataFrame,
    test_set: list[int],
    config,
    best_configuration_for_ari_training_set,
    best_configuration_for_jsd_training_set,
    target_words: list[str],
    exchange_optimized_parameters: bool = False,
) -> tuple[dict, float]:
    results_ari = {}
    results_spr_lscd = {}

    if exchange_optimized_parameters is True:
        return eval_spectral_clustering_method_changing_parameters(
            data,
            test_set,
            best_configuration_for_ari_training_set,
            best_configuration_for_jsd_training_set,
            target_words,
        )

    for validation_method in CLUSTER_VALIDATION_METHODS:
        start = (
            best_configuration_for_jsd_training_set[
                f"index_{validation_method}"
            ]
            * 24
        )
        end = start + 24
        subset_test_for_spr_lscd = (
            data.loc[start : end - 1, :].iloc[test_set].copy()
        )
        gold_change_graded, jsd = prepare_data_for_spr(
            subset_test_for_spr_lscd,
            target_words,
            test_set,
            {"field": validation_method},
        )
        spr, _ = spearmanr(gold_change_graded, jsd)
        results_spr_lscd[f"spr_lscd_{validation_method}"] = spr

        start = (
            best_configuration_for_ari_training_set[
                f"index_{validation_method}"
            ]
            * 24
        )
        end = start + end
        subset_test_for_ari = (
            data.loc[start : end - 1, :].iloc[test_set].copy()
        )
        results_ari[f"avg_ari_{validation_method}"] = subset_test_for_ari[
            f"avg_ari_{validation_method}"
        ].mean(axis=0)

        start = (
            best_configuration_for_ari_training_set[
                f"index_{validation_method}_old"
            ]
            * 24
        )
        end = start + end
        subset_test_for_ari = (
            data.loc[start : end - 1, :].iloc[test_set].copy()
        )
        results_ari[f"avg_ari_{validation_method}_old"] = subset_test_for_ari[
            f"avg_ari_{validation_method}_old"
        ].mean(axis=0)

        start = (
            best_configuration_for_ari_training_set[
                f"index_{validation_method}_new"
            ]
            * 24
        )
        end = start + end
        subset_test_for_ari = (
            data.loc[start : end - 1, :].iloc[test_set].copy()
        )
        results_ari[f"avg_ari_{validation_method}_new"] = subset_test_for_ari[
            f"avg_ari_{validation_method}_new"
        ].mean(axis=0)

    return results_ari, results_spr_lscd


def train(
    method: str, training_set: list[int], target_words: list[str]
) -> tuple[dict, dict]:
    config = CONFIG["5-fold-cv"]
    data = pd.read_csv(
        f"{INPUT_EXPERIMENTS}/results_whole_dataset/{method}.csv"
    )
    data = get_avg_number_clusters_predicted(data, method)
    if method == "spectral_clustering":
        return train_spectral_clustering_method(
            data, config, training_set, target_words
        )

    best_combination_for_ari = {
        "parameters": {},
        "ari": 0.0,
        "index": -1,
    }
    best_combination_for_spr_lscd = {
        "parameters": {},
        "spr_lscd": 0.0,
        "index": -1,
    }
    number_of_subset_of_words = -1
    try:
        number_of_experiments = int(os.environ["NUMBER_EXPERIMENTS"])
    except Exception:
        number_of_experiments = NO_EXPERIMENTS

    for set_of_words in splitted_data(data, 24):
        number_of_subset_of_words += 1
        subset_training = set_of_words.iloc[training_set].copy()
        subset_training["avg_ari"] = subset_training[
            create_n_consecutive_fields(
                config["test"]["avg_ari"]["field"], number_of_experiments
            )
        ].mean(axis=1)
        avg_ari_training_set = subset_training["avg_ari"].mean(axis=0)

        subset_training["avg_jsd"] = subset_training[
            create_n_consecutive_fields(
                config["test"]["avg_jsd"]["field"], number_of_experiments
            )
        ].mean(axis=1)

        gold_change_graded, jsd = prepare_data_for_spr(
            subset_training, target_words, training_set
        )
        spr, _ = spearmanr(gold_change_graded, jsd)

        parameters = F(subset_training["parameters_r1"].to_list()[0])
        parameters.pop("word", None)
        try:
            parameters["hyperparameter"] = subset_training[
                "hyperparameter"
            ].to_list()[0]
        except Exception:
            pass

        parameters = get_extra_fields_to_report(
            subset_training, method, parameters.copy()
        )

        if avg_ari_training_set > best_combination_for_ari["ari"]:
            best_combination_for_ari["ari"] = avg_ari_training_set
            best_combination_for_ari["parameters"] = parameters
            best_combination_for_ari["index"] = number_of_subset_of_words

        if spr > best_combination_for_spr_lscd["spr_lscd"]:
            best_combination_for_spr_lscd["spr_lscd"] = spr
            best_combination_for_spr_lscd["parameters"] = parameters
            best_combination_for_spr_lscd["index"] = number_of_subset_of_words

    return best_combination_for_ari, best_combination_for_spr_lscd


def eval(
    method: str,
    test_set: list[int],
    best_configuration_for_ari_training_set,
    best_configuration_for_jsd_training_set,
    target_words: list[str],
    exchange_optimized_parameters: bool = False,
) -> tuple[float, float]:
    config = CONFIG["5-fold-cv"]
    data = pd.read_csv(
        f"{INPUT_EXPERIMENTS}/results_whole_dataset/{method}.csv"
    )

    try:
        no_experiments = int(os.environ["NUMBER_EXPERIMENTS"])
    except Exception:
        no_experiments = NO_EXPERIMENTS

    if method == "spectral_clustering":
        data = get_avg_ari_subset_words_sc(data, config)
        data = get_avg_jsd_subset_words_sc(data, config)
        return eval_spectral_clustering(
            data,
            test_set,
            config,
            best_configuration_for_ari_training_set,
            best_configuration_for_jsd_training_set,
            target_words,
            exchange_optimized_parameters,
        )

    if exchange_optimized_parameters is True:
        start = best_configuration_for_jsd_training_set["index"] * 24
    else:
        start = best_configuration_for_ari_training_set["index"] * 24

    end = start + 24
    subset_test_for_ari = data.loc[start : end - 1, :].iloc[test_set].copy()
    subset_test_for_ari["avg_ari"] = subset_test_for_ari[
        create_n_consecutive_fields(
            config["test"]["avg_ari"]["field"], no_experiments
        )
    ].mean(axis=1)

    if exchange_optimized_parameters is True:
        start = best_configuration_for_ari_training_set["index"] * 24
    else:
        start = best_configuration_for_jsd_training_set["index"] * 24

    end = start + 24
    subset_test_for_jsd = data.loc[start : end - 1, :].iloc[test_set].copy()
    subset_test_for_jsd["avg_jsd"] = subset_test_for_jsd[
        create_n_consecutive_fields(
            config["test"]["avg_jsd"]["field"], no_experiments
        )
    ].mean(axis=1)

    gold_change_graded, jsd = prepare_data_for_spr(
        subset_test_for_jsd, target_words, test_set
    )

    spr, _ = spearmanr(gold_change_graded, jsd)
    ari = subset_test_for_ari["avg_ari"].mean(axis=0)

    return ari, spr


def get_fields_to_report_for_spectral_clustering():
    config = load_config_file()["5-fold-cv"]
    fields = config["spectral_clustering"].keys()
    results = {}

    for field in fields:
        results[field] = 0.0

    return results


def calculate_results_for_spectral_clustering_method(
    avg_ari_for_spectral_clustering: dict, result_avg_ari: dict
):
    config = CONFIG["5-fold-cv"]
    fields = config["spectral_clustering"][
        "calculate_results_for_spectral_clustering_method"
    ]
    for field in fields:
        avg_ari_for_spectral_clustering[field] += result_avg_ari[field]


def calculate_average(
    results_per_method: dict, methods: list[str], k_fold: int
):
    for m in methods:
        if m != "spectral_clustering":
            results_per_method[m]["ari"] = float(
                results_per_method[m]["ari"] / k_fold
            )
            results_per_method[m]["spr_lscd"] = float(
                results_per_method[m]["spr_lscd"] / k_fold
            )
        else:
            config = CONFIG["5-fold-cv"]
            fields = config["spectral_clustering"][
                "calculate_results_for_spectral_clustering_method"
            ]
            for field in fields:
                results_per_method[m][field] = float(
                    results_per_method[m][field] / k_fold
                )


def aux(results_per_method, key, f_out: TextIO):
    for index, parameter in enumerate(results_per_method):
        f_out.write(f"    Iter-{index+1}: {parameter[key]}\n")
    f_out.write("\n")


def save_parameters_for_spectral_clustering(
    results_per_method: dict, metric: str, f_out: TextIO
):
    assert metric == "ari" or metric == "spr_lscd", "expected [ari | spr_lscd]"

    if metric == "ari":
        fields = [
            "silhouette",
            "calinski",
            "eigengap",
            "silhouette_old",
            "calinski_old",
            "eigengap_old",
            "silhouette_new",
            "calinski_new",
            "eigengap_new",
        ]
    else:
        fields = [
            "silhouette",
            "calinski",
            "eigengap",
        ]

    for m in fields:
        f_out.write(f"  {m}\n")
        aux(
            results_per_method["spectral_clustering"][metric],
            f"parameters_{m}",
            f_out,
        )


def save_parameters_per_method(
    parameters_per_method: dict, methods: list[str]
):
    with open(f"parameter-results-per-method.txt", "w") as f_out:
        for m in methods:
            f_out.write(f"Method: {m}\n")
            f_out.write("  ARI:\n")

            if m != "spectral_clustering":
                for index, parameters in enumerate(
                    parameters_per_method[m]["ari"]
                ):
                    if m != "spectral_clustering":
                        f_out.write(
                            f"    Iter-{index+1}: {parameters['parameters']}\n"
                        )
            else:
                save_parameters_for_spectral_clustering(
                    parameters_per_method, "ari", f_out
                )

            f_out.write(" Spr_LSCD:\n")
            if m != "spectral_clustering":
                for index, parameters in enumerate(
                    parameters_per_method[m]["spr_lscd"]
                ):
                    f_out.write(
                        f"    Iter-{index+1}: {parameters['parameters']}\n"
                    )

                f_out.write("\n")
            else:
                save_parameters_for_spectral_clustering(
                    parameters_per_method, "spr_lscd", f_out
                )


def print_results(
    no_fold: int,
    dev_ari: Union[dict, float],
    test_ari: Union[dict, float],
    dev_spr_lscd: Union[dict, float],
    test_spr_lscd: Union[dict, float],
):
    print(f"  Fold-{no_fold}:")
    if isinstance(dev_ari, dict) and isinstance(dev_spr_lscd, dict):
        print(f"    dev-ARI_Silhouette: {dev_ari['ari_silhouette']}", end=" ")
        print(f"test-ARI_Silhouette: {test_ari['avg_ari_silhouette']}")
        print(
            f"    dev-Spr_LSCD_Silhouette: {dev_spr_lscd['spr_lscd_silhouette']}",
            end=" ",
        )
        print(
            f"test-Spr_LSCD_Silhouette: {test_spr_lscd['spr_lscd_silhouette']}"
        )
        print()
        print(f"    dev-ARI-Calinski: {dev_ari['ari_calinski']}", end=" ")
        print(f"test-ARI_Calinski: {test_ari['avg_ari_calinski']}")
        print(
            f"    dev-Spr_LSCD_Calinski: {dev_spr_lscd['spr_lscd_calinski']}",
            end=" ",
        )
        print(f"test-Spr_LSCD_Calinski: {test_spr_lscd['spr_lscd_calinski']}")
        print()
        print(f"    dev-ARI-Eigengap: {dev_ari['ari_eigengap']}", end=" ")
        print(f"test-ARI_Eigengap: {test_ari['avg_ari_eigengap']}")
        print(
            f"    dev-Spr_LSCD_Eigengap: {dev_spr_lscd['spr_lscd_eigengap']}",
            end=" ",
        )
        print(f"test-Spr_LSCD_Eigengap: {test_spr_lscd['spr_lscd_eigengap']}")

    else:
        print(f"    dev-ARI: {dev_ari}", end=" ")
        print(f"test-ARI: {test_ari}")
        print(f"    dev-Spr_LSCD: {dev_spr_lscd}", end=" ")
        print(f"test-Spr_LSCD: {test_spr_lscd}")
        print()
