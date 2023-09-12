import os
import re
import typing as tp
from collections import defaultdict, OrderedDict

import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from lexsubgen.metrics import wsi_metrics

from multilang_wsi_evaluation.utils import WSIDatasetPart


class SemevalMetricsWrapper:
    def __init__(self, source_semeval_metrics_func):
        self._source_semeval_metrics_func = source_semeval_metrics_func

    def compute_semeval_metrics(
            self,
            gold_labels: tp.List,
            pred_labels: tp.List,
            group_by: tp.List[str],
            context_ids: tp.List[str],
            gold_labels_path: os.PathLike = None,
    ) -> tp.Dict[str, tp.List[float]]:
        try:
            metrics = self._source_semeval_metrics_func(
                gold_labels=gold_labels, pred_labels=pred_labels, group_by=group_by,
                context_ids=context_ids, gold_labels_path=gold_labels_path
            )
        except Exception:
            raise RuntimeError('Invalid dataset or preds file for SemEval evaluation scripts')

        # Since evaluation scripts explicitly add 'all' word with aggregated metrics
        assert 'all' not in set(group_by), "By now evaluation script does not support 'all' as a target word..."
        assert set(metrics.keys()) - {'all'} == set(group_by), \
            f"Not all of the words were processed by the SemEval scripts:"  \
            f" {(set(metrics.keys()) - {'all'}) ^ set(group_by)}"
        return metrics


os.environ['JAVA_TOOL_OPTIONS'] = '-Dfile.encoding=UTF8'  # Set encoding for java eval scripts
wsi_metrics.compute_semeval_2013_metrics = \
    SemevalMetricsWrapper(wsi_metrics.compute_semeval_2013_metrics).compute_semeval_metrics
wsi_metrics.compute_semeval_2010_metrics = \
    SemevalMetricsWrapper(wsi_metrics.compute_semeval_2010_metrics).compute_semeval_metrics


def compute_unsup_metrics(word, word_dists_matrix, features_matrix, clusters):
    try:
        sil_score = silhouette_score(word_dists_matrix, clusters, metric='precomputed')
    except Exception as e:
        print(f'Metrics exception for word {word} in silhouette_score:', e)
        # TODO : check doc or score = -2. Effects
        sil_score = -2
    cal_har_score = calinski_harabasz_score(features_matrix, clusters)
    return {'silhouette': sil_score, 'calinski_harabasz': cal_har_score}


def compute_words_metrics(part: WSIDatasetPart, word_to_preds: tp.Dict[str, tp.Dict[tp.Any, tp.Any]]) -> pd.DataFrame:
    id_to_pred: tp.Dict[tp.Any, tp.Any] = {}
    for word_preds in word_to_preds.values():
        id_to_pred.update(word_preds)
    _, words_metrics = compute_wsi_metrics(part, id_to_pred)
    return words_metrics


def compute_mean_metrics(part: WSIDatasetPart, id_to_pred: tp.Dict[tp.Any, tp.Any]) -> tp.Dict[str, float]:
    mean_metrics, _ = compute_wsi_metrics(part, id_to_pred)
    return mean_metrics


def merge_word_metrics(word_metrics: tp.Dict[str, tp.Dict[str, float]], batch_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = list(list(word_metrics.values())[0].keys())
    for metric in metrics:
        metric_dict = {key: value[metric] for key, value in word_metrics.items()}
        batch_metrics[metric] = batch_metrics['word'].map(metric_dict)
    return batch_metrics


def aggregate_metrics(word_metrics: tp.Dict[tp.Any, pd.DataFrame], unsup_metrics) -> pd.DataFrame:
    all_metrics = pd.DataFrame()
    for hypers, temp_df in word_metrics.items():
        temp_df['hypers'] = [hypers] * len(temp_df)
        all_metrics = pd.concat([all_metrics, temp_df], ignore_index=True)
    dataset_max_metrics = all_metrics.groupby(by='word').max().mean()
    dataset_max_metrics.index = ['max_' + i for i in list(dataset_max_metrics.index)]
    dataset_fh_max_metrics = all_metrics.groupby(by='hypers').mean().max()
    dataset_fh_max_metrics.index = ['fh_max_' + i for i in list(dataset_fh_max_metrics.index)]

    #dataset_sil_metrics2 = #TODO: from max_ari.py

    dataset_unsup_metrics = defaultdict(pd.DataFrame)
    unsup_metrics_df = pd.DataFrame()
    mas_metrics, list_word, list_params = defaultdict(list), [], []
    for key, value in unsup_metrics.items():
        for key1, value1 in value.items():
            list_word.append(key1)
            list_params.append(key)
            for m, score in value1.items():
                mas_metrics[m].append(score)
    unsup_metrics_df['word'] = list_word
    unsup_metrics_df['hypers'] = list_params
    for m, l_score in mas_metrics.items():
        unsup_metrics_df[m] = l_score
    metrics = list(unsup_metrics_df.columns)[2:]
    all_metrics = all_metrics.merge(unsup_metrics_df, how='inner', on=['hypers', 'word'])
    for metric in metrics:
        dataset_unsup_metrics[metric] = all_metrics.groupby(by='word').agg({metric: max}).reset_index()
        dataset_unsup_metrics[metric] = dataset_unsup_metrics[metric].merge(all_metrics, how='inner', on=[metric, 'word'])
        dataset_unsup_metrics[metric] = dataset_unsup_metrics[metric].groupby(by='word').first().mean()
        dataset_unsup_metrics[metric].index = [f'{metric}_' + i for i in list(dataset_unsup_metrics[metric].index)]
    concat_dataset_metrics = [data.reset_index() for key, data in dataset_unsup_metrics.items()]
    concat_dataset_metrics.extend([dataset_max_metrics.reset_index(), dataset_fh_max_metrics.reset_index()])
    return pd.concat(concat_dataset_metrics)


def compute_wsi_metrics(part: WSIDatasetPart, id_to_pred: tp.Dict[tp.Any, tp.Any]) \
        -> tp.Tuple[tp.Dict[str, float], pd.DataFrame]:
    gold_dataset = part.dataset_df
    current_dataset = pd.DataFrame({'context_id': id_to_pred.keys(), 'predict_sense_id': id_to_pred.values()})

    assert len(current_dataset) == len(gold_dataset), 'Lengths of the predictions and golds should be the same'
    assert set(current_dataset['context_id']) == set(gold_dataset['context_id']), \
        f"Invalid context_ids: {set(current_dataset['context_id']) ^ set(gold_dataset['context_id'])}"

    current_dataset = pd.merge(
        left=gold_dataset['context_id'], right=current_dataset, on='context_id', validate='1:1'
    )
    return compute_wsi_metrics_lexsubgen(
        y_true=gold_dataset['gold_sense_id'], y_pred=current_dataset['predict_sense_id'],
        group_by=gold_dataset['word'], context_ids=current_dataset['context_id'].astype(str)
    )


MATCH_SEMEVAL_SCORES_RE = re.compile(r"(\w+|\w+\.\w+)(\t*-?\d+\.?\d*)+")
MATCH_TOTAL_VALUE = re.compile(r"Total (.+):(.+)")
METRICS = [
    "ARI",
    "NMI",
    "goldInstance",
    "sysInstance",
    "goldClusterNum",
    "sysClusterNum",
]
SEMEVAL_METRICS = [
    "S13_Precision",
    "S13_Recall",
    "S13_F1",
    "S13_FNMI",
    "S13_AVG",
    "S10_FScore",
    "S10_Precision",
    "S10_Recall",
    "S10_VMeasure",
    "S10_Homogeneity",
    "S10_Completeness",
    "S10_AVG",
]
ALL_METRICS = METRICS + SEMEVAL_METRICS


def compute_wsi_metrics_lexsubgen(
    y_true: tp.List,
    y_pred: tp.List,
    group_by: tp.List[str],
    context_ids: tp.List[str],
    y_true_file: str = None,
    compute_semeval_metrics_f: bool = True
) -> tp.Tuple[tp.Dict[str, float], pd.DataFrame]:
    """
    Computes clustering metrics: @METRICS
    Args:
        y_true: ground truth
        y_pred: predicted labels
        group_by: @y_true and @y_pred must be grouped by @group_by
            and METRICs must be computed for each group
        context_ids: unique indexes of instances
        y_true_file: if not None true labels will be read from @y_true_file file
    """
    if compute_semeval_metrics_f:
        semeval_2013_values = wsi_metrics.compute_semeval_2013_metrics(
            y_true, y_pred, group_by, context_ids, y_true_file
        )
        semeval_2010_values = wsi_metrics.compute_semeval_2010_metrics(
            y_true, y_pred, group_by, context_ids, y_true_file
        )

    scores_per_word = wsi_metrics.compute_scores_per_word(y_true, y_pred, group_by)
    per_word_df = pd.DataFrame(scores_per_word, columns=['word'] + METRICS)
    if compute_semeval_metrics_f:
        mean_values = (
            wsi_metrics.compute_weighted_avg(per_word_df, METRICS)
            + semeval_2013_values["all"]
            + semeval_2010_values['all']
        )
    else:
        mean_values = (
            wsi_metrics.compute_weighted_avg(per_word_df, METRICS)
        )
    scores_per_word.append(["word_weighted_avg"] + mean_values)
    if compute_semeval_metrics_f:
        for i, word in enumerate(per_word_df.word):
            scores_per_word[i].extend(
                semeval_2013_values[word] + semeval_2010_values[word]
            )

    all_metrics = pd.DataFrame(scores_per_word, columns=["word"] + ALL_METRICS)
    mean_metrics = OrderedDict(
        (metric, value)
        for metric, value in zip(ALL_METRICS, mean_values)
    )

    return mean_metrics, all_metrics
