import os
import sys
import json
import typing as tp
import itertools
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from omegaconf import DictConfig

from multilang_wsi_evaluation.wsi_metrics import compute_unsup_metrics, compute_words_metrics, aggregate_metrics
from multilang_wsi_evaluation.interfaces import IWSIVectorizer
from multilang_wsi_evaluation.utils import WSIDatasetPart, load_parts, load_obj


@dataclass
class BaseClusterer:
    clusterer_type: type
    param_to_values: tp.Dict[str, tp.List[tp.Any]]


class Clusterer:  # Actually should be tp.Protocol (python>=3.8)
    def fit_predict(self, matrix: tp.Any) -> tp.Any:
        pass


def _dict_to_params_str(dct: tp.Dict[str, tp.Any]) -> str:
    return ', '.join(f'{key}={json.dumps(value)}' for key, value in sorted(dct.items()))


def _type_full_name(type_: type) -> str:
    module = type_.__module__
    if module is None or module == str.__module__:
        return type_.__name__
    return f'{module}.{type_.__name__}'


def _dist_full_name(dist: tp.Any) -> str:
    if isinstance(dist, str):
        return dist
    return _type_full_name(dist)


def generate_clusterers(base_clusterers: tp.Iterable[BaseClusterer]) \
        -> tp.Generator[tp.Tuple[Clusterer, str], None, None]:
    for base_clusterer in base_clusterers:
        for param_values in itertools.product(*base_clusterer.param_to_values.values()):
            param_to_value = {param: value for param, value in zip(base_clusterer.param_to_values.keys(), param_values)}
            clusterer = base_clusterer.clusterer_type(**param_to_value)
            clusterer_desc = f'{_type_full_name(base_clusterer.clusterer_type)}({_dict_to_params_str(param_to_value)})'
            yield clusterer, clusterer_desc


class VectorizerEvaluator:
    def __init__(self, vectorizer: IWSIVectorizer, base_clusterers: tp.List[BaseClusterer],
                 dists: tp.List[tp.Union[str, tp.Callable]], verbose: bool = True) -> None:
        self.vectorizer = vectorizer
        self.base_clusterers = base_clusterers
        self.verbose = verbose
        self.dists = dists

    def evaluate(self, part: WSIDatasetPart):
        group_ids, groups_samples_ids, groups_samples = part.groups_samples()
        self.vectorizer.fit([sample for group_samples in groups_samples for sample in group_samples])
        preds = defaultdict(dict)
        word_metrics = defaultdict(dict)
        for group_id, group_samples_ids, group_samples in tqdm(zip(group_ids, groups_samples_ids, groups_samples),
                                                               disable=not self.verbose, total=len(group_ids)):
            matrix = self.vectorizer.predict(group_samples)
            for dist in (self.dists or [None]):
                word_dists_matrix = pairwise_distances(matrix, metric=dist) if dist is not None else matrix
                for clusterer, clusterer_desc in generate_clusterers(self.base_clusterers):
                    clusters = clusterer.fit_predict(word_dists_matrix)
                    clusterer_params = (_dist_full_name(dist), clusterer_desc)
                    word_metrics[clusterer_params][group_id] = \
                        compute_unsup_metrics(group_id, word_dists_matrix, matrix, clusters)
                    preds[clusterer_params][group_id] = \
                        {sample_id: cluster for sample_id, cluster in zip(group_samples_ids, clusters)}
        batch_metrics = defaultdict(pd.DataFrame)
        for hypers, hypers_preds in tqdm(preds.items(), disable=not self.verbose):
            batch_metrics[hypers] = compute_words_metrics(part, hypers_preds)
        agg_metrics = aggregate_metrics(batch_metrics, word_metrics)
        return agg_metrics, batch_metrics


def save_metrics(word_metrics, aggregated_metrics, name: str) -> None:
    #TODO: stat for max_ari(min, max, mean, quartiles...) and params for fh_max_ari
    df_save = pd.DataFrame()
    for part_id, metrics in word_metrics.items():
        for hypers, word_searches in metrics.items():
            word_searches['hypers'] = [hypers] * len(word_searches)
            word_searches['part'] = [part_id] * len(word_searches)
            df_save = pd.concat([df_save, word_searches], ignore_index=True)
    metrics = list(df_save.columns)[1:-2]
    for metric in metrics:
        metric_df = df_save[['hypers', 'part', 'word', metric]]
        part_path = os.path.join(name, 'grid_metrics', f'{metric}.tsv')
        Path(part_path).parent.mkdir(parents=True, exist_ok=True)
        metric_df.to_csv(part_path, index=False, sep='\t')
    df_save = pd.DataFrame()
    for part_id, metrics_df in aggregated_metrics.items():
        metrics_df['part'] = [part_id] * len(metrics_df)
        df_save = pd.concat([df_save, metrics_df], ignore_index=True)
    df_save = df_save.rename(columns={'index': 'metric', '0': 'score'})
    for metric in df_save['metric'].unique():
        agg_df = df_save[df_save['metric'] == metric]
        part_path = os.path.join(name, 'agg_metrics', f'{metric}.tsv')
        Path(part_path).parent.mkdir(parents=True, exist_ok=True)
        agg_df.to_csv(part_path, index=False, sep='\t')


def get_base_clusterers(cfg: tp.Optional[DictConfig]) -> tp.List[BaseClusterer]:
    if cfg is None:
        raise ValueError("set your model_clusterer in config.model_clusterer")

    cfg_dict = dict(cfg)
    cfg_dict.pop('_target_')
    return [BaseClusterer(load_obj(cfg._target_), cfg_dict)]


def get_distances(metrics: tp.Optional[tp.Iterable[str]], callable_metrics: tp.Optional[tp.Iterable[str]]) \
        -> tp.List[tp.Union[str, tp.Callable]]:
    loaded_metrics = [load_obj(metric) for metric in (callable_metrics or [])]
    return list(metrics or []) + loaded_metrics


def vec_eval(vectorizer: IWSIVectorizer, cfg_clusterer: tp.Optional[DictConfig], list_paths_datasets: tp.List[str],
             metrics: tp.Optional[tp.Iterable[str]] = None, callable_metrics: tp.Optional[tp.Iterable[str]] = None,
             opt_metric: tp.Optional[str] = None, opt_dataset: tp.Optional[str] = None,
             vectorizer_name: tp.Optional[str] = None, verbose: bool = True):
    if vectorizer_name is None:
        vectorizer_name = vectorizer.__class__.__name__

    evaluator = VectorizerEvaluator(vectorizer, get_base_clusterers(cfg_clusterer),
                                    get_distances(metrics, callable_metrics), verbose=verbose)
    info_file = sys.stdout if verbose else open(os.devnull, 'w')
    word_metrics, aggregated_metrics = {}, {}
    for part in load_parts(*list_paths_datasets):
        print(f"Evaluating '{part.id}'...", file=info_file)
        aggregated_metrics[part.id], word_metrics[part.id] = evaluator.evaluate(part)

    save_metrics(word_metrics, aggregated_metrics, vectorizer_name)
    if opt_metric and opt_dataset:
        return 1 #list(part_to_words_metrics[opt_dataset][opt_metric][0].metrics)
