import csv
import typing as tp
import importlib
from pathlib import Path

import pandas as pd

from multilang_wsi_evaluation.interfaces import Sample


def _convert_df_to_samples(dataset_df: pd.DataFrame) -> tp.List[Sample]:
    def convert_row_to_sample(row):
        begin, end = row['positions'].split('-')
        return Sample(context=row['context'], begin=int(begin), end=int(end), lemma=row['word'])

    return [convert_row_to_sample(row) for _, row in dataset_df.iterrows()]


class WSIDatasetPart:
    def __init__(self, dataset_name: str, lang: str, part: str, dataset_df: pd.DataFrame):
        self.dataset_name = dataset_name
        self.lang = lang
        self.part = part
        self.dataset_df = dataset_df

    @property
    def id(self) -> str:
        return f'{self.dataset_name}-{self.lang}-{self.part}'

    def groups_samples(self) -> tp.Tuple[tp.List[tp.Any], tp.List[tp.List[tp.Any]], tp.List[tp.List[Sample]]]:
        groups, groups_samples_ids, groups_samples = [], [], []
        for word, word_ids in self.dataset_df.groupby('word').groups.items():
            word_df = self.dataset_df.loc[word_ids]
            groups.append(word)
            groups_samples_ids.append(list(word_df['context_id']))
            groups_samples.append(_convert_df_to_samples(word_df))
        return groups, groups_samples_ids, groups_samples

    def samples(self) -> tp.Tuple[tp.List[tp.Any], tp.List[Sample]]:
        samples = _convert_df_to_samples(self.dataset_df)
        samples_ids = list(self.dataset_df['context_id'])
        return samples_ids, samples

    @classmethod
    def from_file(cls, filepath: str) -> 'WSIDatasetPart':
        dataset_name, lang, part = Path(filepath).resolve().with_suffix('').parts[-3:]
        dataset_df = pd.read_csv(filepath, sep='\t', quoting=csv.QUOTE_MINIMAL)
        return cls(dataset_name, lang, part, dataset_df)


def _load_part(base_path: Path) -> tp.Generator[WSIDatasetPart, None, None]:
    if base_path.is_file() and base_path.suffix == '.tsv':
        yield WSIDatasetPart.from_file(str(base_path))
    elif base_path.is_dir():
        for sub_path in base_path.iterdir():
            yield from _load_part(sub_path)


def load_parts(*base_paths: str) -> tp.Generator[WSIDatasetPart, None, None]:
    for base_path in base_paths:
        yield from _load_part(Path(base_path))


def load_obj(obj_path: str, default_obj_path: str = "") -> tp.Any:
    """Extract an object from a given path.
    Taken from: https://github.com/kedro-org/kedro/blob/e78990c6b606a27830f0d502afa0f639c0830950/kedro/utils.py#L8
    Args:
        obj_path: Path to an object to be extracted, including the object name.
        default_obj_path: Default object path.
    Returns:
        Extracted object.
    Raises:
        AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)
