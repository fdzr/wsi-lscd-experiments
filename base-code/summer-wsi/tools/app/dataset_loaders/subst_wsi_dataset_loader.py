import os.path

from tools.app.interfaces import IDatasetLoader, IDao
from tools.app.data_objects import Sample, Corpus
from tools.app.constants import TASK_SUBST_WSI
from substwsi.substs_loading import load_substs
import pandas as pd
from typing import List
import logging

logger = logging.getLogger(__name__)


class SubstWSIDatasetLoader(IDatasetLoader):
    def __init__(self, dao: IDao, dataset_id: str, input_filenames: List[str], input_directories: List[str],
                 corpora_names: List[str]):
        super(SubstWSIDatasetLoader, self).__init__(dataset_id, input_filenames, input_directories, corpora_names)
        self.input_filenames = input_filenames
        self.input_directories = input_directories
        self.corpora_names = corpora_names
        self.dataset_id = dataset_id
        self.dao = dao

    def load(self) -> List[Corpus]:
        data = []

        for i in range(len(self.input_directories)):

            df = load_substs(os.path.join(self.input_directories[i], self.input_filenames[i]))

            data.append(Corpus(data=self._df_to_samples(df, i), name=self.corpora_names[i]))

        self.dao.add_dataset(self.dataset_id, TASK_SUBST_WSI, data)

        return data

    @staticmethod
    def _df_to_samples(dataset_df: pd.DataFrame, corpus: int = 0):
        def convert_row_to_sample(row):
            begin, end = row['positions']
            substs_probs = row['substs_probs']
            substs_probs_converted = []
            for prob, word in substs_probs:
                substs_probs_converted.append((float(prob), str(word).strip()))

            return Sample(
                context=row['context'],
                begin=int(begin),
                end=int(end),
                lemma=row['word'],
                substs=substs_probs_converted,
                label_id=row.get('gold_sense_id', -1),
                corpus=corpus
            )

        return [convert_row_to_sample(row) for _, row in dataset_df.iterrows()]
