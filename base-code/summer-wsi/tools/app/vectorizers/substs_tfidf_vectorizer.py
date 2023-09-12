from tools.app.interfaces import IWSIVectorizer
from tools.app.data_objects import Sample
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SubstsTfidfVectorizer(IWSIVectorizer):
    def __init__(self, min_df: float, max_df: float, analyzer: str, topk: int):
        self.topk = topk
        self.vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, analyzer=analyzer)
        self._features = None

    def vectorize(self, samples: List[Sample]) -> np.ndarray:
        self._fit(samples)
        self._features = self.vectorizer.get_feature_names()
        return self._transform(samples)

    def _fit(self, samples: List[Sample]) -> None:
        self.vectorizer.fit(self._samples_to_data_list(samples))

    def _transform(self, samples: List[Sample]) -> np.ndarray:
        vectors = self.vectorizer.transform(self._samples_to_data_list(samples))
        return vectors.toarray()

    @staticmethod
    def _samples_to_data_list(samples: List[Sample]) -> List[str]:
        return list(map(lambda s: ' '.join([sub[1] for sub in s.substs]), samples))
