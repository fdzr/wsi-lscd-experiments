from tools.app.interfaces import IWSIVectorizer
from tools.app.data_objects import Sample
from typing import List, Union
import logging
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)


class SubstsFrequencyBasedVectorizer(IWSIVectorizer):
    def __init__(self, analyzer: str, topk: int):
        self.analyzer = analyzer
        self.topk = topk

    def vectorize(self, samples: List[Sample]) -> Union[np.ndarray, List[List[float]]]:
        vectorizer = CountVectorizer(analyzer=self.analyzer)
        vectors = vectorizer.fit_transform([' '.join([subst[1] for subst in s.substs]) for s in samples])
        return vectors.toarray()

