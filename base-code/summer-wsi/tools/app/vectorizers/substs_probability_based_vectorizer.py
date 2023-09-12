from tools.app.interfaces import IWSIVectorizer
from tools.app.data_objects import Sample
from typing import List, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SubstsProbabilityBasedVectorizer(IWSIVectorizer):
    def __init__(self, topk: int, left_out_substs_prob_weight: float):
        self.features_ = None
        self.topk = topk
        self.left_out_substs_prob_weight = left_out_substs_prob_weight

    def vectorize(self, samples: List[Sample]) -> Union[np.ndarray, List[List[float]]]:
        all_substs = set()
        vectors = []

        for sample in samples:
            for prob, subst in sample.substs:
                all_substs.add(subst)

        self.features_ = list(all_substs)
        all_substs_indices = {s: i for i, s in enumerate(self.features_)}

        for sample in samples:
            vector = [-1] * len(self.features_)
            total_prob = 0
            total_substs = 0
            for prob, subst in sample.substs:
                vector[all_substs_indices[subst]] = prob
                total_prob += prob
                total_substs += 1
            remains = 1 - total_prob
            uniform_prob = self.left_out_substs_prob_weight * (remains / (len(vector) - total_substs))
            vectors.append(list(map(lambda i: uniform_prob if i == -1 else i, vector)))

        return np.array(vectors)
