from tools.app.interfaces import IClusterer
from tools.app.data_objects import ClusterSearchResult
from typing import List, Union, Any
import numpy as np
from scipy.spatial.distance import jensenshannon
import logging
from scipy.special import softmax

logger = logging.getLogger(__name__)


class SubstsFrequencyBasedClusterer(IClusterer):

    def __init__(self, softmax_temperature: float, n_clusters: int):
        self.n_clusters = n_clusters
        self._vectors = None
        self.labels_ = None
        self.children_ = None
        self.distances_ = None
        self._softmax_temperature = softmax_temperature

    def cluster(self, vectors: Union[List[List[float]], np.ndarray]) -> ClusterSearchResult:
        self._vectors = vectors
        self.labels_ = np.arange(0, vectors.shape[0]).astype(int)
        self.children_ = np.zeros((vectors.shape[0] - 1, 2))
        self.distances_ = np.zeros((self.children_.shape[0],))
        return ClusterSearchResult(
            clusters=self.fit_predict().tolist(),
            clusterer_type=type(self),
            params={
                "n_clusters": self.n_clusters
            })

    def fit_predict(self) -> Any:

        iteration = 0
        labels = np.unique(self.labels_)
        result = None
        next_label = np.max(labels) + 1
        cache = {}
        while len(labels) > 1:
            min_i, min_j = None, None
            min_dist = float("inf")

            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    cache_key = "{}_{}".format(labels[i], labels[j])
                    if cache_key not in cache:
                        vecs1 = self._vectors[self.labels_ == labels[i]]
                        vecs2 = self._vectors[self.labels_ == labels[j]]

                        cache[cache_key] = self.distance(vecs1.sum(axis=0), vecs2.sum(axis=0))

                    dist = cache[cache_key]
                    if dist < min_dist:
                        min_dist = dist
                        min_i = i
                        min_j = j

            logger.error(
                f'Iteration: {iteration}, n_clusters: {self.n_clusters}, Mins {min_i}:{min_j}, {labels[min_i]}:{labels[min_j]}, {len(labels)}, {min_dist}, {next_label}')

            self.labels_[self.labels_ == labels[min_i]] = next_label
            self.labels_[self.labels_ == labels[min_j]] = next_label

            self.children_[iteration, 0] = labels[min_i]
            self.children_[iteration, 1] = labels[min_j]
            self.distances_[iteration] = min_dist

            labels = np.unique(self.labels_)
            labels.sort()
            iteration += 1
            next_label += 1

            if len(labels) == self.n_clusters:
                result = self.labels_.copy()

        cluster_to_label = {s: i for i, s in enumerate(np.unique(self.labels_))}
        vf = np.vectorize(lambda x: cluster_to_label[x])
        self.labels_ = vf(self.labels_)

        cluster_to_label = {s: i for i, s in enumerate(np.unique(result))}
        vf = np.vectorize(lambda x: cluster_to_label[x])
        result = vf(result)

        self.children_ = self.children_.astype(int)
        return result

    def distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return min([jensenshannon(self.softmax(vec1), self.softmax(vec2)), 1.0])

    def softmax(self, vec: np.ndarray, temperature: int = 1) -> np.ndarray:
        return softmax(vec / temperature)
