from tools.app.interfaces import IClusterer
from tools.app.data_objects import ClusterSearchResult
from typing import List, Union
from sklearn.cluster import AgglomerativeClustering
import numpy as np


class AgglomerativeClusterer(IClusterer):

    def __init__(self, n_clusters: int, affinity: str, linkage: str):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.linkage = linkage
        self.children_ = None
        self.distances_ = None
        self.labels_ = None
        self.clusterer = AgglomerativeClustering(
            compute_distances=True,
            compute_full_tree=True,
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            linkage=self.linkage
        )

    def cluster(self, vectors: Union[List[List[float]], np.ndarray]) -> ClusterSearchResult:
        clusters = self.clusterer.fit_predict(np.array(vectors)).tolist()
        self.labels_ = clusters
        self.distances_ = self.clusterer.distances_
        self.children_ = self.clusterer.children_
        return ClusterSearchResult(
            clusters=clusters,
            clusterer_type=type(self),
            params={
                "n_clusters": self.n_clusters,
                "affinity": self.affinity,
                "linkage": self.linkage
            })
