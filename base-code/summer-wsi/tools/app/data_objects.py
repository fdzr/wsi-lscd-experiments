from typing import Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from numpy import ndarray


@dataclass
class Sample:
    context: str
    begin: int
    end: int
    lemma: str
    # optional corpus id
    corpus: int = 0
    # optional label id for labeled datasets
    label_id: int = None
    # optional list of substitutes, used only with substitutes based methods, otherwise is None
    substs: List[Tuple[float, str]] = None


@dataclass
class ClusterSearchResult:
    clusterer_type: type
    params: Dict[str, Any]
    clusters: List[int]


@dataclass
class Corpus:
    name: str
    data: List[Sample]


@dataclass
class SubstWSISolverConfig:
    clusterers: Dict[str, Dict[str, List[Union[float, str, int]]]]
    vectorizers: Dict[str, Dict[str, List[Union[float, str, int]]]]
    sample_size: int
    random_seed: int


@dataclass
class Experiment:
    _id: str
    status: str
    dataset_id: str
    task: str
    config: Any
    result: Any
    start_time: Union[str, None] = None
    end_time: Union[str, None] = None


@dataclass
class SubstWSIExperimentWordResult:
    ari_score: float
    max_ari_score: float
    score: float
    cluster_pmis: List[float]
    dendrogram: Dict[str, Any]
    graph: Dict[str, Any]
    vectors: ndarray
    clusters: List[int]
    corpus_split_index: int
    sil_scores: List[float]
    ari_scores: List[float]
    n_clusters_list: List[int]
    config: Union[Dict[str, Any], None] = None
    vector_features: List[str] = None
    senses: Dict[int, str] = field(default_factory=dict)


@dataclass
class SubstWSIExperiment(Experiment):
    target_words: set = None
    result: Dict[str, SubstWSIExperimentWordResult] = field(default_factory=list)
