from tools.app.data_objects import Corpus, Sample, ClusterSearchResult, SubstWSIExperimentWordResult, Experiment
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union
import numpy as np


class IWSIVectorizer(ABC):
    @abstractmethod
    def vectorize(self, samples: List[Sample]) -> Union[List[List[float]], np.ndarray]:
        ...


class IClusterer(ABC):
    @abstractmethod
    def cluster(self, vectors: Union[List[List[float]], np.ndarray]) -> ClusterSearchResult:
        ...

    def distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        ...


class IDatasetLoader(ABC):

    @abstractmethod
    def __init__(self, dataset_id: str, input_filenames: List[str], input_directories: List[str],
                 corpora_names: List[str]):
        super(IDatasetLoader, self).__init__()
        self.input_filename = input_filenames
        self.input_directory = input_directories
        self.corpora_names = corpora_names
        self.dataset_id = dataset_id

    @abstractmethod
    def load(self) -> List[Corpus]:
        ...


class IDao(ABC):

    @abstractmethod
    def get_experiment_by_id(self, experiment_id: str) -> Experiment:
        ...

    @abstractmethod
    def get_experiment_list(self, task: str = None) -> List[Experiment]:
        ...

    @abstractmethod
    def cache_get_value(self, key: str, default: Any = None):
        ...

    @abstractmethod
    def cache_set_value(self, key: str, value: Any):
        ...

    @abstractmethod
    def cache_setdefault(self, key: str, generate_value: callable):
        ...

    @abstractmethod
    def add_experiment(self, experiment: Experiment):
        ...

    @abstractmethod
    def update_experiment(self, experiment: Experiment):
        ...

    @abstractmethod
    def get_dataset_by_id(self, dataset_id: str) -> Union[List[Corpus], None]:
        ...

    @abstractmethod
    def add_dataset(self, dataset_id: str, task: str, dataset: List[Corpus]) -> None:
        ...

    @abstractmethod
    def get_datasets(self, task: str = None) -> List[str]:
        ...


class ISolver(ABC):
    @abstractmethod
    def __init__(self, dao: IDao, vectorizers: IWSIVectorizer, clusterers: IClusterer, experiment: Experiment):
        super(ISolver, self).__init__()
        self.experiment = experiment
        self.vectorzers = vectorizers
        self.clusterers = clusterers
        self.dao = dao

    @abstractmethod
    def solve(self, dataset_id: str) -> Union[
        Dict[str, Any], Dict[str, List[SubstWSIExperimentWordResult]]]:
        ...
