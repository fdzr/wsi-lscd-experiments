from tools.app.vectorizers.substs_tfidf_vectorizer import SubstsTfidfVectorizer
from tools.app.vectorizers.substs_probability_based_vectorizer import SubstsProbabilityBasedVectorizer
from tools.app.vectorizers.substs_frequency_based_vectorizer import SubstsFrequencyBasedVectorizer
from tools.app.clusterers.agglomerative_clusterer import AgglomerativeClusterer
from tools.app.clusterers.substs_probability_based_clusterer import SubstsProbabilityBasedClusterer
from tools.app.clusterers.substs_frequency_based_clusterer import SubstsFrequencyBasedClusterer
from tools.app.interfaces import ISolver, IDatasetLoader, IWSIVectorizer, IClusterer, IDao
from tools.app.data_objects import Experiment, SubstWSIExperiment, SubstWSISolverConfig, SubstWSIExperimentWordResult
from tools.app.constants import TASK_SUBST_WSI
from tools.app.solvers.subst_wsi_solver import SubstWSISolver
from tools.app.dataset_loaders.subst_wsi_dataset_loader import SubstWSIDatasetLoader
from typing import Dict, Any, List
from typing import Union
import itertools
import logging
import pickle
import uuid
from os import listdir
from os.path import isfile, join

logger = logging.getLogger(__name__)


class SolverFactory:

    @classmethod
    def _get_vectorizers(cls, config: Union[SubstWSISolverConfig, SubstWSISolverConfig]) -> List[IWSIVectorizer]:
        vectorizers = []

        for vectorizer in config.vectorizers:

            for param_values in itertools.product(*config.vectorizers[vectorizer].values()):
                param_to_value = {param: value for param, value in
                                  zip(config.vectorizers[vectorizer].keys(), param_values)}
                vectorizers.append(eval(vectorizer)(**param_to_value))

        return vectorizers

    @classmethod
    def _get_clusterers(cls, config: Union[SubstWSISolverConfig, SubstWSISolverConfig]) -> List[IClusterer]:
        clusterers = []

        for clusterer in config.clusterers:
            for param_values in itertools.product(*config.clusterers[clusterer].values()):
                param_to_value = {param: value for param, value in
                                  zip(config.clusterers[clusterer].keys(), param_values)}

                clusterers.append(eval(clusterer)(**param_to_value))

        return clusterers

    @classmethod
    def from_experiment(cls, dao: IDao, experiment: Experiment) -> ISolver:
        vectorizers = cls._get_vectorizers(experiment.config)
        clusterers = cls._get_clusterers(experiment.config)

        if isinstance(experiment, SubstWSIExperiment):
            return SubstWSISolver(dao, vectorizers, clusterers, experiment)


class DatasetLoaderFactory:
    @classmethod
    def from_task(cls, dao: IDao, task: str, dataset_id: str, input_filenames: List[str],
                  input_directories: List[str], corpora_names: List[str]) -> IDatasetLoader:
        if task == TASK_SUBST_WSI:
            return SubstWSIDatasetLoader(dao, dataset_id, input_filenames, input_directories, corpora_names)


class SolverConfigFactory:
    @classmethod
    def from_task(cls, task: str, config: Dict[str, Any]):
        if task == TASK_SUBST_WSI:
            return SubstWSISolverConfig(**config)


class ExperimentResultFactory:
    @classmethod
    def from_task(cls, task: str, result: Dict[str, Any]) -> Union[Dict[str, SubstWSIExperimentWordResult]]:
        if task == TASK_SUBST_WSI:
            return {word: SubstWSIExperimentWordResult(**word_result) for word, word_result in result.items()}


class ExperimentFactory:
    @classmethod
    def from_document(cls, document: Dict[str, Any]):
        task = document.get('task')
        if task == TASK_SUBST_WSI:
            return SubstWSIExperiment(
                config=pickle.loads(document.get('config')),
                _id=document.get('_id'),
                start_time=document.get('start_time'),
                end_time=document.get('end_time'),
                result=document.get('result'),
                status=document.get('status'),
                target_words=set(document.get('target_words')),
                dataset_id=document.get('dataset_id'),
                task=document.get('task')
            )

    @classmethod
    def from_task(cls, task: str, dataset_id: str, target_words: List[str], config: Dict[str, Any]):
        experiment_id = str(uuid.uuid4())

        if task == TASK_SUBST_WSI:
            return SubstWSIExperiment(
                config=SolverConfigFactory.from_task(task, config),
                _id=experiment_id,
                start_time=None,
                end_time=None,
                result={},
                status="not_started",
                target_words=set(target_words),
                dataset_id=dataset_id,
                task=task
            )
