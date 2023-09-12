import uuid

import pymongo
import os
from typing import Any, List, Dict, Union
import pickle
from tools.app.data_objects import Experiment, Corpus
from tools.app.interfaces import IDao
from tools.app.factories import ExperimentFactory
import logging

logger = logging.getLogger(__name__)

USER = os.getenv('MONGO_DB_ROOT_USERNAME')
PASSWORD = os.environ.get('MONGO_DB_ROOT_PASSWORD')
HOST = os.environ.get('MONGO_HOST', 'localhost')

DATAFILE_DIRECTORY = os.path.join(os.getcwd(), 'tools', 'app', 'data', 'datafiles')

if USER and PASSWORD:
    conn_str = "mongodb://{}:{}@{}/?retryWrites=true&w=majority".format(USER, PASSWORD, HOST)
else:
    conn_str = "mongodb://{}/?retryWrites=true&w=majority".format(HOST)


class Dao(IDao):

    def __init__(self):
        client = pymongo.MongoClient(conn_str, serverSelectionTimeoutMS=5000)
        self.db = client['WordSense']

    def cache_setdefault(self, key: str, generate_value: callable):
        value = self.cache_get_value(key)
        if value is None:
            value = generate_value()
            self.cache_set_value(key, value)

        return value

    def cache_set_value(self, key: str, value: Any):
        self.db['cache'].replace_one({
            '_id': key,
        }, {
            'value': pickle.dumps(value)
        }, True)

    def cache_get_value(self, key: str, default: Any = None):
        document = self.db['cache'].find_one({
            '_id': key
        })
        if not document:
            return default
        else:
            return pickle.loads(document['value'])

    def get_experiment_list(self, task: str = None) -> List[Experiment]:
        if task:
            documents = self.db['experiments'].find({'task': task})
        else:
            documents = self.db['experiments'].find()

        experiments = []

        for document in documents:
            document['result'] = None
            experiments.append(ExperimentFactory.from_document(document))

        return experiments

    def get_experiment_by_id(self, experiment_id: str) -> Experiment:
        document = self.db['experiments'].find_one({'_id': experiment_id})

        with open(document['result_datafile'], 'rb') as f:
            document['result'] = pickle.load(f)

        del document['result_datafile']

        return ExperimentFactory.from_document(document)

    def add_experiment(self, experiment: Experiment):
        doc = self._experiment_to_doc(experiment)
        datafile_name = str(uuid.uuid4())

        datafile_path = os.path.join(DATAFILE_DIRECTORY, datafile_name)
        doc['result_datafile'] = datafile_path

        with open(datafile_path, 'wb') as f:
            pickle.dump({}, f)

        self.db['experiments'].insert_one(doc)

    def update_experiment(self, experiment: Experiment):
        doc = self._experiment_to_doc(experiment)

        datafile_name = str(uuid.uuid4())

        datafile_path = os.path.join(DATAFILE_DIRECTORY, datafile_name)
        doc['result_datafile'] = datafile_path

        with open(datafile_path, 'wb') as f:
            pickle.dump(doc['result'], f)

        del doc['result']

        self.db['experiments'].replace_one({'_id': experiment._id}, doc, True)

    def get_dataset_by_id(self, dataset_id: str) -> Union[List[Corpus], None]:
        document = self.db['datasets'].find_one({'_id': dataset_id})
        if not document:
            return None

        with open(document['datafile'], 'rb') as f:
            return pickle.load(f)

    def add_dataset(self, dataset_id: str, task: str, data: List[Corpus]) -> None:
        datafile_name = str(uuid.uuid4())

        datafile_path = os.path.join(DATAFILE_DIRECTORY, datafile_name)
        with open(datafile_path, 'wb') as f:
            pickle.dump(data, f)

        self.db['datasets'].insert_one({
            '_id': dataset_id,
            'datafile': datafile_path,
            'task': task
        })

    def get_datasets(self, task: str = None) -> List[str]:
        if task:
            documents = self.db['datasets'].find({'task': task})
        else:
            documents = self.db['datasets'].find()

        return documents

    @staticmethod
    def _experiment_to_doc(experiment: Experiment) -> Dict[str, Any]:
        doc = experiment.__dict__.copy()
        doc['config'] = pickle.dumps(doc['config'])
        doc['target_words'] = list(doc['target_words'])
        return doc

