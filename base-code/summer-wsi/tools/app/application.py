from builtins import str
import zipfile
import uuid
from flask import Flask, request, render_template, redirect, url_for, Response
from tools.app.factories import SolverFactory, DatasetLoaderFactory, ExperimentFactory
from tools.app.constants import TASK_SUBST_WSI
import json
import logging
import os
from tools.app.dao import Dao
from celery import Celery
import datetime
from scipy.spatial.distance import cdist
import numpy as np
from typing import List, Dict, Any
from tools.app.clusterers.substs_frequency_based_clusterer import SubstsFrequencyBasedClusterer
from tools.app.clusterers.substs_probability_based_clusterer import SubstsProbabilityBasedClusterer
from tools.app.clusterers.agglomerative_clusterer import AgglomerativeClusterer
from pathlib import Path
from sklearn.preprocessing import normalize
import inspect
from tools.app.interfaces import IDao

logger = logging.getLogger(__name__)

REDIS_HOST: str = os.environ.get('REDIS_HOST', 'localhost')

celery_app = Celery('application', backend="redis://{}".format(REDIS_HOST), broker='redis://{}'.format(REDIS_HOST))
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'tools', 'app', 'data', 'tmp', 'data_loading')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.secret_key = 'LSCD_TOOL_SEC_KEY'

dao = None


def get_dao() -> IDao:
    global dao
    if dao:
        return dao
    dao = Dao()
    return dao


DEFAULT_CONFIG = """
{
    "clusterers": {
        "AgglomerativeClusterer": {
            "n_clusters": [3,4],
            "linkage": ["average"],
            "affinity": ["cosine"]
            }
        },
    "vectorizers": {
        "SubstsTfidfVectorizer": {
            "analyzer": ["word"],
            "min_df": [0.3],
            "max_df": [0.9],
            "topk": [128]
            }
        },
    "k": 10,
    "n": 15
}""".strip()


def filter_dict(dict_to_filter, thing_with_kwargs):
    sig = inspect.signature(thing_with_kwargs)
    filter_keys = [param.name for param in sig.parameters.values() if
                   param.kind == param.POSITIONAL_OR_KEYWORD]
    filtered_dict = {filter_key: dict_to_filter.get(filter_key) for filter_key in filter_keys if
                     not filter_key.startswith('_') and not filter_key.endswith('_')}
    return filtered_dict


@app.route('/', defaults={'word': None, 'subtask': 'overview', 'experiment_id': None})
@app.route('/experiment/<experiment_id>', defaults={'subtask': 'overview', 'word': None})
@app.route('/experiment/<experiment_id>/<subtask>', defaults={'word': None})
@app.route('/experiment/<experiment_id>/<subtask>/<word>', methods=["GET"])
def index(experiment_id: str = None, subtask: str = None, word: str = None) -> str:
    experiment = None
    experiments = None
    if experiment_id:
        experiment = get_dao().get_experiment_by_id(experiment_id)
        try:
            del experiment.result[word].config['clusterer_params']['children_']
            del experiment.result[word].config['clusterer_params']['labels_']
            del experiment.result[word].config['clusterer_params']['distances_']
        except:
            pass
    else:
        experiments = get_dao().get_experiment_list(TASK_SUBST_WSI)

    datasets = get_dao().get_datasets(TASK_SUBST_WSI)

    return render_template(
        'layout.html',
        page='index',
        task=TASK_SUBST_WSI,
        subtask=subtask,
        word=word,
        experiment=experiment,
        experiments=experiments,
        datasets=datasets,
        menu=[
            {'url_for': 'index', 'params': {}, 'text': 'Home'},
            {'url_for': 'dataset_get', 'params': {}, 'text': 'Upload dataset'}
        ]
    )


@app.route('/experiment/<experiment_id>/graph/<word>/data', methods=["GET"])
def get_graph(experiment_id: str = None, word: str = None) -> str:
    experiment = get_dao().get_experiment_by_id(experiment_id)
    graph_data = experiment.result[word].graph
    graph_data['vectors'] = experiment.result[word].vectors.tolist()
    return json.dumps(graph_data)


@app.route('/experiment/<experiment_id>/graph/<word>/embeddings', methods=["GET"])
def get_graph_embeddings(experiment_id: str = None, word: str = None) -> str:
    experiment = get_dao().get_experiment_by_id(experiment_id)
    method = request.args.get('method')

    embeddings = []

    vectors = experiment.result[word].vectors

    if method == 'tsne':
        early_exaggeration = float(request.args.get('early_exaggeration'))
        early_exaggeration_n_iter = int(request.args.get('early_exaggeration_n_iter'))
        exaggeration = float(request.args.get('exaggeration'))
        n_iter = int(request.args.get('n_iter'))
        metric = request.args.get('metric')
        perplexity = float(request.args.get('perplexity'))

        from openTSNE import TSNEEmbedding
        from openTSNE import affinity
        from openTSNE import initialization

        if metric == 'clusterer':
            from sklearn.metrics import pairwise_distances

            clusterer_class = globals()[experiment.result[word].config['clusterer']]
            clusterer_params = experiment.result[word].config['clusterer_params']
            clusterer = clusterer_class(**filter_dict(clusterer_params, clusterer_class))
            if isinstance(clusterer, (SubstsProbabilityBasedClusterer, SubstsFrequencyBasedClusterer)):
                metric_method = clusterer.distance
            else:
                metric_method = clusterer.affinity
            vectors = pairwise_distances(vectors, metric=metric_method)
            metric = 'precomputed'

        affinities_train = affinity.PerplexityBasedNN(
            vectors,
            perplexity=perplexity,
            metric=metric,
            n_jobs=8,
            random_state=42,
            verbose=False,
        )

        init_train = initialization.pca(vectors, random_state=42)

        embedding_train = TSNEEmbedding(
            init_train,
            affinities_train,
            negative_gradient_method="fft",
            n_jobs=8,
            verbose=True,
            callbacks=[
                lambda i, err, emb: embeddings.append(emb)
            ],
            callbacks_every_iters=1
        )

        embedding_train = embedding_train.optimize(n_iter=early_exaggeration_n_iter, exaggeration=early_exaggeration,
                                                   momentum=0.5)
        embedding_train.optimize(n_iter=n_iter, exaggeration=exaggeration, momentum=0.5)

    if method == 'pca':
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        embeddings = [pca.fit_transform(vectors)]

    for e in embeddings:
        e[:, 0] = (e[:, 0] - e[:, 0].min()) * (1000 / (e[:, 0].max() - e[:, 0].min()))
        e[:, 1] = (e[:, 1] - e[:, 1].min()) * (1000 / (e[:, 1].max() - e[:, 1].min()))

    return json.dumps([embs.tolist() for embs in embeddings])


@app.route('/experiment/<experiment_id>/dendrogram/<word>/data', methods=["GET"])
def get_dendrogram(experiment_id: str = None, word: str = None) -> str:
    experiment = get_dao().get_experiment_by_id(experiment_id)
    dataset = get_dao().get_dataset_by_id(experiment.dataset_id)
    dendrogram_data = experiment.result[word].dendrogram
    dendrogram_data['data'] = experiment.result[word].graph['data']
    dendrogram_data['distances'] = experiment.result[word].graph['distances']
    dendrogram_data['vectors'] = experiment.result[word].vectors.tolist()
    dendrogram_data['corpora_names'] = [corpus.name for corpus in dataset]
    dendrogram_data['cluster_pmis'] = experiment.result[word].cluster_pmis
    return json.dumps(dendrogram_data)


@app.route('/experiment/<experiment_id>/clustering/<word>/data', methods=["GET"])
def get_clustering_data(experiment_id: str = None, word: str = None) -> str:
    experiment = get_dao().get_experiment_by_id(experiment_id)
    dataset = get_dao().get_dataset_by_id(experiment.dataset_id)
    clustering_data = experiment.result[word].dendrogram
    clustering_data['data'] = experiment.result[word].graph['data']
    clustering_data['corpora_names'] = [corpus.name for corpus in dataset]
    clustering_data['cluster_pmis'] = experiment.result[word].cluster_pmis
    clustering_data['senses'] = experiment.result[word].senses
    clustering_data['sil_scores'] = experiment.result[word].sil_scores
    clustering_data['ari_scores'] = experiment.result[word].ari_scores
    clustering_data['n_clusters_list'] = experiment.result[word].n_clusters_list
    clustering_data['counts'] = get_dao().cache_setdefault(f"{experiment_id}-{word}-counts", lambda: get_counts(experiment_id, word))
    clustering_data['distances'] = get_dao().cache_setdefault(f"{experiment_id}-{word}-dist", lambda: get_distances(experiment_id, word))
    return json.dumps(clustering_data)


@app.route('/experiment/<experiment_id>/clustering/<word>/save-senses', methods=["POST"])
def save_senses(experiment_id: str = None, word: str = None) -> str:
    experiment = get_dao().get_experiment_by_id(experiment_id)
    experiment.result[word].senses = request.json
    get_dao().update_experiment(experiment)
    return json.dumps({'status': 'OK'})


def get_counts(experiment_id: str = None, word: str = None) -> List[Dict[str, Any]]:
    experiment = get_dao().get_experiment_by_id(experiment_id)
    dataset = get_dao().get_dataset_by_id(experiment.dataset_id)
    word_results = experiment.result[word]
    clusters = list(set(word_results.clusters))

    data = [
        {
            'label': dataset[0].name,
            'backgroundColor': 'red',
            'data': [word_results.clusters[:word_results.corpus_split_index].count(c) for c in clusters]
        }
    ]
    if len(word_results.clusters[word_results.corpus_split_index:]) > 0:
        data.append({
            'label': dataset[1].name, 'backgroundColor': 'blue',
            'data': [word_results.clusters[word_results.corpus_split_index:].count(c) for c in clusters]
        })

    return data


def get_distances(experiment_id: str = None, word: str = None) -> List[Dict[str, Any]]:
    experiment = get_dao().get_experiment_by_id(experiment_id)
    dataset = get_dao().get_dataset_by_id(experiment.dataset_id)
    clusterer_class = globals()[experiment.result[word].config['clusterer']]
    clusterer_params = experiment.result[word].config['clusterer_params']
    clusterer = clusterer_class(**filter_dict(clusterer_params, clusterer_class))
    if isinstance(clusterer, (SubstsProbabilityBasedClusterer, SubstsFrequencyBasedClusterer)):
        metric = clusterer.distance
    else:
        metric = clusterer.affinity

    distances = []

    split = experiment.result[word].corpus_split_index

    dists11 = cdist(experiment.result[word].vectors[:split], experiment.result[word].vectors[:split],
                    metric=metric)
    dists11 = dists11[np.triu_indices_from(dists11, k=1)]
    hist_range = (dists11.min(), dists11.max())

    if len(experiment.result[word].vectors[split:]) > 0:
        dists12 = cdist(experiment.result[word].vectors[:split], experiment.result[word].vectors[split:],
                        metric=metric)
        dists12 = dists12[np.triu_indices_from(dists12, k=1)]

        hist_range = (np.min([dists12.min(), hist_range[0]]), np.max([dists12.max(), hist_range[1]]))

        dists22 = cdist(experiment.result[word].vectors[split:], experiment.result[word].vectors[split:],
                        metric=metric)
        dists22 = dists22[np.triu_indices_from(dists22, k=1)]
        hist_range = (np.min([dists22.min(), hist_range[0]]), np.max([dists22.max(), hist_range[1]]))

    histogram, bins = np.histogram(dists11, range=hist_range, bins=30)
    distances.append({
        'label': '{} to {}'.format(dataset[0].name, dataset[0].name),
        'backgroundColor': 'red',
        'data': normalize(histogram.reshape(1, -1), axis=1, norm='l2')[0].tolist(),
        'x': bins.tolist()
    })

    if len(experiment.result[word].vectors[split:]) > 0:
        histogram, bins = np.histogram(dists12, range=hist_range, bins=30)
        distances.append({
            'label': '{} to {}'.format(dataset[0].name, dataset[1].name),
            'backgroundColor': 'blue',
            'data': normalize(histogram.reshape(1, -1), axis=1, norm='l2')[0].tolist(),
            'x': bins.tolist()
        })

        histogram, bins = np.histogram(dists22, range=hist_range, bins=30)
        distances.append({
            'label': '{} to {}'.format(dataset[1].name, dataset[1].name),
            'backgroundColor': 'green',
            'data': normalize(histogram.reshape(1, -1), axis=1, norm='l2')[0].tolist(),
            'x': bins.tolist()
        })

    return distances


@app.route('/experiment/<experiment_id>/examples/<word>/data', methods=["GET"])
def get_examples_samples(experiment_id: str, word: str) -> str:
    experiment = get_dao().get_experiment_by_id(experiment_id)
    dataset = get_dao().get_dataset_by_id(experiment.dataset_id)

    return json.dumps({
        'data': experiment.result[word].graph['data'],
        'vectors': experiment.result[word].vectors.tolist(),
        'corpora_names': [corpus.name for corpus in dataset],
        'cluster_pmis': experiment.result[word].cluster_pmis,
        'distances': get_dao().cache_setdefault(f"{experiment_id}-{word}-dist", lambda: get_distances(experiment_id, word)),
        'counts': get_dao().cache_setdefault(f"{experiment_id}-{word}-counts", lambda: get_counts(experiment_id, word)),
        'sil_scores': experiment.result[word].sil_scores,
        'ari_scores': experiment.result[word].ari_scores,
        'n_clusters_list': experiment.result[word].n_clusters_list,
    })


@app.route('/dataset', methods=["GET"])
def dataset_get() -> str:
    datasets = get_dao().get_datasets()
    return render_template(
        'layout.html',
        page='dataset',
        task=None,
        datasets=list(datasets),
        inouts=None,
        results=None,
        TASK_SUBST_WSI=TASK_SUBST_WSI,
        menu=[
            {'url_for': 'index', 'params': {}, 'text': 'Home'},
            {'url_for': 'dataset_get', 'params': {}, 'text': 'Upload dataset'}
        ]
    )


@app.route('/dataset', methods=["POST"])
def dataset_post() -> Response:
    input_directories = []
    input_filenames = []
    corpora_names = []
    uid = str(uuid.uuid4())
    for corpus in request.files:
        directory = os.path.join(UPLOAD_FOLDER, uid, corpus)
        os.makedirs(directory)
        filename = request.files[corpus].filename
        filepath = os.path.join(UPLOAD_FOLDER, uid, corpus, filename)
        if filename:
            request.files[corpus].save(filepath)
            input_directories.append(os.path.join(UPLOAD_FOLDER, uid, corpus))
            input_filenames.append(request.form['{}_input_filename'.format(corpus)])
            corpora_names.append(request.form['{}_name'.format(corpus)])
            with zipfile.ZipFile(filepath, 'r') as f:
                for fn in f.namelist():
                    fname = fn.encode('cp437').decode('utf-8')
                    extracted_path = Path(f.extract(fn, directory))
                    extracted_path.rename(directory + '/' + fname)

            os.remove(filepath)

    upload_dataset.delay(request.form['task'], request.form['new_dataset_id'], input_filenames, input_directories,
                         corpora_names)
    return redirect(url_for('dataset_get'))


@app.route('/solve/<task>', methods=["POST"])
def solve(task: str) -> Response:
    user_input = request.form

    params = {
        'AgglomerativeClusterer': {
            'n_clusters': lambda: list(map(int, user_input.getlist('n_clusters'))),
            'linkage': lambda: user_input.getlist('linkage'),
            'affinity': lambda: user_input.getlist('affinity'),
        },
        'SubstsProbabilityBasedClusterer': {
            'n_clusters': lambda: list(map(int, user_input.getlist('n_clusters'))),
            'linkage': lambda: user_input.getlist('linkage'),

        },
        'SubstsFrequencyBasedClusterer': {
            'n_clusters': lambda: list(map(int, user_input.getlist('n_clusters'))),
            'softmax_temperature': lambda: list(map(float, user_input.getlist('softmax_temperature')))
        },
        'SubstsTfidfVectorizer': {
            'analyzer': lambda: user_input.getlist('analyzer'),
            'min_df': lambda: list(map(float, user_input.get('min_df').split(','))),
            'max_df': lambda: list(map(float, user_input.get('max_df').split(','))),
            'topk': lambda: list(map(int, user_input.get('topk').split(','))),
        },
        'SubstsProbabilityBasedVectorizer': {
            'topk': lambda: list(map(int, user_input.get('topk').split(','))),
            'left_out_substs_prob_weight': lambda: list(map(float, user_input.getlist('left_out_substs_prob_weight'))),
        },
        'SubstsFrequencyBasedVectorizer': {
            'analyzer': lambda: user_input.getlist('analyzer'),
            'topk': lambda: list(map(int, user_input.get('topk').split(',')))
        }
    }

    configuration = {
        'clusterers': {
            user_input.get('clusterer'): {k: v() for k, v in params.get(user_input.get('clusterer')).items()}
        },
        'vectorizers': {
            user_input.get('vectorizer'): {k: v() for k, v in params.get(user_input.get('vectorizer')).items()}
        },
        'sample_size': int(user_input.get('sample_size')),
        'random_seed': int(user_input.get('random_seed')),
    }

    target_words = [w.strip() for w in user_input['target_words'].split('\r\n') if w]

    dataset_id = user_input.get('selected_dataset_id').strip()

    experiment = ExperimentFactory.from_task(task, dataset_id, target_words, configuration)

    get_dao().add_experiment(experiment)

    solve.delay(experiment._id)

    return redirect(url_for('index'))


@celery_app.task
def upload_dataset(task: str, dataset_id: str, input_filenames: List[str], input_directories: List[str],
                   corpora_names: List[str]):
    dataset_loader = DatasetLoaderFactory.from_task(get_dao(), task, dataset_id, input_filenames, input_directories,
                                                    corpora_names)
    dataset_loader.load()


@celery_app.task
def solve(experiment_id: str):
    experiment = get_dao().get_experiment_by_id(experiment_id)

    experiment.start_time = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S %Z")
    experiment.status = 'in_progress'

    get_dao().update_experiment(experiment)

    solver = SolverFactory.from_experiment(get_dao(), experiment)

    experiment.result = solver.solve(experiment.dataset_id)
    experiment.end_time = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S %Z")
    experiment.status = 'finished'

    get_dao().update_experiment(experiment)
