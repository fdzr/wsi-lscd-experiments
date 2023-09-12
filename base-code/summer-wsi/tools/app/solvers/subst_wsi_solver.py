from sklearn.metrics import adjusted_rand_score as ARI
from tools.app.interfaces import ISolver, IWSIVectorizer, IClusterer, IDao
from tools.app.data_objects import SubstWSIExperimentWordResult, SubstWSIExperiment, Sample, ClusterSearchResult
from tools.app.exceptions import DatasetNotFoundError, ClustererHasNoAffinityError
import itertools
from typing import Dict, Any, List, Tuple
import logging
import numpy as np
from substwsi.max_ari import preprocess_substs, get_nf_cnt
import pandas as pd
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import to_tree
from nltk.tokenize import RegexpTokenizer
from tools.app.clusterers.substs_probability_based_clusterer import SubstsProbabilityBasedClusterer
from tools.app.clusterers.substs_frequency_based_clusterer import SubstsFrequencyBasedClusterer
from scipy.spatial.distance import cosine  # this is important to have because of globals()[affinity]
import math
import random

logger = logging.getLogger(__name__)


class SubstWSISolver(ISolver):
    def __init__(self, dao: IDao, vectorizers: List[IWSIVectorizer], clusterers: List[IClusterer],
                 experiment: SubstWSIExperiment):
        self.experiment = experiment
        self.vectorizers = vectorizers
        self.clusterers = clusterers
        self.memory = {}
        self.clean_samples = None
        self.result_clusters = None
        self.dao = dao
        self.word_tokenizer = RegexpTokenizer(r'\w+')

    def solve(self, dataset_id: str) -> Dict[str, SubstWSIExperimentWordResult]:
        dataset = self.dao.get_dataset_by_id(dataset_id)

        if not dataset:
            raise DatasetNotFoundError()

        results = {}
        corpus1 = dataset[0].data

        if len(dataset) > 1:
            corpus2 = dataset[1].data
        else:
            corpus2 = []

        if not len(self.experiment.target_words):
            self.experiment.target_words = set([s.lemma for s in (corpus1 + corpus2)])

        for word in self.experiment.target_words:
            result, \
            corpus_split_index, \
            word_samples, \
            clusterer, \
            vectorizer, \
            all_vectors, \
            score, \
            ari_score, \
            max_ari_score, \
            sil_scores, \
            ari_scores, \
            n_clusters_list = self.grid_search_best_cluster(word, corpus1, corpus2)
            results[word] = self._prepare_word_results(
                result,
                corpus_split_index,
                word_samples,
                clusterer,
                vectorizer,
                all_vectors,
                score,
                ari_score,
                max_ari_score,
                sil_scores,
                ari_scores,
                n_clusters_list
            )
            results[word].score = score

        return results

    def grid_search_best_cluster(self, word: str, corpus1: List[Sample], corpus2: List[Sample]) -> \
            Tuple[
                ClusterSearchResult,
                int,
                List[Sample],
                IClusterer,
                IWSIVectorizer,
                np.ndarray,
                float,
                float,
                float,
                List[float],
                List[float],
                List[int],
            ]:

        def fix_labels(s: Sample):
            s.label_id = s.label_id if not math.isnan(s.label_id) else -1
            return s

        corpus1 = list(map(fix_labels, corpus1))
        corpus2 = list(map(fix_labels, corpus2))

        has_gold_labels = len(list(filter(lambda s: s.label_id != -1, corpus1 + corpus2))) > 0

        best_score = None
        best_corpus_split_index = None
        best_result = None
        best_word_samples = None
        best_clusterer = None
        best_vectorizer = None
        best_all_nonzero_vectors = None
        best_ari_score = None
        max_ari_score = float("-inf") if has_gold_labels else None

        sil_scores = []
        ari_scores = []
        n_clusters_list = []

        for clusterer, vectorizer in itertools.product(self.clusterers, self.vectorizers):
            all_nonzero_vectors, \
            word_samples, \
            corpus_split_index = self.memory.setdefault(
                (vectorizer, word), self._get_vectors(vectorizer, corpus1, corpus2, word)
            )

            if len(all_nonzero_vectors) == 0:
                continue

            result = clusterer.cluster(all_nonzero_vectors)

            if hasattr(clusterer, 'n_clusters'):
                n_clusters_list.append(clusterer.n_clusters)

            if isinstance(clusterer, (SubstsProbabilityBasedClusterer, SubstsFrequencyBasedClusterer)):
                sil_score = silhouette_score(all_nonzero_vectors, result.clusters, metric=clusterer.distance)
            elif hasattr(clusterer, 'affinity'):
                sil_score = silhouette_score(all_nonzero_vectors, result.clusters, metric=clusterer.affinity)
            else:
                raise ClustererHasNoAffinityError()

            if has_gold_labels:
                ari_score = ARI(
                    [s.label_id for _, s in enumerate(word_samples) if s.label_id != -1],
                    [result.clusters[i] for i, s in enumerate(word_samples) if s.label_id != -1]
                )
                ari_scores.append(ari_score)
                if ari_score > max_ari_score:
                    max_ari_score = ari_score
            else:
                ari_score = None

            sil_scores.append(sil_score)

            if best_score is None or best_score < sil_score:
                best_score = sil_score
                best_word_samples = word_samples
                best_corpus_split_index = corpus_split_index
                best_result = result
                best_clusterer = clusterer
                best_vectorizer = vectorizer
                best_all_nonzero_vectors = all_nonzero_vectors
                best_ari_score = ari_score

        return best_result, \
               best_corpus_split_index, \
               best_word_samples, \
               best_clusterer, \
               best_vectorizer, \
               best_all_nonzero_vectors, \
               best_score, \
               best_ari_score, \
               max_ari_score, \
               sil_scores, \
               ari_scores, \
               n_clusters_list

    def _get_vectors(
            self, vectorizer: IWSIVectorizer,
            corpus1: List[Sample],
            corpus2: List[Sample],
            word: str
    ) -> Tuple[
        np.ndarray,
        List[Sample],
        int
    ]:
        random.seed(self.experiment.config.random_seed)
        word_samples_filtered1 = [s for s in corpus1 if s.lemma == word]
        word_samples_filtered2 = [s for s in corpus2 if s.lemma == word]
        word_samples1 = self._get_preprocessed_samples(word, random.sample(word_samples_filtered1,
                                                                           min(self.experiment.config.sample_size,
                                                                               len(word_samples_filtered1))),
                                                       vectorizer.topk)
        word_samples2 = self._get_preprocessed_samples(word, random.sample(word_samples_filtered2,
                                                                           min(self.experiment.config.sample_size,
                                                                               len(word_samples_filtered2))),
                                                       vectorizer.topk)

        all_vectors = vectorizer.vectorize(word_samples1 + word_samples2)

        vectors1 = all_vectors[:len(word_samples1)]
        vectors2 = all_vectors[len(word_samples1):]

        nonzero_vecs_mask1 = ~np.all(np.array(vectors1) < 1e-6, axis=1)
        nonzero_vecs_mask2 = ~np.all(np.array(vectors2) < 1e-6, axis=1) if len(vectors2) > 0 else np.array([])
        nonzero_vecs_indices1 = np.where(nonzero_vecs_mask1)[0]
        nonzero_vecs_indices2 = np.where(nonzero_vecs_mask2)[0]

        all_nonzero_vectors = np.concatenate(
            [vectors1[nonzero_vecs_indices1], vectors2[nonzero_vecs_indices2]], axis=0) if len(vectors2) > 0 else \
            vectors1[nonzero_vecs_indices1]

        clean_samples1 = [word_samples1[i] for i in list(nonzero_vecs_indices1)]
        clean_samples2 = [word_samples2[i] for i in list(nonzero_vecs_indices2)]

        return all_nonzero_vectors, clean_samples1 + clean_samples2, len(nonzero_vecs_indices1)

    def _prepare_word_results(
            self,
            result: ClusterSearchResult,
            corpus_split_index: int,
            word_samples: List[Sample],
            clusterer: IClusterer,
            vectorizer: IWSIVectorizer,
            vectors: np.ndarray,
            score: float,
            ari_score: float,
            max_ari_score: float,
            sil_scores: List[float],
            ari_scores: List[float],
            n_clusters_list: List[int]
    ) -> SubstWSIExperimentWordResult:
        clustering_tree = self.get_clustering_tree(clusterer)

        self.label_tree(word_samples, corpus_split_index, result.clusters, clustering_tree)

        dendrogram = {'tree': clustering_tree}

        graph = self.get_graph(word_samples, result.clusters, clusterer, vectors)

        cluster_pmis = self._get_cluster_pmis(result.clusters, corpus_split_index)

        result = SubstWSIExperimentWordResult(
            ari_score=ari_score,
            max_ari_score=max_ari_score,
            cluster_pmis=cluster_pmis,
            dendrogram=dendrogram,
            corpus_split_index=corpus_split_index,
            graph=graph,
            vectors=vectors,
            sil_scores=sil_scores,
            ari_scores=ari_scores,
            n_clusters_list=n_clusters_list,
            vector_features=vectorizer._features if hasattr(vectorizer, '_features') else None,
            clusters=result.clusters,
            score=score,
            config={
                'vectorizer': type(vectorizer).__name__,
                'clusterer': type(clusterer).__name__,
                'clusterer_params': {i: clusterer.__dict__[i] for i in clusterer.__dict__.keys() if i[:1] != '_'},
                'vectorizer_params': {i: vectorizer.__dict__[i] for i in vectorizer.__dict__.keys() if i[:1] != '_'}
            }
        )

        return result

    def get_graph(self, samples: List[Sample], clusters: List[int], clusterer: IClusterer, vecs: np.ndarray) -> Dict[
        str, Any]:
        graph = {'elements': {'nodes': [], 'edges': []}, 'distances': {}, 'data': {}}

        for i in range(len(samples)):
            sample = samples[i]
            cluster = clusters[i]
            node_id = i
            for node in graph['elements']['nodes']:
                edge_id = "e_{}_{}".format(
                    max(node_id, node['data']['id']),
                    min(node_id, node['data']['id'])
                )
                if isinstance(clusterer, (SubstsProbabilityBasedClusterer, SubstsFrequencyBasedClusterer)):
                    graph['distances'][edge_id] = clusterer.distance(vecs[node_id], vecs[node['data']['id']])
                elif hasattr(clusterer, 'affinity'):
                    graph['distances'][edge_id] = globals()[clusterer.affinity](vecs[node_id], vecs[node['data']['id']])
                else:
                    raise ClustererHasNoAffinityError()

            graph['data'][node_id] = {
                'context': sample.context,
                'substs': [s[1].strip() for s in sample.substs],
                'substs_probs': [s[0] for s in sample.substs],
                'corpus': sample.corpus,
                'begin': sample.begin,
                'end': sample.end,
                'cluster': int(cluster)
            }

            graph['elements']['nodes'].append({
                'data': {
                    'id': node_id,
                    'cluster': int(cluster),
                    'corpus': sample.corpus,
                    'label_id': int(sample.label_id)
                }
            })

        return graph

    def get_clustering_tree(self, model: IClusterer) -> Any:
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        return self.build_clustering_tree(linkage_matrix)

    def build_clustering_tree(self, linkage_matrix: np.ndarray) -> Any:
        tree = to_tree(linkage_matrix, rd=False)

        root_node = dict(children=[], name="Root1")
        self.add_tree_node(tree, root_node)

        return root_node["children"][0]

    def add_tree_node(self, node: Any, parent: Dict[str, Any]) -> None:
        # First create the new node and append it to its parent's children
        new_node = dict(id=int(node.id), children=[], dist=node.dist)
        parent["children"].append(new_node)

        # Recursively add the current node's children
        if node.left:
            self.add_tree_node(node.left, new_node)
        if node.right:
            self.add_tree_node(node.right, new_node)

    def label_tree(self, word_samples: List[Sample], corpus_split_index: int, result_clusters: List[int],
                   node: Dict[str, Any]):
        if len(node["children"]) == 0:

            node['substs'] = [subst[1] for subst in word_samples[node["id"]].substs]
            node['substs_probs'] = [subst[0] for subst in word_samples[node["id"]].substs]
            node["context"] = word_samples[node["id"]].context
            node["lemma"] = word_samples[node["id"]].lemma
            node["cluster"] = int(result_clusters[node["id"]])
            node["corpus"] = word_samples[node["id"]].corpus
            node["label_id"] = word_samples[node["id"]].label_id
            node["begin"] = word_samples[node["id"]].begin
            node["end"] = word_samples[node["id"]].end
            node["corpus"] = 1 if node['id'] >= corpus_split_index else 0

            sample = word_samples[node["id"]]
            word_in_context = node["context"][sample.begin:sample.end]
            words = self.word_tokenizer.tokenize(node["context"])
            node['word_in_context'] = word_in_context
            try:
                target_word_idx = words.index(word_in_context)
                node["name"] = '...' + ' '.join(words[max(0, target_word_idx - 3):target_word_idx + 3]) + '...'
            except:
                node["name"] = '...' + word_in_context + '...'

        else:
            [self.label_tree(word_samples, corpus_split_index, result_clusters, child) for child in node['children']]
            node["substs"] = None
            node["name"] = None
            node["context"] = None
            node["cluster"] = None
            node["label_id"] = None
            node["begin"] = None
            node["end"] = None

        node["id"] = str(node["id"])

    def _get_preprocessed_samples(self, word: str, samples: List[Sample], topk: int) -> List[Sample]:
        # convert samples list to df for preprocessing
        df = self._sample_list_to_df(samples)

        nf_cnt = get_nf_cnt(df['substs_probs'])
        df['substs_probs'] = df.substs_probs.apply(
            lambda r: list(
                zip(
                    [ps[0] for ps in sorted(r, key=lambda x: x[0], reverse=True)[:topk]],
                    preprocess_substs(sorted(r, key=lambda x: x[0], reverse=True)[:topk], nf_cnt=nf_cnt, lemmatize=True,
                                      exclude_lemmas=[word])
                )
            )
        )

        # back to samples list and return
        return self._to_samples(df)

    @staticmethod
    def _to_samples(dataset_df: pd.DataFrame):
        def convert_row_to_sample(row):
            begin, end = row['positions']
            substs_probs = row['substs_probs']
            substs_probs_converted = []
            for prob, word in substs_probs:
                substs_probs_converted.append((float(prob), str(word)))

            return Sample(
                context=str(row['context']),
                begin=int(begin),
                end=int(end),
                lemma=str(row['word']),
                substs=substs_probs_converted,
                label_id=row['label_id'],
                corpus=row['corpus']
            )

        return [convert_row_to_sample(row) for _, row in dataset_df.iterrows()]

    @staticmethod
    def _sample_list_to_df(samples: List[Sample]):
        data = []
        for sample in samples:
            data.append(
                [
                    sample.context,
                    sample.lemma,
                    sample.substs,
                    (sample.begin, sample.end),
                    sample.label_id,
                    sample.corpus,
                ]
            )
        return pd.DataFrame(data=data, columns=['context', 'word', 'substs_probs', 'positions', 'label_id', 'corpus'])

    def _get_cluster_pmis(self, clusters: List[int], corpus_split_index: int) -> List[List[float]]:
        unique_clusters = set(clusters)
        result = []
        counts1 = [clusters[:corpus_split_index].count(cluster) for cluster in unique_clusters]
        counts2 = [clusters[corpus_split_index:].count(cluster) for cluster in unique_clusters]

        count_corpus1 = len(clusters[:corpus_split_index])
        p_corpus1 = count_corpus1 / len(clusters)

        count_corpus2 = len(clusters[corpus_split_index:])
        p_corpus2 = count_corpus2 / len(clusters)

        for i in range(len(counts1)):
            pmis = []
            if counts1[i] == 0:
                pmis.append(-9999)
            else:
                p_corpus1_cluster = counts1[i] / (counts1[i] + counts2[i])
                pmis.append(self._pmi(p_corpus1_cluster, p_corpus1))

            if counts2[i] == 0:
                pmis.append(-9999)
            else:
                p_corpus2_cluster = counts2[i] / (counts1[i] + counts2[i])
                pmis.append(self._pmi(p_corpus2_cluster, p_corpus2))
            result.append(pmis)
        return result

    def _pmi(self, p1: float, p2: float):
        return np.log2(p1) - np.log2(p2)
