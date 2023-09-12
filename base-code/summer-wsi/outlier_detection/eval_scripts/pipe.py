from typing import Dict, Any, List, Tuple, Union, Iterable
from dataclasses import dataclass
from sklearn.preprocessing import normalize
import numpy as np
import warnings
import os
import sys
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import scipy as sp
import scipy.sparse
from tqdm import tqdm
import torch
import nltk
from sklearn.neighbors import BallTree
from transformers import XLMRobertaConfig, RobertaModel, AutoTokenizer, AutoModelForMaskedLM, XLMRobertaModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_curve
import maha
import cosine
from multilang_wsi_evaluation.interfaces import IWSIVectorizer
from utils import SampleNSD
import re

def mean_pool(embds):
    """pooling function"""
    return embds.mean(axis=0)


def first_pool(embds):
    """pooling function"""
    return embds[0]


class Vectorizer_NSD(IWSIVectorizer):  # NSD - Novel Sense Detection

    def __init__(self, model_str='xlm-roberta-large', pretrained_path=None, pooling=mean_pool, layer=12):
        if pretrained_path:
            self.model = XLMRobertaModel.from_pretrained(model_str)
            self.model.load_state_dict(torch.load(pretrained_path))
            print('loaded!')
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(model_str)
            self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_str)
        self.layer = layer
        self.model_str = model_str
        if pooling == 'mean':
            self.pooling = mean_pool
        elif pooling == 'first':
            self.pooling = first_pool
        else:
            self.pooling = mean_pool
        if pretrained_path:
            self.pretrained_path = os.path.abspath(__file__ + "/../../../") + '/' + pretrained_path

    @staticmethod
    def get_ix(word, toklist):
        curr_word = word
        ix = []
        for i, t in enumerate(toklist):
            if curr_word.startswith(t.lstrip('▁')):
                curr_word = curr_word[len(t.lstrip('▁')):]
                ix.append(i)
            elif len(curr_word) <= len(t.lstrip('▁')):
                if t.lstrip('▁').startswith(curr_word):  ## added this closure after bts-rnc
                    curr_word = ''
                    ix.append(i)
                elif t.lstrip('▁').startswith(word):  ## added this closure after bts-rnc
                    curr_word = ''
                    ix = [i]
                elif word.startswith(t.lstrip('▁')):  ## added this closure after bts-rnc
                    curr_word = word[len(t.lstrip('▁')):]
                    ix = [i]
            else:
                curr_word = word
                ix = []
                if curr_word.startswith(t.lstrip('▁')):
                    curr_word = curr_word[len(t.lstrip('▁')):]
                    ix.append(i)
            if not curr_word:
                return ix
        return ix


    def fit(self, samples):
        pass


    def predict(self, corpora: List[SampleNSD]):
        out = []
        for samp in tqdm(corpora):
            encoded_input = self.tokenizer(samp.context, return_tensors='pt')
            output = self.model(**encoded_input, output_hidden_states=True)
            toklist = list(map(self.tokenizer.convert_ids_to_tokens, encoded_input.input_ids))[0]
            try:
                ix = self.get_ix(samp.context[samp.begin:samp.end], toklist)
                emb = self.pooling((output.hidden_states[self.layer].squeeze()[ix]).detach().numpy())
            except Exception as e:
                print(e, ix, samp)
            out.append(emb)
        out = np.array(out)
        return out


class Context_vectorizer(IWSIVectorizer):

    def __init__(self, radius=None):
        self.radius = radius

    def fit(self, samples):
        pass

    def predict(self, corpora: List[SampleNSD]):
        if not self.radius:
            samples = []
            for samp in tqdm(corpora):
                samples.append(samp.context)
        else:
            samples = []
            for samp in tqdm(corpora):
                temp = ''
                target = samp.context[samp.begin:samp.end]
                t = samp.context
                t = re.sub('[,.!?:]', '', t)
                tocs = nltk.tokenize.word_tokenize(t)
                try:
                    ix = tocs.index(target)
                except:
                    for j, token in enumerate(tocs):
                        if target in token:
                            ix = j
                try:
                    for t in range(self.radius):
                        temp += ' ' + (tocs[ix + t])
                        temp += ' ' + (tocs[ix - t])
                except:
                    pass
                samples.append(temp)
        vectorizer = CountVectorizer()  # stop_words='english')
        X = vectorizer.fit_transform(samples)
        return X


class Detector_pipe():

    def __init__(self, dist_func='euclidean', threshold=1.0):
        self.dist_func = dist_func
        self.threshold = threshold

    def get_predict(self, train, x, tree, threshold=1.0):
        """ get label wether given x embedding is unknown (1) or known (0) for given train embedding set """
        x = x.reshape(1, -1)
        d1, ix = tree.query(x, k=1)  # obtain distance d[xt]: x to NN in train
        d2, ix2 = tree.query([train[ix.item()]],
                             k=2)  # obtain distance d[tt']: distance between t and its NN in train, k=2 as there is also distance to itself
        assert d2.tolist()[0][0] == 0, "Vector's distance to itself not zero"
        ratio = (d1 / d2.tolist()[0][1])
        res = ratio[0][0] > threshold  # get ratio to compare to threshold tetta. 1 if outlier, 0 otherwise.

        return int(res)


    def detect_outliers(self, train, test):
        labels = []
        tree = BallTree(train, metric=self.dist_func, leaf_size=40)  # init BallTree
        for t_ in test:  # sample lvl
            pred = self.get_predict(train, t_, tree, self.threshold)
            labels.append(pred)
        return labels



class NSD_pipe():

    def __init__(self, dist_func='euclidean', threshold=1.0, vectorizer=None, norm=None):
        self.dist_func = dist_func
        self.vectorizer = vectorizer
        self.threshold = threshold
        self.detector = Detector_pipe(dist_func, threshold=threshold)
        self.norm = norm
        print('pipe norm:', self.norm)

    def get_sample_ix(self, sample_list):
        out = []
        for el in sample_list:
            out.append(self.samples.index(el))
        return out

    def get_Erk_ratio(self, train, test):
        tree = BallTree(train, metric=self.dist_func, leaf_size=40)
        out = []
        for t_ in test:
            x = t_.reshape(1, -1)
            d1, ix = tree.query(x, k=1)  #
            d2, ix2 = tree.query([train[ix.item()]], k=2)
            assert d2.tolist()[0][0] == 0, "Vector's distance to itself not zero"
            ratio = (d1 / d2.tolist()[0][1])
            if ratio != np.inf and not np.isnan(ratio):
                out.append(ratio[0][0])
            else:
                out.append(1.0)
        return out

    def obtain_scores(self, train, test):
        xx = self.get_sample_ix(train)
        yy = self.get_sample_ix(test)
        if isinstance(self.embeds, scipy.sparse.csr.csr_matrix):
            self.embeds = self.embeds.toarray()
        train = self.embeds[[xx]]
        test = self.embeds[[yy]]
        scores = []
        if self.dist_func == 'mahalanobis':
            m = maha.Maha()
            m.fit(train)
            for i, el in enumerate(test):
                scores.append(m.predict(el))
        elif self.dist_func == 'cosine':
            cos = cosine.Cos()
            cos.fit(train)
            for i, el in enumerate(test):
                scores.append(cos.predict(el))
        else:
            scores.extend(self.get_Erk_ratio(train, test))

        return scores

    def predict_threshold_based(self, all_labels, decision_scores, min_precision):
        pr, rec, thr = precision_recall_curve(all_labels, decision_scores)
        optimal_thr = thr[np.where(pr == min(x for x in pr if x > min_precision))]
        preds = list(map(int, (decision_scores > optimal_thr)))
        return preds

    def predict_threshold(self, decision_scores, threshold):
        preds = list(map(int, (np.array(decision_scores) > threshold)))
        return preds

    def apply_norm(self):
        if self.norm:
            self.embeds = normalize(self.embeds, norm=self.norm, axis=1)

    def fit(self):
        self.embeds = self.vectorizer.predict(self.samples)
        if isinstance(self.embeds, scipy.sparse.csr.csr_matrix):
            self.embeds = self.embeds.toarray()
        else:
            self.embeds = np.array(self.embeds)
        return self.embeds

    def predict(self, train: List[SampleNSD], test: List[SampleNSD]):
        if self.dist_func in ['mahalanobis', 'cosine']:
            scores = self.obtain_scores(train, test)
            preds = self.predict_threshold(scores, self.threshold)
            return preds
        xx = self.get_sample_ix(train)
        yy = self.get_sample_ix(test)
        # print(xx)
        if isinstance(self.embeds, scipy.sparse.csr.csr_matrix):
            self.embeds = self.embeds.toarray()
        emb_x = self.embeds[[xx]]
        emb_y = self.embeds[[yy]]

        return self.detector.detect_outliers(emb_x, emb_y)
