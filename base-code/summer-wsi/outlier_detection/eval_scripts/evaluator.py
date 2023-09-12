import os
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, precision_recall_curve
from collections import defaultdict
import pipe as pipe  # outlier_detection.
import graph as graph
import hydra
import scipy.sparse
from hydra.utils import instantiate
from omegaconf import DictConfig
from utils import get_samples_from_df, get_df_from_samples
from sklearn.metrics import precision_recall_curve, average_precision_score
import sys
import matplotlib.pyplot as plt
import random
import warnings

warnings.filterwarnings("ignore")


class NSD_evaluator():

    def __init__(self, path='vectors/framenet17_gold.csv', eval_type='synchronic',  # or diachronic
                 dist_func='euclidean', vectorizer=None, threshold=1.0, visualize=False, window='subword',
                 norm=None, use_decision=False, min_precision=0.5):
        self.visualize = visualize
        self.path = path  # self.dir +'/'+
        self.eval_type = eval_type
        self.samples = pd.read_csv(self.path, sep='\t')
        self.window = window
        self.dist_func = dist_func
        self.norm = norm
        self.pipe = pipe.NSD_pipe(vectorizer=vectorizer, dist_func=dist_func, threshold=threshold, norm=norm)
        self.use_decision = use_decision
        self.min_precision = min_precision
        print('use_decision:', self.use_decision)
        if not os.path.exists('../../../../../outlier_detection/' + 'vectors'):  # vectors results
            os.makedirs('../../../../../outlier_detection/' + 'vectors')
        if not os.path.exists('../../../../../outlier_detection/' + 'results'):
            os.makedirs('../../../../../outlier_detection/' + 'results')

    def get_splits(self):
        resulting_splits = []
        words = self.samples.word.unique()
        if self.eval_type == 'diachronic':
            train_index = [1, 2]
            train_index1 = [2, 1]
            for word in words:
                temp_splits = []
                data = self.samples[self.samples.word == word]
                for k in range(len(train_index)):
                    s1 = set(data[data['grouping'] == train_index[k]]['gold_sense_id'].unique())
                    s2 = set(data[data['grouping'] == train_index1[k]]['gold_sense_id'].unique())
                    ix_train = get_samples_from_df(data[data['grouping'] == train_index[k]], window=self.window)
                    ix_test = get_samples_from_df(data[data['grouping'] == train_index1[k]], window=self.window)
                    embeds1 = data[data['grouping'] == train_index[k]]['pooled_embeds']
                    embeds2 = data[data['grouping'] == train_index1[k]]['pooled_embeds']
                    labels = [0 for _ in range(len(ix_test))]
                    senses = data[data['grouping'] == train_index1[k]]['gold_sense_id']
                    if s2 - s1:
                        senses_test = [x.gold_sense_id for k, x in data[data['grouping'] == train_index1[k]].iterrows()]
                        labels = [1 if t in s2 - s1 else 0 for t in senses_test]
                    temp_splits.append({'train': ix_train, 'test': ix_test, 'labels': labels, 'embeds_train': embeds1,
                                        'embeds_test': embeds2, 'senses': senses, 'word': word})
                resulting_splits.append(temp_splits)
        elif self.eval_type == 'equal_test':
            train_index = [1, 2]
            train_index1 = [2, 1]
            for word in words:
                temp_splits = []
                data = self.samples[self.samples.word == word]
                meanings = data.gold_sense_id.unique()
                for i in meanings:
                    for k in range(len(train_index)):
                        test1 = data[(data['grouping'] == train_index1[k]) & (data['gold_sense_id'] != i)]
                        samp = test1.sample(frac=0.5)
                        train = get_samples_from_df(samp, window=self.window)
                        if len(train) < 2:
                            continue
                        test = get_samples_from_df(data[data['grouping'] == train_index1[k]].drop(samp.index),
                                                   window=self.window)
                        lbls = [1 if x.gold_sense_id == i else 0 for x in test]
                        train1 = data[(data['grouping'] == train_index[k]) & (data['gold_sense_id'] != i)]
                        if len(train1) < 2:
                            continue
                        train1 = get_samples_from_df(train1.sample(min(len(train), len(train1))), window=self.window)
                        temp_splits.append({'train': train, 'test': test, 'labels': lbls, 'case': 'synchronic'})
                        temp_splits.append({'train': train1, 'test': test, 'labels': lbls, 'case': 'diachronic'})
                resulting_splits.append(temp_splits)
        elif self.eval_type == 'synchronic':
            splits = 3 if len(self.samples)/len(words) < 20 else 5
            for word in words:
                temp_splits = []
                data = self.samples[self.samples.word == word]
                meanings = data.gold_sense_id.unique()
                for i in meanings:  # repeat for each meaning of given lemma
                    train_for_cv = []
                    test = []
                    curr_unknown = i  # label some meaning as unknown
                    for j in meanings:
                        if j == curr_unknown:  # put all context_id's (each id corresponds to one sentence) for unknown sense in test set
                            for id_ in data[data['gold_sense_id'] == j].index:
                                test.append(id_)
                        else:
                            for id_ in data[
                                data['gold_sense_id'] == j].index:  # rest lemmas stored in a list train_for_cv
                                train_for_cv.append(id_)
                    kf = KFold(n_splits=splits)  # use KFold for cv-splits
                    if len(train_for_cv) < splits:
                        #print('small sample for: ', word)
                        continue
                    for to_train, to_test in kf.split(
                            train_for_cv):  # get cv_folds(oroginally 5) splits. Bigger split goes to train, smaller to test data
                        test_ = test.copy()
                        test_.extend(list(np.array(train_for_cv)[[tuple(to_test)]]))
                        labels = np.ones(len(test_))  # labeling: 1 means "unknown" sense
                        labels[-len(to_test):] = 0  # label 0 for extra added known senses, obtained through K-fold
                        ix_train = list(np.array(train_for_cv)[[tuple(to_train)]])
                        train_samples = data.loc[ix_train]
                        test_samples = data.loc[test_]
                        embeds1 = train_samples['pooled_embeds']
                        embeds2 = test_samples['pooled_embeds']
                        senses = test_samples['gold_sense_id']
                        temp_splits.append({'train': get_samples_from_df(train_samples, window=self.window),
                                            'test': get_samples_from_df(test_samples, window=self.window),
                                            'labels': labels, 'word': word,
                                            'embeds_train': embeds1, 'embeds_test': embeds2, 'senses': senses})
                resulting_splits.append(temp_splits)
        return resulting_splits

    def get_stats(self, pred, labels):
        """ calculate various statistics"""
        out = []
        pr = precision_score(labels, pred)
        out.append(pr)
        rec = recall_score(labels, pred)
        out.append(rec)
        pos_rate = sum(pred) / len(pred)
        out.append(pos_rate)
        actual_pos_rate = sum(labels) / len(labels)
        out.append(actual_pos_rate)
        return out

    def prepare_dict_for_viz(self, split):
        train = split['train']
        test = split['test']
        tr_pooled_embeds = split['embeds_train'].values
        te_pooled_embeds = split['embeds_test'].values

        tr_out = pd.DataFrame(columns=['context_id', 'word', 'positions', 'context', 'gold_sense_id'])
        for i, sample in enumerate(train):
            tr_out.loc[i] = [i, sample.lemma, str(sample.begin) + '-' + str(sample.end), sample.context,
                             sample.gold_sense_id]
        tr_out['pooled_embeds'] = tr_pooled_embeds
        te_out = pd.DataFrame(columns=['context_id', 'word', 'positions', 'context', 'gold_sense_id'])
        for i, sample in enumerate(test):
            te_out.loc[i] = [i, sample.lemma, str(sample.begin) + '-' + str(sample.end), sample.context,
                             sample.gold_sense_id]
        te_out['pooled_embeds'] = te_pooled_embeds
        return tr_out, te_out

    @staticmethod
    def plot_pre_rec(all_labels, decisions, path):
        pr, rec, thr = precision_recall_curve(all_labels, decisions)
        stat = average_precision_score(all_labels, decisions)
        linestyle = '-'
        color = 'r'
        plt.ylabel('precision')
        plt.xlabel('recall')
        warn = ''
        if len(set(all_labels)) <= 1:
            warn = 'Corner case! All labels were equal!'
        plt.plot(rec, pr, color=color, linestyle=linestyle,  label=f"precision-recall curve for experiment, score: {round(stat,4)}")
        xx = [random.random() for _ in range(len(all_labels))]
        pr, rec, thr = precision_recall_curve(all_labels, xx)
        stat = average_precision_score(all_labels, xx)
        plt.plot(rec, pr, color='purple', label=f"random values, score: {round(stat,4)}")
        plt.title(warn)
        plt.legend() #bbox_to_anchor=(1.1, 1.05))
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path+'/'+'precision_recall_curve.png')
        plt.close('all')

    def evaluate(self, path, preload=None, save_embeds=None):
        if preload:
            if preload.endswith('.npz'):
                embeds = scipy.sparse.load_npz(preload)
                embeds = embeds.toarray()
            else:
                embeds = np.load(preload)
            self.pipe.embeds = embeds
            self.pipe.samples = get_samples_from_df(self.samples, window=self.window)
        else:
            self.pipe.samples = get_samples_from_df(self.samples, window=self.window)
            print('calculating embeddings..')
            embeds = self.pipe.fit()
        if save_embeds:
            np.save(save_embeds, embeds)
        self.pipe.apply_norm()
        self.dict_for_viz = defaultdict(lambda: defaultdict(list))
        self.samples['pooled_embeds'] = list(self.pipe.embeds)
        splits = self.get_splits()
        true_labels = [[x['labels'] for x in y] for y in splits]
        labels = []
        stats = defaultdict(list)
        out = pd.DataFrame(columns=['word', 'split', 'precision', 'recall', 'pos_rate', 'actual_pos_rate'])
        decisions_per_split = defaultdict(lambda: defaultdict(list))
        labels_per_split = defaultdict(lambda: defaultdict(list))
        if self.use_decision:
            decisions = []
            all_labels = []
            decisions_sync = []
            decisions_diachr = []
            for i, word in tqdm(enumerate(splits)):
                for j, split in enumerate(word):
                    if 'case' in split:
                        if split['case'] == 'diachronic':
                            scores = self.pipe.obtain_scores(split['train'], split['test'])
                            decisions_diachr.extend(scores)
                            all_labels.extend(split['labels'])
                        elif split['case'] == 'synchronic':
                            scores = self.pipe.obtain_scores(split['train'], split['test'])
                            decisions_sync.extend(scores)
                    else:
                        scores = self.pipe.obtain_scores(split['train'], split['test'])
                        decisions_per_split[split['word']][j].extend(scores)
                        labels_per_split[split['word']][j].extend(split['labels'])
                        decisions.extend(scores)
                        all_labels.extend(split['labels'])
                       # pred = self.pipe.predict(split['train'], split['test'])
                        if self.visualize:
                            tr_samp, test_samp = self.prepare_dict_for_viz(split)
                            self.dict_for_viz[self.samples.word.unique()[i]][j] = (split['labels'], tr_samp, test_samp)
            #  if self.eval_type != 'equal_test':
            #      all_preds = self.pipe.predict_threshold_based(all_labels, decisions, min_precision=self.min_precision)
            #      stats = self.get_stats(all_preds, all_labels)
            #  print('stats for given precision:', stats)
            for word in decisions_per_split.keys():
                for split in decisions_per_split[word].keys():
                    self.plot_pre_rec(labels_per_split[word][split], decisions_per_split[word][split], path=os.getcwd() + f'/pr/{word}/{split}/')
            np.save(os.getcwd() + '/all_labels.npy', np.array(all_labels))  # np array with labels for dataset
            if len(decisions_sync) > 0:
                np.save(os.getcwd() + f'/decisions_{self.norm}_{self.dist_func}_dia.npy', np.array(decisions_diachr))
                np.save(os.getcwd() + f'/decisions_{self.norm}_{self.dist_func}_syn.npy', np.array(decisions_sync))
            else:
                np.save(os.getcwd() + f'/decisions_{self.norm}_{self.dist_func}.npy', np.array(decisions))
                self.plot_pre_rec(all_labels, decisions, path=os.getcwd()+'/graphs/')
                stat = average_precision_score(all_labels, decisions)
                df = pd.DataFrame({"Average precision": [round(stat, 4)]})
                df.to_csv(os.getcwd() + '/stats.csv')

        else:
            count = 0
            for i, word in tqdm(enumerate(splits)):
                temp = []
                for j, split in enumerate(word):
                    pred = self.pipe.predict(split['train'], split['test'])
                    temp.append(pred)
                    if self.visualize:
                        tr_samp, test_samp = self.prepare_dict_for_viz(split)
                        self.dict_for_viz[self.samples.word.unique()[i]][j] = (pred, tr_samp,
                                                                               test_samp)  # get_df_from_samples(split['train']), get_df_from_samples(split['test']))
                    st = self.get_stats(pred, true_labels[i][j])
                    stats[self.samples.word.unique()[i]].append(st)
                    out.loc[count] = [self.samples.word.unique()[i], str(j), *st]
                    count += 1
                labels.append(temp)
            out.to_csv(path + f'/stats.csv')
        return stats

    def illustrate_splits(self, path, lemma, pred, scaling_method='tSNE', metric='euclidean'):
        graph.illustrate_splits(path=path, data=self.samples, pred=pred, lemma=lemma,
                                scaling_method=scaling_method, metric='euclidean')

    def illustrate_lemma(self, path, lemmas):
        graph.illustrate_lemma(data=self.samples, path=path, lemma=lemmas)


abs_dir = os.path.dirname(os.getcwd())


@hydra.main(config_path=f"{abs_dir}/src/config", config_name="eval_dwug")
def run_script(cfg: DictConfig):
    print(cfg)
    print(cfg.model_vec)
    model_vectorizer = instantiate(cfg.model_vec)
    evaler = NSD_evaluator(path=cfg.evaluator.input_path, eval_type=cfg.evaluator.eval_type,
                           window=cfg.evaluator.window,
                           dist_func=cfg.evaluator.dist_func, vectorizer=model_vectorizer,
                           threshold=cfg.evaluator.threshold, visualize=cfg.evaluator.visualize,
                           norm=cfg.evaluator.norm, use_decision=cfg.evaluator.use_decision,
                           min_precision=cfg.evaluator.min_precision)
    evaler.evaluate(path=cfg.evaluator.output_path, preload=cfg.evaluator.preload,
                    save_embeds=cfg.evaluator.save_embeds)
    if cfg.evaluator.visualize:
        words = evaler.samples.word.unique()
        for word in words:
            # evaler.illustrate_lemma(path='../../../../results/graphs/' + word, lemmas=word)
            # evaler.illustrate_splits(path='../../../../results/graphs/' + word, lemma=word,
            #                          pred=evaler.dict_for_viz,
            #                          scaling_method='tSNE', metric=cfg.evaluator.dist_func)
            evaler.illustrate_lemma(path=os.getcwd()+'/graphs/'+word, lemmas=word)
            evaler.illustrate_splits(path=os.getcwd()+'/graphs/'+word, lemma=word,
                                     pred=evaler.dict_for_viz,
                                     scaling_method='tSNE', metric=cfg.evaluator.dist_func)


if __name__ == '__main__':
    # sys.path.append('C:/Users/79222/summer-wsi')
    run_script()
    print('Task complete!')
