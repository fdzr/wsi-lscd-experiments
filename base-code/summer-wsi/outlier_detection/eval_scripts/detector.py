import os
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
import re
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pipe as pipe
from utils import get_samples_from_df
import scipy.sparse
import time

class NSD_detector():

    def __init__(self, path1, path2, wordlist, vectorizer=None,
                  dist_func='euclidean', threshold=1.0,  window=None, norm=None):
        self.path1 = path1
        self.path2 = path2
        self.wordlistpath = wordlist
        with open(self.path1, 'r', encoding='utf-8') as f:
            self.corpus1 = f.read().split('\n')
        with open(self.path2, 'r', encoding='utf-8') as f:
            self.corpus2 = f.read().split('\n')
        self.window = window
        self.pipe = pipe.NSD_pipe(vectorizer=vectorizer, dist_func=dist_func, threshold=threshold, norm=norm)
        with open(self.wordlistpath, 'r', encoding='utf-8') as wl:
            self.wordlist = wl.read().split()
        self.samples = pd.DataFrame(columns=['context_id', 'word', 'positions', 'context', 'grouping'])
        self.form_df()
        print('len samples:', len(self.samples))

    def form_df(self):
        count = 0
        self.actual_wordlist = []
        for word in tqdm(self.wordlist):
            found1 = False
            found2 = False
            for sent in self.corpus1:
                m = re.search(rf'\b{word}\b', sent)
                if m:
                    found1 = True
            for sent in self.corpus2:
                m = re.search(rf'\b{word}\b', sent)
                if m:
                    found2 = True
            if found1 and found2:
                self.actual_wordlist.append(word)
                for sent in self.corpus1:
                    m = re.search(rf'\b{word}\b', sent)
                    if m:
                        self.samples.loc[count] = [count, word, str(m.span()[0]) + '-' + str(m.span()[1]), sent, '1']
                        count += 1
                for sent in self.corpus2:
                    m = re.search(rf'\b{word}\b', sent)
                    if m:
                        self.samples.loc[count] = [count, word, str(m.span()[0]) + '-' + str(m.span()[1]), sent, '2']
                        count += 1

    def get_splits(self):
        resulting_splits = []
        for word in self.actual_wordlist:
            data = self.samples[self.samples.word == word]
            temp_ = []
            for i in range(2):
                ix_train = data[data['grouping'] == str(i % 2 + 1)]['context_id'].values
                ix_test = data[data['grouping'] == str((i + 1) % 2 + 1)]['context_id'].values

                train_samples = data.loc[ix_train]
                test_samples = data.loc[ix_test]
                labels = None
                temp_.append({'train': get_samples_from_df(train_samples, no_gs=True, window=self.window),
                              'test': get_samples_from_df(test_samples, no_gs=True, window=self.window), 'labels': labels})
            resulting_splits.append(temp_)
        return resulting_splits

    def detect(self, output_path, preload=False, save_embeds=None):
        start = time.time()
        if preload:
            if preload.endswith('.npz'):
                embeds = scipy.sparse.load_npz(preload)
                embeds = embeds.toarray()
            else:
                embeds = np.load(preload)
            self.pipe.embeds = embeds
            self.pipe.samples = get_samples_from_df(self.samples, no_gs=True, window=self.window)
        else:
            self.pipe.samples = get_samples_from_df(self.samples, no_gs=True, window=self.window)
            print('calculating embeddings..')
            embeds = self.pipe.fit()
        if save_embeds:
            np.save(save_embeds, embeds)
        self.pipe.apply_norm()
        end = time.time() - start
        print('time for embedding passed: ', end)
        labels = []
        splits = self.get_splits()
        for word in splits:
            temp = []
            for split in word:
                temp.append(self.pipe.predict(split['train'], split['test']))
            labels.append(temp)
        detected_outliers1 = []
        detected_outliers2 = []
        temp = 0
        for w in labels:
            for i, corp in enumerate(w):
                for j in corp:
                    if j and i == 0:
                        detected_outliers1.append(self.samples['context'].values[temp])
                    if j and i == 1:
                        detected_outliers2.append(self.samples['context'].values[temp])
                    temp += 1

        with open(os.getcwd()+'/detection_results.txt', 'w', encoding='utf-8') as o:
            o.write('Outliers in corpus 1\n')
            for sent in detected_outliers1:
                o.write(sent+'\n')
            o.write('Outliers in corpus 2\n')
            for sent in detected_outliers2:
                o.write(sent+'\n')
        return detected_outliers1, detected_outliers2

abs_dir = os.path.dirname(os.getcwd())
@hydra.main(config_path=f"{abs_dir}/src/config", config_name="detector_cfg")
def run_script(cfg: DictConfig):
    # # experiments_dict = [{'path1': '../../../../txt_for_detector/post_1000.txt',
    # #                      'path2': '../../../../txt_for_detector/pre_1000.txt',
    # #                      'wordlist': '../../../../txt_for_detector/wordlist.txt',
    # #                      'norm': 'l1', 'dist_func': 'cosine'}]
    # for exp in experiments_dict:
    start = time.time()
    # cfg.detector.path1 = exp['path1']
    # cfg.detector.path2 = exp['path2']
    # cfg.detector.wordlist = exp['wordlist']
    # cfg.detector.norm = exp['norm']
    # cfg.detector.dist_func = exp['dist_func']
    print(cfg)
    model_vectorizer = instantiate(cfg.model_vec)
    print('vectorizer ready!')
    detector = NSD_detector(path1=cfg.detector.path1, path2=cfg.detector.path2, wordlist=cfg.detector.wordlist,
                            dist_func=cfg.detector.dist_func, vectorizer=model_vectorizer,
                            window=cfg.detector.window, threshold=cfg.detector.threshold,
                            norm=cfg.detector.norm)
    detector.detect(output_path=cfg.detector.output_path, preload=cfg.detector.preload, save_embeds=cfg.detector.save_embeds)
    end = time.time()-start
    print('time passed: ', end)

if __name__ == '__main__':
    run_script()
    print('Task complete!')
