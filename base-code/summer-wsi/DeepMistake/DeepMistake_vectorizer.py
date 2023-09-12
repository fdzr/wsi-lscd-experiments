import fire
import torch
import pandas as pd
from typing import List
from pathlib import Path
import tempfile
from multilang_wsi_evaluation.interfaces import IWSIVectorizer
from multilang_wsi_evaluation.vectorizer_evaluator import vec_eval
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import subprocess
import json
import os
import shutil


def csv_to_json(df, path_to_save):
    word = df.iloc[0].word
    json_file = [
        {'id': f'{word}.{row[0]}', 'lemma': row[1].word, 'pos': "NOUN",
         'sentence1': row[1].sent1, 'sentence2': row[1].sent2,
         'start1': row[1].pos1[0], 'end1': row[1].pos1[1],
         'start2': row[1].pos2[0], 'end2': row[1].pos2[1]}
        for row in df.iterrows()]
    f = open(path_to_save, 'w')
    json.dump(json_file, f, indent=4)
    f.flush()


def avg_scores(i):
    val = float(i['score'][0]) if len(i['score']) < 2 else (float(i['score'][0]) + float(i['score'][1])) / 2
    return val


def json_to_csv(file):
    fr = open(file, 'r')
    data = json.load(fr)
    ids = [i['id'] for i in data]
    #sores = [1 - avg_scores(i) for i in data]
    sores = [avg_scores(i) for i in data]
    df_scores = pd.DataFrame()
    df_scores['id'] = ids
    df_scores['score'] = sores
    return df_scores


def f(len_samples, num):
    ans = 0
    while num != 0:
        ans += len_samples
        len_samples -= 1
        num -= 1
    return ans


def make_matrix(df, samples):
    scores = list(df.score)
    len_samples = len(samples)
    temp_mat = [[0.0] + scores[f(len_samples - 1, num):f(len_samples - 1, num + 1)] for num in range(len_samples)]
    for i in range(len(temp_mat)):
        for j in range(len(temp_mat)):
            if i == j:
                break
            else:
                temp_mat[i].insert(j, temp_mat[j][i])
    return temp_mat

def bin_scores(scores, treshold):
    new_scores = []
    for i in scores:
        if i >= treshold:
            new_scores.append(1)
        else:
            new_scores.append(i)
    return new_scores

def make_matrix_bin(df, len_samples, treshold):
    scores = list(df.score)
    temp_mat = [[0.0] + bin_scores(scores[f(len_samples - 1, num):f(len_samples - 1, num + 1)], treshold) for num in range(len_samples)]
    for i in range(len(temp_mat)):
        for j in range(len(temp_mat)):
            if i == j:
                break
            else:
                temp_mat[i].insert(j, temp_mat[j][i])
    return temp_mat


class DMVectorizer(IWSIVectorizer):
    def __init__(self, path_to_scr='run_model.py', path_model='../mean_dist_l1ndotn_CE', eval_flag='--do_eval',
                 eval_output_dir='predictions/',
                 output_dir='../mean_dist_l1ndotn_CE', loss='crossentropy_loss', pool_type='mean', symmetric='true'):
        self.path_to_scr = path_to_scr
        self.path_model = path_model
        self.eval_flag = eval_flag
        self.eval_output_dir = eval_output_dir
        self.output_dir = output_dir
        self.loss = loss
        self.pool_type = pool_type
        self.symmetric = symmetric

    def fit(self, all_samples):
        self.treshold = 0.5
        pass

    def predict(self, samples):
        t = '/home/daniil/Downloads/l1ndotn_schemas-20211227T071727Z-001/l1ndotn_schemas/ru-ru/'
        if samples[0].lemma not in os.listdir(t):
            return [[0 for _ in range(len(samples))] for _ in range(len(samples))]
        #pairs = [(i, j) for num1, i in enumerate(samples) for num2, j in enumerate(samples) if num1 < num2]
        #df = pd.DataFrame()
        #df['sent1'] = [i.context for i, j in pairs]
        #df['pos1'] = [(int(i.begin), int(i.end)) for i, j in pairs]
        #df['sent2'] = [j.context for i, j in pairs]
        #df['pos2'] = [(int(j.begin), int(j.end)) for i, j in pairs]
        #df['word'] = [i.lemma for i, j in pairs]
        last_output = os.getcwd().split('/')[-1]
        eval_output_dir = 'predictions/'
        Path("cashDeepMistake").mkdir(exist_ok=True)
        Path(f"cashDeepMistake/{samples[0].lemma}").mkdir(parents=True, exist_ok=True)
        Path(f"cashDeepMistake/{eval_output_dir}").mkdir(parents=True, exist_ok=True)
        # Path(f"cashDeepMistake/{samples[0].lemma}/tsv_files").mkdir(parents=True, exist_ok=True)
        eval_input_dir = f"cashDeepMistake/{samples[0].lemma}"
        # df.to_csv(f'{eval_input_dir}/tsv_files/{samples[0].lemma}', sep='\t')
        #csv_to_json(df, f'{eval_input_dir}/{samples[0].lemma}.data')
        # os.chdir("../../DeepMistake/DeepMistake/mcl-wic")
        self.eval_input_dir = f"../../../outputs/{last_output}/{eval_input_dir}"
        output_dir = f"../../../outputs/{last_output}/cashDeepMistake"
        """subprocess.run(["python",
                        self.path_to_scr,
                        self.eval_flag,
                        "--ckpt_path", self.path_model,
                        "--eval_input_dir", self.eval_input_dir,
                        "--eval_output_dir", self.eval_output_dir,
                        "--output_dir", self.output_dir,
                        "--loss", self.loss,
                        "--pool_type", self.pool_type,
                        "--symmetric", self.symmetric])"""
        # path_to_preds = f"{self.output_dir}/{self.eval_output_dir}{samples[0].lemma}.scores"
        df_scores = json_to_csv(f'{t}{samples[0].lemma}/russe_bts-rnc.{samples[0].lemma}.scores')
        # shutil.copyfile(t, f"{output_dir}/{eval_output_dir}/{samples[0].lemma}.scores")
        matrix = make_matrix_bin(df_scores, len(samples), self.treshold)
        # matrix = [[0 for i in range(len(samples))] for j in range(len(samples))]
        #os.chdir(f"../../../outputs/{last_output}")
        return matrix


@hydra.main(config_path='conf', config_name='config')
def run(cfg: DictConfig):
    print(f'Config: {OmegaConf.to_yaml(cfg)}')
    model_vectorizer = instantiate(cfg.model_vec)
    param = dict(cfg.model_vec)
    del param['_target_']
    metric = cfg.return_value['metric']
    dataset = cfg.return_value['dataset']
    cfg_clusterer = cfg.model_clusterer
    val = vec_eval(model_vectorizer,
                   cfg_clusterer=cfg_clusterer,
                   opt_metric=metric,
                   opt_dataset=dataset,
                   dists=['precomputed'])
    return 1


if __name__ == '__main__':
    run()

    '''def predict(self, samples):
        """pairs = [(i, j) for i in samples for j in samples if samples.index(i) < samples.index(j)]
        df = pd.DataFrame()
        df['sent1'] = [i.context for i, j in pairs]
        df['pos1'] = [(int(i.begin), int(i.end)) for i, j in pairs]
        df['sent2'] = [j.context for i, j in pairs]
        df['pos2'] = [(int(j.begin), int(j.end)) for i, j in pairs]
        df['word'] = [i.lemma for i, j in pairs]
        last_output = os.getcwd().split('/')[-1]"""
        #eval_output_dir = 'predictions/'
        #Path("cashDeepMistake").mkdir(exist_ok=True)
        #Path(f"cashDeepMistake/{samples[0].lemma}").mkdir(parents=True, exist_ok=True)
        #Path(f"cashDeepMistake/{eval_output_dir}").mkdir(parents=True, exist_ok=True)
        # Path(f"cashDeepMistake/{samples[0].lemma}/tsv_files").mkdir(parents=True, exist_ok=True)
        #eval_input_dir = f"cashDeepMistake/{samples[0].lemma}"
        # df.to_csv(f'{eval_input_dir}/tsv_files/{samples[0].lemma}', sep='\t')
        #csv_to_json(df, f'{eval_input_dir}/{samples[0].lemma}.data')
        #os.chdir("../../baselines/DeepMistake/mcl-wic")
        #self.eval_input_dir = f"../../../outputs/{last_output}/{eval_input_dir}"
        #output_dir = f"../../../outputs/{last_output}/cashDeepMistake"
        """subprocess.run(["python",
                        self.path_to_scr,
                        self.eval_flag,
                        "--ckpt_path", self.path_model,
                        "--eval_input_dir", self.eval_input_dir,
                        "--eval_output_dir", self.eval_output_dir,
                        "--output_dir", self.output_dir,
                        "--loss", self.loss,
                        "--pool_type", self.pool_type,
                        "--symmetric", self.symmetric])"""
        #path_to_preds = f"{self.output_dir}/{self.eval_output_dir}{samples[0].lemma}.scores"
        path_to_preds = f'/home/daniil/sum_wsi_old/summer-wsi/temp_dir/2021-12-06_15-00-24/cashDeepMistake/predictions/{samples[0].lemma}.scores'
        df_scores = json_to_csv(path_to_preds)
        #shutil.copyfile(path_to_preds, f"{output_dir}/{eval_output_dir}/{samples[0].lemma}.scores")
        matrix = make_matrix(df_scores, len(samples))
        #matrix = make_matrix_bin(df_scores, len(samples), self.treshold)
        #matrix = [[0 for i in range(len(samples))] for j in range(len(samples))]
        #os.chdir(f"../../../outputs/{last_output}")
        return matrix'''
