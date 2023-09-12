from tqdm import tqdm
import nltk
from nltk.corpus import framenet as fn
import pandas as pd
from collections import defaultdict
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument(dest="output_path",  help='path to save parsed data (in .csv format)')
parser.add_argument('--small', action='store_true')
args = parser.parse_args()
OUTPUT_PATH = args.output_path
SMALL = args.small


def parse_framenet(path='datasets/framenet/en/framenet_v17_small.csv'):
    nltk.download('framenet_v17')
    # first select all LexicalUnits belonging to more then 1 frame (has multiple meanings) and has examplars
    res = defaultdict(list)
    for el in tqdm(fn.lus()):
        if len(el.exemplars) > 0:
            res[el.name].append(el.frame)
    polysom = {k: v for k, v in res.items() if len(v) > 1}
    # now create a mapping lemma: examplars, taking only lemmas with 5 and more examplars for each meaning
    lemma_sent = defaultdict(list)
    for k, v in tqdm(polysom.items()):
        temp = []
        more_then_5_each = True
        for frame in v:
            if len(frame.lexUnit[k].exemplars) < 5:
                more_then_5_each = False
            temp.append(frame.lexUnit[k].exemplars)
        if more_then_5_each:
            lemma_sent[k] = temp
    # create dataframe for further processing
    # embeddings_ix column contains token index of lemma we intrested in in whole sentence tokenization
    context_id = []
    word = []
    sense = []
    position1 = []
    position2 = []
    text = []
    j = 0
    if SMALL:
        subsample = random.sample(lemma_sent.keys(), int(len(lemma_sent)*0.1))
        dict_ = {}
        for k in subsample:
            dict_[k] = lemma_sent[k]
        print(int(len(lemma_sent)*0.1))
    else:
        dict_= lemma_sent
    for k, v in dict_.items():
        for sense_, frame in enumerate(v):
            for sent in frame:
                position1.append(sent.Target[0][0])
                position2.append(sent.Target[0][1])
                context_id.append(j)
                j += 1
                word.append(k)
                sense.append(sense_)
                text.append(sent.text)
    to_save = pd.DataFrame()
    to_save['context_id'] = context_id
    to_save['word'] = word
    to_save['gold_sense_id'] = sense
    to_save['positions'] = [str(position1[i])+'-'+str(position2[i]) for i in range(len(position1))]
    to_save['context'] = text
    to_save.to_csv(path,  sep='\t', index=False)



if __name__ == '__main__':
    parse_framenet(path=OUTPUT_PATH)