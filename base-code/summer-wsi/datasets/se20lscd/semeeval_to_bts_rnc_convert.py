import os
import argparse
import pandas as pd
import csv
from collections import Counter
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(dest="dir_name")
parser.add_argument(dest="clusters")
parser.add_argument(dest="output_path")
parser.add_argument('--grouping', action='store_true')
parser.add_argument('--split_groupings', action='store_true')
args = parser.parse_args()
DIR_NAME = args.dir_name
CLUSTERS_NAME = args.clusters
OUTPUT_PATH = args.output_path
GROUPING = args.grouping
SPLIT = args.split_groupings

if SPLIT and not GROUPING:
    print('Could not split into groupings without --grouping flags set!')
    exit()


data = pd.DataFrame()
if CLUSTERS_NAME != "sense":
    words = [filename[:-4] for filename in os.listdir("{}/clusters/{}".format(DIR_NAME, CLUSTERS_NAME))]
else:
    words = [filename for filename in os.listdir("{}/data".format(DIR_NAME)) if os.path.exists("{}/data/{}/judgments_senses.csv".format(DIR_NAME, filename))]
for word in sorted(words):
    try:
        sentenses = pd.read_csv("{}/data/{}/uses.csv".format(DIR_NAME, word), delimiter="\t", quoting=csv.QUOTE_NONE)
    except Exception as ex:
        print("can't download data from {}      {}".format("{}/data/{}/uses.csv".format(DIR_NAME, word), ex))
        continue
    if CLUSTERS_NAME != "sense":
        try:
            clusters = pd.read_csv("{}/clusters/{}/{}.csv".format(DIR_NAME, CLUSTERS_NAME, word).format(word), delimiter="\t")
        except Exception as ex:
            print("can't download data from {}      {}".format("{}/clusters/{}/{}.csv".format(DIR_NAME, CLUSTERS_NAME, word), ex))
            continue
        chunk_of_data = pd.merge(sentenses, clusters, left_on='identifier', right_on='identifier')
    else:
        try:
            threshold = 3 # threshold for majority labels
            clusters = pd.read_csv("{}/data/{}/judgments_senses.csv".format(DIR_NAME, word), delimiter="\t")
            senses = pd.read_csv("{}/data/{}/senses.csv".format(DIR_NAME, word), delimiter="\t")
            clusters = pd.merge(clusters, senses, left_on='identifier_sense', right_on="identifier_sense")
            clusters = clusters[~(clusters['description_sense'] == 'andere')] # remove andereinstances
            lemmas = clusters.groupby('identifier').agg({'lemma':lambda x: list(x)[0]})
            judgments = clusters.groupby('identifier')['identifier_sense'].apply(list).reset_index(name='judgments')
            clusters = pd.merge(lemmas, judgments, left_on='identifier', right_on="identifier")
            
            # Extract majority labels
            def extract_majority_label(judgments, threshold):
                judgments = list(judgments)
                label2count = Counter(judgments)
                majority_labels = [l for l, c in label2count.items() if c >= threshold]
                if len(majority_labels) > 0:
                    label = np.random.choice(majority_labels)
                else:
                    label = np.NaN  
                return label
            
            #clusters = clusters[clusters['judgments'].apply(lambda x: len(list(x))>threshold)] # remove instances with less than threshold remaining judgments, not needed for now
            #clusters = clusters[~clusters['judgments'].apply(lambda x: extract_majority_label(list(x), threshold)).isnull()] # remove instances which do not reach threshold for majority labeling, not needed for now
            clusters['identifier_sense'] = clusters['judgments'].apply(lambda x: extract_majority_label(list(x), threshold)) # add majority label column
            clusters = clusters[~clusters['identifier_sense'].isnull()] # remove instances which do not reach threshold for majority labeling
            #print(clusters)
        except Exception as ex:
            print("can't download data from {}      {}".format("{}/data/{}/judgments_senses.csv".format(DIR_NAME, word), ex))
            continue
        chunk_of_data = pd.merge(sentenses, clusters[["identifier", "identifier_sense"]], left_on='identifier', right_on='identifier')
        identifier_sense_to_id_mapping = {ident: idx for idx, ident in enumerate(pd.unique(chunk_of_data['identifier_sense']))}
        chunk_of_data["cluster"] = chunk_of_data['identifier_sense'].apply(lambda x: identifier_sense_to_id_mapping[x])
    data = pd.concat([data, chunk_of_data], ignore_index=True)

data["indexes_target_token"] = data["indexes_target_token"].str.replace(":", "-")
data = data[data["indexes_target_token"].str.len() > 2]
if GROUPING:
    bts_rnc_like_data = pd.DataFrame(
        dict(
            context_id=range(1, len(data['lemma']) + 1),
            word=data['lemma'],
            gold_sense_id=data['cluster'],
            positions=data["indexes_target_token"],
            context=data['context'],
            grouping=data['grouping']))
    if SPLIT:
        grpgs = bts_rnc_like_data['grouping'].unique()
        base_path = OUTPUT_PATH[:-4]
        extension = OUTPUT_PATH[-4:]
        if len(grpgs) > 1:
            for grp in grpgs:
                to_save = bts_rnc_like_data[bts_rnc_like_data['grouping']==grp]
                to_save.to_csv(base_path+'_'+str(grp)+extension, sep='\t', index=False, quoting=csv.QUOTE_MINIMAL, quotechar='"', doublequote=True)
else:
    bts_rnc_like_data = pd.DataFrame(
        dict(
            context_id=range(1, len(data['lemma']) + 1),
            word=data['lemma'],
            gold_sense_id=data['cluster'],
            positions=data["indexes_target_token"],
            context=data['context']))
bts_rnc_like_data.to_csv(OUTPUT_PATH, sep='\t', index=False, quoting=csv.QUOTE_MINIMAL, quotechar='"', doublequote=True)
