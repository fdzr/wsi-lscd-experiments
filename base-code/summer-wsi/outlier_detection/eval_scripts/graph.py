from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, cosine_similarity
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn import manifold
from sklearn.metrics import DistanceMetric
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
import scipy as sp

def get_forms(data, word, metric='euclidian'):
    if metric == 'euclidian':
        metr = euclidean_distances
    elif metric == 'cosine':
        metr = cosine_distances
    df = data[data['word']==word] 
    same_form = []
    diff_form = []
    for i, v in df.iterrows():
        p1, p2 = v.positions.split('-')
        p1, p2 = int(p1), int(p2)
        word = v.context[p1:p2]
        for j, v1 in df.iterrows():
            p3, p4 = v1.positions.split('-')
            p3, p4 = int(p3), int(p4)
            word1 = v1.context[p3:p4]
            if i!=j and word.lower() == word1.lower():
                same_form.append(euclidean_distances([v.pooled_embeds], [v1.pooled_embeds])[0][0])
            if i!=j and word.lower() != word1.lower():
                diff_form.append(euclidean_distances([v.pooled_embeds], [v1.pooled_embeds])[0][0])
    return same_form, diff_form

def plot_pairwise_dist(data, lemma, path,  metric='euclidian'):
    def get_pairwise_dist(data, word, metric='euclidian'):
        if metric == 'euclidian':
            metr = euclidean_distances
        elif metric == 'cosine':
            metr = cosine_distances
        df = data[data['word']==word]
        same_sense = []
        diff_sense = []
        for i in df['gold_sense_id'].unique():
            X = np.stack(df['pooled_embeds'][df['gold_sense_id']==i])
            #get_inverse_for_maha() ???
            dists = metr(X,X)
            d = np.triu(dists, k=1).reshape(-1)
            d = np.delete(d, np.where(d == 0))
            same_sense.extend(list(d))
            for j in df['gold_sense_id'].unique():
                if j!=i and j>i:
                    Y = np.stack(df['pooled_embeds'][df['gold_sense_id']==j])
                    dists = metr(X,Y)
                    d = dists.reshape(-1)
                    d = np.delete(d, np.where(d == 0))
                    diff_sense.extend(list(d))  
        same_form, diff_form = get_forms(data, word, metric)
        return same_sense, diff_sense, same_form, diff_form
    if lemma not in data['word'].unique():
        print(f'no data for word {lemma}!')
        return
    sss, dss, sfs, dfs = [],[],[],[]
    ss, ds, sf, df = get_pairwise_dist(data, lemma, metric)
    sss.append(ss)
    dss.append(ds)
    sfs.append(sf)
    dfs.append(df)
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(18.5, 5)
    ax[0].set(xlabel=str(metric)+' distance', ylabel='count')
    # sns.histplot(sss[0], ax=ax[0], color='red', element="poly", legend=True)
    # sns.histplot(dss[0], ax=ax[0],  element="poly")
    # sns.histplot(sfs[0], ax=ax[1], color='green', element="poly", legend=True)
    # sns.histplot(dfs[0], ax=ax[1], color='brown',  element="poly")
    sns.histplot(sss[0], ax=ax[0], color='red', legend=True, kde=True)
    sns.histplot(dss[0], ax=ax[0], kde=True)
    sns.histplot(sfs[0], ax=ax[1], color='green', legend=True, kde=True)
    sns.histplot(dfs[0], ax=ax[1], color='brown',  kde=True)

    ax[0].legend(labels=['same sense', 'different senses'])
    ax[1].legend(labels=['same form', 'different form']) 
    ax[0].set_title(lemma)
    ax[1].set_title(lemma)
    plt.suptitle('Embedding distances histogram for form/sense combinations', fontsize=20)
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(path+'/'+f'pairwise_distances_{lemma}.png')
    plt.close('all')

def get_embeds_2d_PCA(df):
    X = np.stack(df['pooled_embeds'])
    pca = PCA(n_components=2)
    pos = pca.fit_transform(X)
    return pos

def get_embeds_2d_MDS(df): 
    X = np.stack(df['pooled_embeds'])
    similarities = euclidean_distances(X)
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=42,
                   dissimilarity="precomputed")
    pos = mds.fit(similarities).embedding_
    return pos
    

def get_embeds_2d_tSNE(df, perplexity=5):
    X = np.stack(df['pooled_embeds'])
    pca = manifold.TSNE(n_components=2, perplexity=perplexity, n_iter=10000, init='pca', random_state=17)
    pos = pca.fit_transform(X)
    return pos

def graph_quality(train_X, test_X, preds_, dist_func='euclidean', threshold=1):
    tree = BallTree(train_X, metric=dist_func, leaf_size=40)
    preds = []
    for x in test_X:
        res = get_predict(train_X, x, tree, data=None, actual_label=None, threshold=threshold, verbose=False)
        preds.append(int(res))
   # print(f'Graph Q: accuracy score for tetta={threshold}: ', accuracy_score(preds_, preds))
   # print('Graph Q: confusion matrix:\n', confusion_matrix(preds_, preds))
    return preds

def get_predict(train, x, tree, data, actual_label, threshold=1, verbose=False):
    """ get label wether given x embedding is unknown (1) or known (0) for given train embedding set """
    x = x.reshape(1,-1)
    d1, ix = tree.query(x, k=1) # obtain distance d[xt]: x to NN in train
    d2, ix2 = tree.query([train[ix.item()]], k=2) # obtain distance d[tt']: distance between t and its NN in train, k=2 as there is also distance to itself
    assert d2.tolist()[0][0]==0, "Vector's distance to itself not zero"
    ratio = (d1/d2.tolist()[0][1])
    res = ratio[0][0] > threshold # get ratio to compare to threshold tetta. 1 if outlier, 0 otherwise.
    if verbose:
        s1, s2, s3, se1, se2, se3 = obtain_sentence_pair(x[0], train[ix][0][0], train[ix2][0][1], data)
        print(f'''Test sentence: {s1,se1}\nSentence from train set: {s2,se2}\nSentence 2 from train set {s3,se3}\nratio {ratio[0]} \nVerdict: {res} \nActual label: {bool(actual_label)}\n''')
    
    return res   
    
def plot_embeds_2d_pltly(data, word, path, scaling_method=get_embeds_2d_MDS, preds=None, special_ix=None, perplexities=None, split_no=0):
    if scaling_method=='MDS':
        scaling_method=get_embeds_2d_MDS
    elif  scaling_method=='tSNE':
        scaling_method=get_embeds_2d_tSNE
    elif  scaling_method=='PCA':
        scaling_method=get_embeds_2d_PCA

    preds_, df_train, df_test = preds[word][split_no]
    
    pos_train = scaling_method(df_train)
    pos_test = scaling_method(df_test)
    twod_preds = graph_quality(pos_train, pos_test, preds_)

    df_train['part'] = ['train' for _ in range(len(df_train))]
    df_train['2dpreds'] = ['NA' for _ in range(len(df_train))]
    df_test['2dpreds'] = twod_preds
    df_test['part'] = ['test outlier' if t else 'test seen' for t in preds_]
    df_ = pd.concat([df_train, df_test])
    x_ = np.append(pos_train[:, 0], pos_test[:, 0])
    y_ = np.append(pos_train[:, 1], pos_test[:, 1])
    fig = px.scatter(df_, x=x_ , y=y_, color=df_['gold_sense_id'].apply(str), symbol='part', hover_data=['context', '2dpreds'])
    fig.update_layout(autosize=False, width=1400, height=800, hoverlabel_font_size=12, legend_xanchor='left',
                      title_text=f'{word}')
    fig.update_traces(marker={'size': 15})
    if not os.path.exists(path):
        os.makedirs(path)
    fig.write_html(path+'/'+f'plotly_2d_skatterplot_{word}_{split_no}.html')

def get_4(data, word, metric='euclidean'):
    if metric == 'euclidian':
        metr = euclidean_distances
    elif metric == 'cosine':
        metr = cosine_distances
    df = data[data['word']==word]
    same_form_diff_sense  = []
    diff_form_same_sense = []
    diff_form_diff_sense = []
    same_form_same_sense = []
    for i, v in df.iterrows():
        p1, p2 = v.positions.split('-')
        p1, p2 = int(p1), int(p2)
        word = v.context[p1:p2]
        for j, v1 in df.iterrows():
            p3, p4 = v1.positions.split('-')
            p3, p4 = int(p3), int(p4)
            word1 = v1.context[p3:p4]
            if i!=j and word.lower() == word1.lower() and v.gold_sense_id == v1.gold_sense_id:
                same_form_same_sense.append(euclidean_distances([v.pooled_embeds], [v1.pooled_embeds])[0][0])
            if i!=j and word.lower() == word1.lower() and v.gold_sense_id != v1.gold_sense_id:
                same_form_diff_sense.append(euclidean_distances([v.pooled_embeds], [v1.pooled_embeds])[0][0])
            if i!=j and word.lower() != word1.lower() and v.gold_sense_id == v1.gold_sense_id:
                diff_form_same_sense.append(euclidean_distances([v.pooled_embeds], [v1.pooled_embeds])[0][0])
            if i!=j and word.lower() != word1.lower() and v.gold_sense_id != v1.gold_sense_id:
                diff_form_diff_sense.append(euclidean_distances([v.pooled_embeds], [v1.pooled_embeds])[0][0])
    return same_form_diff_sense, same_form_same_sense, diff_form_same_sense, diff_form_diff_sense

def plot_four_dist(data, lemma, path, metric='euclidian'):
    sd, ss, ds, dd = get_4(data, lemma, metric)
    fig, ax = plt.subplots(1, 1)
    x, y = 1, 1
    fig.set_size_inches(18.5, 5*x)
    fig.tight_layout(pad=3.0)
    ax.set(xlabel='distance', ylabel='count')
    sns.histplot(ss, ax=ax, color='red', kde=True, legend=True)
    sns.histplot(sd, ax=ax, kde=True)
    sns.histplot(ds, ax=ax, color='green', kde=True, legend=True)
    sns.histplot(dd, ax=ax, color='yellow', kde=True,)
    ax.legend(labels=['same form same sense', 'same form diff senses', 'diff form same sense', 'diff form diff sense'], loc='upper right')
    ax.set_title(lemma)
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(path + '/' + f'pairwise_distances_four_distributions_{lemma}.png')
    plt.close('all')

def get_inverse_for_maha(matix, epsilon=1e-15):
    cov = np.cov(matix.T)
    cov[np.isnan(cov)] = 0
    N = matix.shape[1]
    eps_mtrx = np.zeros((N, N), float)
    np.fill_diagonal(eps_mtrx, epsilon)
    inv_cov = sp.linalg.inv(cov + eps_mtrx)
    return inv_cov
    
def inner_split_hist(data, path, word, preds, metric='euclidean', split_no=0):
    labels, train, test = preds[word][split_no]
    if metric == 'maha':
        metric = DistanceMetric.get_metric('mahalanobis', V=get_inverse_for_maha(np.array(train['pooled_embeds'].tolist())))
    tree = BallTree(np.array(train['pooled_embeds'].tolist()), metric=metric, leaf_size=40) # init BallTree
    inner_train = []
    test_seen = []
    test_outlier = []
    labels = [bool(x) for x in labels]
    # print('labels: ', labels)
    for i, el in train.iterrows():
        x = el.pooled_embeds
        size_v = len(x)
        x = x.reshape(-1, size_v)
        d1, ix = tree.query(x, k=2)
        inner_train.append(d1[0][1])
    for i, el in test[labels].iterrows():
        x = el.pooled_embeds
        size_v = len(x)
        x = x.reshape(-1, size_v)
        d1, ix = tree.query(x, k=1)
        test_outlier.append(d1[0][0])
    labels = [not x for x in labels]
    for i, el in test[labels].iterrows():
        x = el.pooled_embeds
        size_v = len(x)
        x = x.reshape(-1, size_v)
        d1, ix = tree.query(x, k=1)
        test_seen.append(d1[0][0])
    fig = plt.figure(figsize=(10,6))
    sns.histplot(test_seen,  color='blue', legend=True, alpha=.7, kde=True, label='test seen').set_title("Distances between embedding vectors from train, test set (outliers and "
                                                                                                         "non-outliers) and its NN from train set")
    sns.histplot(inner_train,  color='red', legend=True, alpha=.1, kde=True, label='train')
    sns.histplot(test_outlier,  color='green', legend=True, alpha=.5, kde=True, label='test outlier')
    plt.legend()
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(path+'/'+f'distances_to_NN_in_trainset_{word}_{split_no}.png')
    plt.close('all')


def distances_to_NN_test_set(path, word, preds, metric='euclidean', split_no=0):
    labels, train, test = preds[word][split_no]
    if metric == 'maha':
        metric = DistanceMetric.get_metric('mahalanobis', V=get_inverse_for_maha(np.array(test['pooled_embeds'].tolist())))
    tree = BallTree(np.array(test['pooled_embeds'].tolist()), metric=metric, leaf_size=40)  # init BallTree
    test_seen = []
    test_outlier = []
    labels = [bool(x) for x in labels]
    for i, el in test[labels].iterrows():
        x = el.pooled_embeds
        size_v = len(x)
        x = x.reshape(-1, size_v)
        d1, ix = tree.query(x, k=2)
        test_outlier.append(d1[0][1])
    labels = [not x for x in labels]
    for i, el in test[labels].iterrows():
        x = el.pooled_embeds
        size_v = len(x)
        x = x.reshape(-1, size_v)
        d1, ix = tree.query(x, k=2)
        test_seen.append(d1[0][1])
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(test_outlier, color='red', kde=True, legend=True, alpha=.1, label='test seen').set_title(
        "Distance to Nearest Neighbour among test set embeddings")
    sns.histplot(test_seen, color='green', kde=True, legend=True, alpha=.5, label='test outlier')
    plt.legend()
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(path+'/'+f'distance_to_NN_among_testset_{word}_{split_no}.png')
    plt.close('all')

def distances_to_NN_train_set(path, word, preds, metric='euclidean', split_no=0):
    labels, train, test = preds[word][split_no]
    if metric == 'maha':
        metric = DistanceMetric.get_metric('mahalanobis', V=get_inverse_for_maha(np.array(train['pooled_embeds'].tolist())))
    tree = BallTree(np.array(train['pooled_embeds'].tolist()), metric=metric, leaf_size=40)  # init BallTree
    train_ = []
    for i, el in train.iterrows():
        x = el.pooled_embeds
        size_v = len(x)
        x = x.reshape(-1, size_v)
        d1, ix = tree.query(x, k=2)
        train_.append(d1[0][1])

    fig = plt.figure(figsize=(10, 6))
    sns.histplot(train_, color='red', kde=True, legend=True, alpha=.1, label='train set').set_title(
        "Distance to Nearest Neighbour among train set embeddings")
    plt.legend()
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(path+'/'+f'distance_to_NN_among_trainset_{word}_{split_no}.png')
    plt.close('all')

def ratio_distance(path, word, preds, metric='euclidean', split_no=0):
    labels, train, test = preds[word][split_no]
    if metric == 'maha':
        metric1 = DistanceMetric.get_metric('mahalanobis', V=get_inverse_for_maha(np.array(train['pooled_embeds'].tolist())))
        metric2 = DistanceMetric.get_metric('mahalanobis', V=get_inverse_for_maha(np.array(test['pooled_embeds'].tolist())))
        tree1 = BallTree(np.array(train['pooled_embeds'].tolist()), metric=metric1, leaf_size=40) # init BallTree
        tree2 = BallTree(np.array(test['pooled_embeds'].tolist()), metric=metric2, leaf_size=40) # init BallTree
    else:
        tree1 = BallTree(np.array(train['pooled_embeds'].tolist()), metric=metric, leaf_size=40)  # init BallTree
        tree2 = BallTree(np.array(test['pooled_embeds'].tolist()), metric=metric, leaf_size=40)  # init BallTree
    inner_train = []
    test_seen = []
    test_outlier = []
    labels = [bool(x) for x in labels]
    for i, el in test[labels].iterrows():
        x = el.pooled_embeds
        size_v = len(x)
        x = x.reshape(-1, size_v)
        d1, ix = tree1.query(x, k=1)
        d_train = d1[0][0]
        d2, ix = tree2.query(x, k=2)
        d_test = d2[0][1]
        if d_train/d_test < np.inf:
            test_outlier.append(d_train/d_test)
    labels = [not x for x in labels]
    for i, el in test[labels].iterrows():
        x = el.pooled_embeds
        size_v = len(x)
        x = x.reshape(-1, size_v)
        d1, ix = tree1.query(x, k=1)
        d_train = d1[0][0]
        d2, ix = tree2.query(x, k=2)
        d_test = d2[0][1]
        if d_train / d_test < np.inf:
            test_seen.append(d_train/d_test)
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(test_outlier,  color='red', kde=True, legend=True, alpha=.1, label='test seen').set_title("Ratio of Distance to NN in train set to Distance to NN in test set")
    sns.histplot(test_seen,  color='green', kde=True, legend=True, alpha=.5, label='test outlier')
    plt.legend()
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(path + '/' + f'Erk_ratio_{word}_{split_no}.png')
    plt.close('all')


def illustrate_lemma(data, path, lemma):
    plot_pairwise_dist(data, lemma=lemma, path=path)
    plot_four_dist(data, lemma=lemma, path=path)

def illustrate_splits(path, lemma, data, pred, scaling_method, metric='euclidean'):
   # metric = 'maha'
    splits = list(range(len(pred[lemma])))
    for split in splits:
        distances_to_NN_test_set(word=lemma, split_no=split, preds=pred, path=path, metric=metric)
        ratio_distance(word=lemma, split_no=split, preds=pred, path=path, metric=metric)
        distances_to_NN_train_set(word=lemma, split_no=split, preds=pred, path=path, metric=metric)
        plot_embeds_2d_pltly(data, word=lemma, split_no=split, preds=pred, path=path, scaling_method=scaling_method)
        inner_split_hist(data, preds=pred, path=path, word=lemma, split_no=split, metric=metric)