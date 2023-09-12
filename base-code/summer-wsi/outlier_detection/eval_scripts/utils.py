from dataclasses import dataclass
import pandas as pd
from multilang_wsi_evaluation.interfaces import Sample

@dataclass()
class SampleNSD(Sample):
    gold_sense_id: int = None
    grouping: int = None



def helper_subword(begin, end, maxlen=512):
    n = end - begin
    x = begin - min((maxlen-n)//2, begin)
    y = end + (maxlen - n - min((maxlen-n)//2, begin))
    return x, y, min((maxlen-n)//2, begin), min((maxlen-n)//2, begin)+n

def get_samples_from_df(data, no_gs=False, window=None):
    sample_iter = []
    for i, v in data.iterrows():
        if window == 'subword':
            begin, end = v.positions.split('-')
            x, y, z, t = helper_subword(int(begin), int(end))
            begin, end = z, t
            ctx = v.context[x:y]
        else:
            ctx = v.context
            begin, end = v.positions.split('-')
        if no_gs:
            sample_iter.append(SampleNSD(ctx, int(begin), int(end), v.word))
        else:
            sample_iter.append(SampleNSD(ctx, int(begin), int(end), v.word, int(v.gold_sense_id)))
    return sample_iter

def get_df_from_samples(samples, no_gs=False):
    out = pd.DataFrame(columns=['context_id', 'word', 'positions', 'context', 'gold_sense_id', 'pooled_embeds'])
    for i, sample in enumerate(samples):
        out.loc[i] = [i, sample.lemma, str(sample.begin) + '-' + str(sample.end), sample.context, sample.gold_sense_id]
    return out

