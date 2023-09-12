import os
import math
import random
from typing import List

import torch
import hydra
import numpy as np
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from nltk.tokenize import word_tokenize
from omegaconf import DictConfig

from multilang_wsi_evaluation.interfaces import Sample, IWSIVectorizer
from multilang_wsi_evaluation import evaluate_vectorizer

# Todo: кастомная нормализация как ввести?

# TODO: Or should we just use Blevins repo as a package?
class ContextEncoder(torch.nn.Module):
    def __init__(self, encoder_name, device, encoder_weights_path=None):
        super(ContextEncoder, self).__init__()

        # load pretrained model as base for context encoder and gloss encoder
        self.context_encoder = XLMRobertaModel.from_pretrained(encoder_name)
        if encoder_weights_path is not None:
            self.context_encoder.load_state_dict(torch.load(encoder_weights_path, map_location=device))

    @staticmethod
    def process_encoder_outputs(output, mask, as_tensor=False):
        combined_outputs = []
        position = -1
        avg_arr = []
        for idx, rep in zip(mask, torch.split(output, 1, dim=0)):
            # ignore unlabeled words
            if idx == -1:
                continue
            # average representations for units in same example
            elif position < idx:
                position = idx
                if len(avg_arr) > 0:
                    combined_outputs.append(torch.mean(torch.stack(avg_arr, dim=-1), dim=-1))
                avg_arr = [rep]
            else:
                assert position == idx
                avg_arr.append(rep)
        # get last example from avg_arr
        if len(avg_arr) > 0:
            combined_outputs.append(torch.mean(torch.stack(avg_arr, dim=-1), dim=-1))
        if as_tensor:
            if len(combined_outputs) <= 0:
                raise RuntimeError(f'Empty vectors list for sentence. mask={mask.detach().cpu().tolist()}')
            return torch.cat(combined_outputs, dim=0)
        else:
            return combined_outputs

    def forward(self, input_ids, attn_mask, output_mask):
        # encode context
        context_output = self.context_encoder(input_ids, attention_mask=attn_mask)[0]

        # average representations over target word(s)
        example_arr = []
        for i in range(context_output.size(0)):
            example_arr.append(self.process_encoder_outputs(context_output[i], output_mask[i], as_tensor=True))
        context_output = torch.cat(example_arr, dim=0)

        return context_output


class GLMVectorizer(IWSIVectorizer):
    def __init__(self, encoder_name='xlm-roberta-large', encoder_weights_path=None,
                 device='cpu', max_context_words_count=None, target_window_size=None, random_seed=42, debug=False):
        self.set_random_seeds(random_seed)
        self.device = torch.device(device)
        print('Initializing GLMVectorizer: ', encoder_name, encoder_weights_path, self.device)
        self.encoder = ContextEncoder(encoder_name, self.device, encoder_weights_path)
        self.tokenizer = self.load_tokenizer(encoder_name)
        self.encoder = self.encoder.to(self.device)
        self.max_context_words_count = max_context_words_count
        self.target_window_size = target_window_size
        self.debug = debug

    @staticmethod
    def set_random_seeds(rand_seed):
        torch.manual_seed(rand_seed)
        os.environ['PYTHONHASHSEED'] = str(rand_seed)
        torch.cuda.manual_seed(rand_seed)
        torch.cuda.manual_seed_all(rand_seed)
        np.random.seed(rand_seed)
        random.seed(rand_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def load_tokenizer(model_name):
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

        tokenizer.real_encode = tokenizer.encode
        tokenizer.encode = lambda text, **args: (
            tokenizer.real_encode(text, **args)
            if 'add_special_tokens' in args
            else tokenizer.real_encode(text, add_special_tokens=False, **args)
        )

        return tokenizer

    @staticmethod
    def normalize_length(ids, attn_mask, o_mask, max_len, pad_id):
        if max_len == -1:
            return ids, attn_mask, o_mask
        else:
            if len(ids) < max_len:
                while len(ids) < max_len:
                    ids.append(torch.tensor([[pad_id]]))
                    attn_mask.append(0)
                    o_mask.append(-1)
            else:
                ids = ids[:max_len - 1] + [ids[-1]]
                attn_mask = attn_mask[:max_len]
                o_mask = o_mask[:max_len]

            assert len(ids) == max_len
            assert len(attn_mask) == max_len
            assert len(o_mask) == max_len

            return ids, attn_mask, o_mask

    @staticmethod
    def get_window_around_target(o_masks, window_width):
        first_target_ind = None
        for i, mask in enumerate(o_masks):
            if mask != -1:
                first_target_ind = i
                break
        assert first_target_ind is not None

        end_target_ind = first_target_ind + 1
        while end_target_ind < len(o_masks) and o_masks[end_target_ind] != -1:
            end_target_ind += 1

        left_width = window_width - (end_target_ind - first_target_ind)
        assert left_width >= 0

        shift = left_width / 2
        left_shift = math.ceil(shift)
        right_shift = math.floor(shift)
        left_context_len = first_target_ind
        right_context_len = len(o_masks) - end_target_ind

        left_remainder = max(0, left_shift - left_context_len)
        right_remainder = max(0, right_shift - right_context_len)
        left_shift += right_remainder
        right_shift += left_remainder

        start_index = max(0, first_target_ind - left_shift)
        end_index = min(len(o_masks), end_target_ind + right_shift)

        return start_index, end_index

    @staticmethod
    def preprocess_context(tokenizer, text_data, bsz=1, max_len=-1, target_window_size=None):
        if max_len == -1:
            assert bsz == 1  # otherwise need max_length for padding

        context_ids = []
        context_attn_masks = []

        example_keys = []

        context_output_masks = []
        instances = []
        labels = []

        # tensorize data
        for sent in text_data:
            c_ids = [torch.tensor(
                [tokenizer.encode(tokenizer.cls_token)])]  # cls token aka sos token, returns a list with index
            o_masks = [-1]
            sent_insts = []
            sent_keys = []
            sent_labels = []

            # For each word in sentence...
            for idx, (word, lemma, pos, inst, label) in enumerate(sent):
                # tensorize word for context ids
                word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word.lower())]
                c_ids.extend(word_ids)

                # if word is labeled with WSD sense...
                if inst != -1:
                    # add word to bert output mask to be labeled
                    o_masks.extend([idx] * len(word_ids))
                    # track example instance id
                    sent_insts.append(inst)
                    # track example instance keys to get glosses
                    sent_keys.append(None)
                    sent_labels.append(label)
                else:
                    # mask out output of context encoder for WSD task (not labeled)
                    o_masks.extend([-1] * len(word_ids))

                # break if we reach max len
                if max_len != -1 and len(c_ids) >= (max_len - 1):
                    break

            if target_window_size is not None:
                c_ids_first, c_ids_rest = c_ids[0], c_ids[1:]
                o_masks_first, o_masks_rest = o_masks[0], o_masks[1:]
                left_target_size = target_window_size - 2  # CLS and SEP tokens
                truncated_start, truncate_end = GLMVectorizer.get_window_around_target(o_masks_rest, left_target_size)
                c_ids_rest = c_ids_rest[truncated_start:truncate_end]
                o_masks_rest = o_masks_rest[truncated_start:truncate_end]
                c_ids = [c_ids_first] + c_ids_rest
                o_masks = [o_masks_first] + o_masks_rest
                assert len(c_ids) == len(o_masks)

            c_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token)]))  # aka eos token
            c_attn_mask = [1] * len(c_ids)
            o_masks.append(-1)
            c_ids, c_attn_masks, o_masks = GLMVectorizer.normalize_length(
                c_ids, c_attn_mask, o_masks, max_len,
                pad_id=tokenizer.encode(tokenizer.pad_token)[0]
            )

            # y = torch.tensor([1] * len(sent_insts), dtype=torch.float)
            # not including examples sentences with no annotated sense data
            if len(sent_insts) > 0:
                context_ids.append(torch.cat(c_ids, dim=-1))
                context_attn_masks.append(torch.tensor(c_attn_masks).unsqueeze(dim=0))
                context_output_masks.append(torch.tensor(o_masks).unsqueeze(dim=0))
                example_keys.append(sent_keys)
                instances.append(sent_insts)
                labels.append(sent_labels)

        # package data
        data = list(zip(context_ids, context_attn_masks, context_output_masks, example_keys, instances, labels))

        # batch data if bsz > 1
        if bsz > 1:
            print('Batching data with bsz={}...'.format(bsz))
            batched_data = []
            for idx in range(0, len(data), bsz):
                if idx + bsz <= len(data):
                    b = data[idx:idx + bsz]
                else:
                    b = data[idx:]
                context_ids = torch.cat([x for x, _, _, _, _, _ in b], dim=0)
                context_attn_mask = torch.cat([x for _, x, _, _, _, _ in b], dim=0)
                context_output_mask = torch.cat([x for _, _, x, _, _, _ in b], dim=0)
                example_keys = []
                for _, _, _, x, _, _ in b:
                    example_keys.extend(x)
                instances = []
                for _, _, _, _, x, _ in b:
                    instances.extend(x)
                labels = []
                for _, _, _, _, _, x in b:
                    labels.extend(x)
                batched_data.append(
                    (context_ids, context_attn_mask, context_output_mask, example_keys, instances, labels))
            return batched_data
        else:
            return data

    def construct_vectors(self, eval_data):
        self.encoder.eval()
        id_to_vector = {}

        for context_ids, context_attn_mask, context_output_mask, example_keys, insts, _ in eval_data:
            with torch.no_grad():
                context_ids = context_ids.to(self.device)
                context_attn_mask = context_attn_mask.to(self.device)
                context_output = self.encoder.forward(context_ids, context_attn_mask, context_output_mask)

                for output, key, inst in zip(context_output.split(1, dim=0), example_keys, insts):
                    current_context_output = output.squeeze().detach().cpu().tolist()
                    id_to_vector[inst] = current_context_output

        return id_to_vector

    def construct_sample(self, left_context, word, right_context, sample_id, target_lemma, pos):
        left_tokens = word_tokenize(left_context)
        right_tokens = word_tokenize(right_context)

        def get_context_token_info(token):
            return token, '', -1, -1, False

        left_tokens = [get_context_token_info(token) for token in left_tokens]
        right_tokens = [get_context_token_info(token) for token in right_tokens]
        target_token = (word, target_lemma, pos, sample_id, True)

        if self.max_context_words_count is None or len(left_tokens) + len(right_tokens) <= self.max_context_words_count:
            return left_tokens + [target_token] + right_tokens

        shift = self.max_context_words_count // 2
        left_remainder = max(0, shift - len(left_tokens))
        right_remainder = max(0, shift - len(right_context))

        return left_tokens[-(shift + right_remainder):] + [target_token] + right_tokens[:shift + left_remainder]

    def transform_samples(self, samples: List[Sample]):
        wsd_data = []

        for i, sample in enumerate(samples):
            start = int(sample.begin)
            end = int(sample.end)
            transformed_sample = self.construct_sample(
                left_context=sample.context[:start],
                word=sample.context[start:end],
                right_context=sample.context[end:],
                sample_id=i,
                target_lemma=sample.lemma, pos=None
            )
            wsd_data.append(transformed_sample)

        return wsd_data

    def fit(self, samples):
        pass

    def predict(self, samples: List[Sample]):
        if self.debug: print('GLMVectorizer: predict() for ', len(samples), 'samples')
        if self.debug: print('\n'.join([f'{s.context[s.begin:s.end]} | {s.context}' for s in samples[:15]]))
        text_data = self.transform_samples(samples)
        processed_data = self.preprocess_context(
            self.tokenizer, text_data, bsz=1, max_len=-1, target_window_size=self.target_window_size
        )
        id_to_vector = self.construct_vectors(processed_data)

        vectors = [[] for _ in range(len(id_to_vector))]
        for vector_id, vector in id_to_vector.items():
            vectors[vector_id] = vector

        return vectors


def norm_dist(vec1, vec2, d):
    vec1 = vec1 / np.linalg.norm(vec1, ord=d)
    vec2 = vec2 / np.linalg.norm(vec2, ord=d)

    return np.linalg.norm(vec1 - vec2, ord=d)


def norm_l1(vec1, vec2):
    return norm_dist(vec1, vec2, d=1)


def norm_l2(vec1, vec2):
    return norm_dist(vec1, vec2, d=2)


@hydra.main(config_path='conf', config_name='config')
def run(cfg: DictConfig):
    return evaluate_vectorizer.run(cfg)


if __name__ == '__main__':
    run()
