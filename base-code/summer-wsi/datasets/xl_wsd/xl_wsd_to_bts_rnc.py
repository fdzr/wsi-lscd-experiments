import os
import sys
import argparse
import glob
import yaml
import torch
import csv
import gdown
import zipfile
import shutil
import pandas as pd

from tqdm import tqdm
from collections import defaultdict
from mosestokenizer import MosesDetokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--xl-wsd-code', type=os.path.abspath, required=True)
parser.add_argument('--xl-wsd-data', type=os.path.abspath, required=True)
parser.add_argument('--out-wsi-dir', type=os.path.abspath, required=True)
args = parser.parse_args()


def download_xl_wsd_data():
	url = 'https://drive.google.com/uc?id=19YTL-Uq95hjiFZfgwEpXRgcYGCR_PQY0'
	zip_file = 'xl-wsd-data.zip'
	gdown.download(url, zip_file, quiet=False)

	with zipfile.ZipFile(zip_file, 'r') as zip_ref:
		zip_ref.extractall()

	if os.path.exists(args.xl_wsd_data):
		shutil.rmtree(args.xl_wsd_data)
	os.rename('xl-wsd', args.xl_wsd_data)

	os.remove(zip_file)


download_xl_wsd_data()
os.chdir(args.xl_wsd_code)
sys.path.append('.')
sys.path.append('./src')

try:  # for MacOS Pytorch bug: https://github.com/pytorch/pytorch/issues/23466
	from src.datasets.dataset_utils import get_data, get_allen_datasets, get_dataset
except OSError as e:
	os.system(f'pip install torch=={torch.__version__}')
	from src.datasets.dataset_utils import get_data, get_allen_datasets, get_dataset


def convert_instance_to_sample(instance, detokenizer):
	tokens = [str(token) for token in instance['tokens']]
	labels = instance['labels']
	assert len(tokens) == len(labels)

	labeled_lemmapos = instance['labeled_lemmapos']
	assert all([lemmapos.count('#') == 1 for lemmapos in labeled_lemmapos])
	labeled_lemmas = [lemmapos.split('#')[0] for lemmapos in labeled_lemmapos]
	assert len([sense_labels for sense_labels in labels if len(sense_labels) > 0]) == len(labeled_lemmas)
	labeled_lemmas = iter(labeled_lemmas)

	sentence_samples = []
	processed_words = []
	current_lemma = None

	for i, (word, sense_labels) in enumerate(zip(tokens, labels)):
		if len(sense_labels) > 0:
			current_lemma = next(labeled_lemmas)
		if len(sense_labels) == 1:
			assert current_lemma is not None
			# By now it is O(n^2) but could be optimized with one detokenization
			current_right_index = len(detokenizer(processed_words + [word]))
			sentence_samples.append({
				'word': current_lemma, 'gold_sense_id': sense_labels[0], 'predict_sense_id': None,
				'positions': f'{current_right_index - len(word)}-{current_right_index}'
			})

		processed_words.append(word)

	assert not any(True for _ in labeled_lemmas), 'Not all labels were processed'
	sentence_str = detokenizer(processed_words)
	for sample in sentence_samples:
		sample['context'] = sentence_str

	return sentence_samples


def construct_lang_samples(lang_dataset, detokenizer):
	lang_samples = []

	for instance in lang_dataset.instances:
		instance_samples = convert_instance_to_sample(instance, detokenizer)
		lang_samples.extend(instance_samples)

	lang_samples = pd.DataFrame.from_records(lang_samples)
	lang_samples.sort_values(by=['word', 'gold_sense_id'], inplace=True)

	lang_samples.index = range(len(lang_samples))
	lang_samples.index.rename('context_id', inplace=True)
	lang_samples.reset_index(inplace=True)

	return lang_samples


def wsd_to_wsi(lang_to_path, lemma_to_synsets, label_vocab, source_name, output_dir):
	encoder_name = 'bert-base-multilingual-cased'

	lang_to_datasets = {
		lang: get_dataset(
			encoder_name, {lang: [path]}, lemma_to_synsets,
			label_mapper=None, label_vocab=label_vocab, pos=None
		)
		for lang, path in lang_to_path.items()
	}

	for lang, dataset in tqdm(lang_to_datasets.items()):
		with MosesDetokenizer(lang=lang) as detokenizer:
			lang_samples = construct_lang_samples(dataset, detokenizer)

		output_lang_dir = os.path.join(output_dir, lang)
		os.makedirs(output_lang_dir, exist_ok=True)
		output_path = os.path.join(output_lang_dir, f'{source_name}.tsv')

		lang_samples.to_csv(
			output_path, sep='\t', index=False, quoting=csv.QUOTE_MINIMAL,
			quotechar='"', doublequote=True
		)


def filter_names(lang_to_names, predicate):
	lang_to_name = {}

	for lang, names in lang_to_names.items():
		for name in names:
			if predicate(lang, name):
				assert lang not in lang_to_name
				lang_to_name[lang] = name

	return lang_to_name


def get_paths(lang_to_names, data_dir, name_predicate):
	lang_to_name = filter_names(lang_to_names, name_predicate)

	lang_to_path = {
		lang: os.path.join(data_dir, name, name + '.data.xml')
		for lang, name in lang_to_name.items()
	}

	return lang_to_path


def load_config(config_path):
	with open(config_path) as reader:
		config = yaml.load(reader, Loader=yaml.FullLoader)

	data_config = config['data']

	result_config = {
		'langs': data_config['langs'],
		'mfs_file': data_config.get('mfs_file', None),
		'inventory_dir': os.path.join(args.xl_wsd_data, 'inventories'),
		'lang_to_names': data_config['test_names']
	}
	return result_config


def get_datasets(datasets_dir):
	datasets_paths = glob.glob(f'{datasets_dir}/*')
	datasets_paths = [path for path in datasets_paths if len(glob.glob(f'{path}/*')) > 0]

	return datasets_paths


def get_dataset_name(dataset_path):
	_, filename = os.path.split(dataset_path)

	return filename


def get_lang_to_paths(datasets_paths):
	lang_to_paths = defaultdict(list)

	for path in datasets_paths:
		lang = path.split('_')[-1]
		lang_to_paths[lang].append(path)

	return lang_to_paths


if __name__ == '__main__':
	# Loading info from configs and data dirs
	test_config_path = './config/config_en_semcor_wngt.test.yaml'
	train_config_path = './config/config_en_semcor_wngt.train.yaml'

	test_data_path = os.path.join(args.xl_wsd_data, 'evaluation_datasets')
	train_data_path = os.path.join(args.xl_wsd_data, 'training_datasets')

	test_config = load_config(test_config_path)
	train_config = load_config(test_config_path)  # or create normal train config

	train_datasets_paths = get_datasets(train_data_path)
	train_lang_to_paths = get_lang_to_paths(train_datasets_paths)
	train_config['langs'] = train_lang_to_paths.keys()
	train_config['lang_to_names'] = {lang: [get_dataset_name(path) for path in paths]
	                                 for lang, paths in train_lang_to_paths.items()}

	test_lemma_to_synsets, _, test_label_vocab = get_data(
		test_config['langs'], test_config['mfs_file'], inventory_dir=test_config['inventory_dir']
	)
	train_lemma_to_synsets, _, train_label_vocab = get_data(
		train_config['langs'], train_config['mfs_file'], inventory_dir=train_config['inventory_dir']
	)

	# Converting training XL-WSD datasets
	if os.path.exists(args.out_wsi_dir):
		shutil.rmtree(args.out_wsi_dir)
	output_dir_format = os.path.join(args.out_wsi_dir, 'WSI_XL-WSD-{0}')

	semcor_lang_to_path = get_paths(train_config['lang_to_names'], train_data_path,
	                                lambda lang, name: name == f'semcor_{lang}')
	wsd_to_wsi(semcor_lang_to_path, train_lemma_to_synsets, train_label_vocab, 'train_semcor',
	           output_dir_format.format('train-semcor'))

	wngt_examples_lang_to_path = get_paths(train_config['lang_to_names'], train_data_path,
	                                       lambda lang, name: name == f'wngt_examples_{lang}')
	wsd_to_wsi(wngt_examples_lang_to_path, train_lemma_to_synsets, train_label_vocab, 'train_wngt_examples',
	           output_dir_format.format('train-wngt-examples'))

	wngt_glosses_lang_to_path = get_paths(train_config['lang_to_names'], train_data_path,
	                                      lambda lang, name: name == f'wngt_glosses_{lang}')
	wsd_to_wsi(wngt_glosses_lang_to_path, train_lemma_to_synsets, train_label_vocab, 'train_wngt_glosses',
	           output_dir_format.format('train-wngt-glosses'))

	# Converting dev and test XL-WSD datasets
	dev_test_output_dir = output_dir_format.format('dev-test')

	dev_lang_to_path = get_paths(test_config['lang_to_names'], test_data_path, lambda lang, name: name == f'dev-{lang}')
	wsd_to_wsi(dev_lang_to_path, test_lemma_to_synsets, test_label_vocab, 'dev', dev_test_output_dir)

	test_lang_to_path = get_paths(test_config['lang_to_names'], test_data_path,
	                              lambda lang, name: name == f'test-{lang}')
	wsd_to_wsi(test_lang_to_path, test_lemma_to_synsets, test_label_vocab, 'test', dev_test_output_dir)

	coarse_lang_to_path = get_paths(test_config['lang_to_names'], test_data_path,
	                                lambda lang, name: name == f'test-{lang}-coarse')
	wsd_to_wsi(coarse_lang_to_path, test_lemma_to_synsets, test_label_vocab, 'test-coarse', dev_test_output_dir)
