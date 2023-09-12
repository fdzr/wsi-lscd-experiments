This folder contains two runnable scripts: detector.py and evaluator.py.   
Scripts support hydra lib for configs. Configs stored at ./config.  

To run evaluator call: `python evaluator.py`
User can specify path_to_config_folder/config_filename with following arguments:  
`python evaluator.py --config-path=CONFIG_PATH --config-name=CONFIG_FILE_NAME`  
There are three vectorizers available at the moment: my_vec(default), bow(for bag of word features) and glm.

If `use_decions` flag is set, output decision values would be saved to`eval_scripts/outputs/` and 
particular hydra run folde name in format `decsions_NORM_DISTANCE.npy`.  
To generate plots set flag `vizualize` to True. Results would be saved in`/graphs` folder of particular hydra run
(`eval_scripts/outputs`) where for each lemma a separate folder would be created.

**All scripts below run from outlier_detection/eval_scripts directory  
    You might want to append project opath to sys!**

# Reproduction commands
  1) For given dataset generate decision values for each vectorizer  
  2) Run jupyter notebook (explorer.ipynb) to generate visualizations and obtain statistics  on obtained .npy  
files with decision values and all_labels.npy with gold labels for experiment.
  Results present in thesis Figs 4, 8, 12, 17; Tables 1, 2, 3.
## 1. FrameNet
  |         Vectorizer           | best average precision | 
|:----------------------------:|:----------------------:|
|            XLM-R             |         0.9041         |
|             GLM              |         0.9315         |
|             BOW              |         0.8623         | 
  Fig. 4 in thesis.  
For GLM: `python evaluator.py --config-name=eval_framenet.yaml model_vec=glm model_vec.encoder_weights_path=/home/auditor/summer-wsi/outlier_detection/XLMR_large_05_unbal.ckpt evaluator.norm=l1 evaluator.dist_func=mahalanobis`  
For XLM-R: `python evaluator.py --config-name=eval_framenet.yaml model_vec=my_vec evaluator.norm=l1 evaluator.dist_func=mahalanobis`  
For BOW: `python evaluator.py --config-name=eval_framenet.yaml model_vec=bow evaluator.norm=l2 evaluator.dist_func=mahalanobis`  

## 2. Dwug de 
**Daichronic evaluation** 

|         Vectorizer           | best average precision | 
|:----------------------------:|:----------------------:|
|            XLM-R             |         0.3958         |
|             GLM              |         0.7428         |
|             BOW              |         0.3821         | 
 Fig. 8 in thesis.   
For GLM: `python evaluator.py --config-name=eval_dwug.yaml model_vec=glm model_vec.encoder_weights_path=/home/auditor/summer-wsi/outlier_detection/XLMR_large_05_unbal.ckpt evaluator.norm=l1 evaluator.dist_func=cosine evaluator.eval_type=diachronic`  
For XLM-R: `python evaluator.py --config-name=eval_dwug.yaml model_vec=my_vec evaluator.norm=l1 evaluator.dist_func=mahalanobis evaluator.eval_type=diachronic`   
For BOW: `python evaluator.py --config-name=eval_dwug.yaml model_vec=bow evaluator.norm=l1 evaluator.dist_func=euclidean evaluator.eval_type=diachronic`  

**Synchronic evaluation, new corpus**  

|         Vectorizer         | best average precision | 
|:--------------------------:|:----------------------:|
|           XLM-R            |         0.8474         |
|   GLM without loaded wights   |         0.8405         |
| GLM with loaded weights |         0.9435         |
|            BOW             |         0.8629         | 
Fig. 17 in thesis.  
For GLM with loaded weigths: `python evaluator.py --config-name=eval_dwug.yaml model_vec=glm model_vec.encoder_weights_path=/home/auditor/summer-wsi/outlier_detection/XLMR_large_05_unbal.ckpt evaluator.input_path=../../../../../datasets/se20lscd/de/sense_grouping_1.csv evaluator.norm=l1 evaluator.dist_func=mahalanobis evaluator.eval_type=synchronic`  
For GLM without loaded weights: `python evaluator.py --config-name=eval_dwug.yaml model_vec=glm evaluator.input_path=../../../../../datasets/se20lscd/de/sense_grouping_1.csv evaluator.norm=l1 evaluator.dist_func=mahalanobis evaluator.eval_type=synchronic`  
For XLM-R: `python evaluator.py --config-name=eval_dwug.yaml model_vec=my_vec evaluator.input_path=../../../../../datasets/se20lscd/de/sense_grouping_1.csv evaluator.norm=l1 evaluator.dist_func=mahalanobis evaluator.eval_type=synchronic`  
For BOW: `python evaluator.py --config-name=eval_dwug.yaml model_vec=bow evaluator.input_path=../../../../../datasets/se20lscd/de/sense_grouping_1.csv evaluator.norm=l2 evaluator.dist_func=mahalanobis evaluator.eval_type=synchronic`  

**Synchronic evaluation, old corpus**  

|         Vectorizer         | best average precision | 
|:--------------------------:|:----------------------:|
|           XLM-R            |         0.7754         |
|   GLM with loaded wights   |         0.837          |
| GLM without loaded weights |         0.7754         |
|            BOW             |         0.7949         | 
Fig. 12 in thesis  
For GLM with loaded weights: `python evaluator.py --config-name=eval_dwug.yaml model_vec=glm model_vec.encoder_weights_path=/home/auditor/summer-wsi/outlier_detection/XLMR_large_05_unbal.ckpt evaluator.input_path=../../../../../datasets/se20lscd/de/sense_grouping_2.csv evaluator.norm=l1 evaluator.dist_func=mahalanobis evaluator.eval_type=synchronic`  
For GLM without loaded weights: `python evaluator.py --config-name=eval_dwug.yaml model_vec=glm evaluator.input_path=../../../../../datasets/se20lscd/de/sense_grouping_2.csv evaluator.norm=l1 evaluator.dist_func=mahalanobis evaluator.eval_type=synchronic`  
For XLM-R: `python evaluator.py --config-name=eval_dwug.yaml model_vec=my_vec evaluator.input_path=../../../../../datasets/se20lscd/de/sense_grouping_2.csv evaluator.norm=l1 evaluator.dist_func=mahalanobis evaluator.eval_type=synchronic`  
For BOW: `python evaluator.py --config-name=eval_dwug.yaml model_vec=bow evaluator.input_path=../../../../../datasets/se20lscd/de/sense_grouping_2.csv evaluator.norm=l2 evaluator.dist_func=mahalanobis evaluator.eval_type=synchronic`  

## 3. BTS-RNC
|        Vectorizer         | best average precision | 
|:-------------------------:|:----------------------:|
|           XLM-R           |         0.9152         |
|  GLM with loaded wights   |         0.9201         |
| GLM without loaded wights |         0.9165         | 
Tables 1,2,3 in thesis.  
For GLM with loaded weights: `python evaluator.py --config-name=eval_bts.yaml model_vec=glm model_vec.encoder_weights_path=/home/auditor/summer-wsi/outlier_detection/XLMR_large_05_unbal.ckpt evaluator.norm=l1 evaluator.dist_func=mahalanobis evaluator.eval_type=synchronic`  
For GLM without loaded weights: `python evaluator.py --config-name=eval_bts.yaml model_vec=glm evaluator.norm=l1 evaluator.dist_func=mahalanobis evaluator.eval_type=synchronic`  
For XLM-R: `python evaluator.py --config-name=eval_bts.yaml model_vec=my_vec evaluator.norm=l1 evaluator.dist_func=mahalanobis evaluator.eval_type=synchronic`  
For BOW: `python evaluator.py --config-name=eval_bts.yaml model_vec=bow evaluator.norm=l2 evaluator.dist_func=mahalanobis evaluator.eval_type=synchronic`  

## 4.Fixed testset experiments on Dwug de
For GLM with loaded weights:`python evaluator.py --config-name=eval_dwug.yaml model_vec=glm model_vec.encoder_weights_path=/home/auditor/summer-wsi/outlier_detection/XLMR_large_05_unbal.ckpt evaluator.norm=l1 evaluator.dist_func=manhattan evaluator.eval_type=equal_test`  
For GLM without loaded weights: `python evaluator.py --config-name=eval_dwug.yaml model_vec=glm evaluator.norm=l1 evaluator.dist_func=mahalanobis evaluator.eval_type=equal_test`  
For XLM-R: `python evaluator.py --config-name=eval_dwug.yaml model_vec=my_vec evaluator.norm=l1 evaluator.dist_func=mahalanobis evaluator.eval_type=equal_test`  
For BOW: `python evaluator.py --config-name=eval_dwug.yaml model_vec=my_vec evaluator.norm=l2 evaluator.dist_func=mahalanobis evaluator.eval_type=equal_test`

## 5. Detection speed-tests on RNC-data
Size 10, GLM, mahalanobis:
`python detector.py --config-name=detector_cfg.yaml model_vec=glm model_vec.encoder_weights_path=/home/auditor/summer-wsi/outlier_detection/XLMR_large_05_unbal.ckpt detector.path1=../../../../txt_for_detector/pre_10.txt detector.path2=../../../../txt_for_detector/post_10.txt
detector.wordlist=../../../../txt_for_detector/wordlist.txt detector.norm=l1 detector.dist_func=mahalanobis`  

Size 10, GLM, Euclidean:
`python detector.py --config-name=detector_cfg.yaml model_vec=glm model_vec.encoder_weights_path=/home/auditor/summer-wsi/outlier_detection/XLMR_large_05_unbal.ckpt detector.path1=../../../../txt_for_detector/pre_10.txt detector.path2=../../../../txt_for_detector/post_10.txt
detector.wordlist=../../../../txt_for_detector/wordlist.txt detector.norm=l1 detector.dist_func=euclidean`  

Size 10, GLM, Cosine:
`python detector.py --config-name=detector_cfg.yaml model_vec=glm model_vec.encoder_weights_path=/home/auditor/summer-wsi/outlier_detection/XLMR_large_05_unbal.ckpt detector.path1=../../../../txt_for_detector/pre_10.txt detector.path2=../../../../txt_for_detector/post_10.txt
detector.wordlist=../../../../txt_for_detector/wordlist.txt detector.norm=l1 detector.dist_func=cosine`  
To test on other sizes, replace 10 with 30, 100, 300, 1000.


# Anachronical evaluation quick start  
  to run with XLM-R vectorizer: `python evaluator.py --config-name=eval_framenet.yaml`  
  to run with GLM vectorizer: `python evaluator.py --config-name=eval_framenet.yaml model_vec=glm`  
  to run with GLM vectorizer and pretrained weights:  
`python evaluator.py --config-name=eval_framenet.yaml model_vec=glm model_vec.encoder_weights_path=PATH_TO_WEIGHTS`  
  Output would be decisions.npy and all_labels.npy with decisions value and actual label for each sample.
  
# Diachronical evaluation quick start  
  to run with XLM-R vectorizer: `python evaluator.py --config-name=eval_dwug.yaml`  
  to run with GLM vectorizer: `python evaluator.py --config-name=eval_dwug.yaml model_vec=glm`  
  to run with GLM vectorizer and pretrained weights:  
`python evaluator.py --config-name=eval_bts.yaml model_vec=glm eval_dwug.encoder_weights_path=PATH_TO_WEIGHTS`  
  Output would be decisions.npy and all_labels.npy with decisions value and actual label for each sample.
  
# Detection mode quick start  
  1) place two corpora .txt files (each sample from new line) and .txt file with words of interests (each word separated by space)
  in some directory folder  
  2) Provide path to both corpora and wordlist files in detecor_cfg.yaml config file  
  3) run python detector.py --config-name=detecor_cfg.yaml  
  

## Currently available configs:
  eval_framenet.yaml - evaluation on framenet data  
  eval_bts.yaml - evaluation on bts-rnc test data  
  detector_cfg.yaml - empty generic config for detection task  

# List of hyperparameters for my_vec vectorizer  
* model_str:  model string from hugginface to generate embeddings  
* pretrained_path: relative possible path to load pretrained model from  
* pooling: available: mean/first - two available pooling types   
* layer: layer from which to extract embeddings, 0-12 for XLM-R 

# List of hyperparameters for bow vectorizer  
* radius: number of words close to target to be taken in bag of words. If empty all the sentence is taken.

# List of hyperparameters for evaluator:  
* input_path: relative path to input data in .csv format with columns: context_id,word,gold_sense_id,positions,context  
* output_path: relative path to folder where to store evaluation results  
* preload: relative path to already existing embeddings file in .npy format  
* save_embeds: if preload is not set, relative path to file in .npy format, where calculated embeddings should be saved  
* eval_type: evaluation scenario: diachronical or anachronical  
* vectorizer: leave empty to use default vectorizer (otherwise available option: glm) # will change later to use separate config for vectorizers   
* dist_func: distance function to use in NN-based outlier detector  
* threshold: threshold tetta to use for Erk's method  
* window: available: subword.  Window method to use in case sentences longer, then model can process  
* visualize: True/False. Wether to draw graphs for each split  
  
# List of hyperparameters for detector  
* path1: relative path to first corpus in .txt format, each new sentence in a new line  
* path2: relative path to second corpus in .txt format, each new sentence in a new line  
* wordlist: relative path to .txt file with list of words of interest, each word is separated by space  
* output_path: relative path to folder where to store evaluation results  
* preload: relative path to already existing embeddings file in .npy format  
* save_embeds: if preload is not set, relative path to file in .npy format, where calculated embeddings should be saved  
* vectorizer:leave empty to use default vectorizer (otherwise available option: glm) # will change later to use separate config for vectorizers  
* eval_type: evaluation scenario: diachronical or anachronical  
* dist_func: distance function to use in NN-based outlier detector  
* threshold: threshold tetta to use for Erk's method  

