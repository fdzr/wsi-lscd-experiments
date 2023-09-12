#!/bin/sh
#SBATCH --gres=gpu:1
nvidia-smi
date

MODEL_PATH="/home/myurachinskiy/WSD/wsd-multilingual-biencoders-checkpoints/checkpoints_context/XLMR_large_05_unbal.ckpt"
#MODEL_PATH="/home/myurachinskiy/WSD/wsd-multilingual-biencoders/checkpoints_05_unbal_large/best_model.ckpt"
ENCODER_NAME="xlm-roberta-large"
#PY_PATH="GLM/glm_vectorizer.py"
SAMPLES_PATH="/home/myurachinskiy/WSI/RNC/rnc_full.json"
OUTPUT_PATH="/home/myurachinskiy/WSI/summer-wsi/GLM/tests/data/glm_rnc_full_v2.json"
DEVICE="cuda"

#export JAVA_TOOL_OPTIONS="-Dfile.encoding=UTF8"
export PYTHONPATH="${PYTHONPATH}:/home/myurachinskiy/WSI/summer-wsi"

python ../glm_inference.py --weights-path $MODEL_PATH --samples-path $SAMPLES_PATH --encoder-name $ENCODER_NAME --output-path $OUTPUT_PATH --device $DEVICE 
