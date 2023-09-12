#!/bin/sh
#SBATCH --gres=gpu:1
nvidia-smi
date

#MODEL_PATH="/home/myurachinskiy/WSD/wsd-multilingual-biencoders-checkpoints/checkpoints_context/XLMR_large_05_unbal.ckpt"
MODEL_PATH="/home/myurachinskiy/WSD/wsd-multilingual-biencoders/checkpoints_05_unbal_large/best_model.ckpt"
#ENCODER_NAME="xlm-roberta-large"
#PY_PATH="GLM/glm_vectorizer.py"

#export JAVA_TOOL_OPTIONS="-Dfile.encoding=UTF8"
export PYTHONPATH="${PYTHONPATH}:/home/myurachinskiy/WSI/summer-wsi"

#python -m pytest -s test_glm.py
#python -m pytest -s test_glm.py --weights-path $MODEL_PATH
python -m pytest -s ../test_glm.py --weights-path $MODEL_PATH --samples-path /home/myurachinskiy/WSI/RNC/rnc_full.json --gold-vectors-path /home/myurachinskiy/WSI/preds/preds_rnc_full.json
