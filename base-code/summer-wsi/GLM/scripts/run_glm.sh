#!/bin/sh
#SBATCH --gres=gpu:1
nvidia-smi
date

PY_PATH="GLM/glm_vectorizer.py"

export JAVA_TOOL_OPTIONS="-Dfile.encoding=UTF8"
export PYTHONPATH="${PYTHONPATH}:/home/myurachinskiy/WSI/summer-wsi"

cd ../..
python -u $PY_PATH
