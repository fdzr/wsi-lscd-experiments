#!/bin/bash

git clone https://github.com/SapienzaNLP/xl-wsd-code.git
cd xl-wsd-code || exit
pip install -r requirements.txt
cd ..

pip install mosestokenizer==1.1.0
pip install gdown==3.3.1

OUTPUT_DIR="WSI_XL-WSD_datasets"
python xl_wsd_to_bts_rnc.py --xl-wsd-code xl-wsd-code --xl-wsd-data xl-wsd-data --out-wsi-dir $OUTPUT_DIR
