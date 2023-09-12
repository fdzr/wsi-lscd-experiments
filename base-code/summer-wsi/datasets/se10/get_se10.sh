#!/bin/bash
git clone https://github.com/Samsung/LexSubGen
cd LexSubGen
conda create -n lexsubgen python=3.7.4
conda activate lexsubgen
pip install -r requirements.txt
./init.sh
python setup.py install
pip install word_forms
cd ..
mv se10_to_bts-rnc.py LexSubGen/se10_to_bts-rnc.py
cd LexSubGen
python se10_to_bts-rnc.py
