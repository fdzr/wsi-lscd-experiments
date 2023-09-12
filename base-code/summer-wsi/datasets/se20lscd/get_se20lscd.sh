#!/bin/bash
wget https://zenodo.org/record/5544444/files/dwug_en.zip
wget https://zenodo.org/record/5544198/files/dwug_de.zip
wget https://zenodo.org/record/5090648/files/dwug_sv.zip
wget https://zenodo.org/record/5255228/files/dwug_la.zip
echo Unzipping...
for x in *zip; do unzip -q $x; done

for p in opt semeval sense; do 
  for l in en de sv la; do
    if [ "$p" = sense  ] && [ "$l" != de ]
    then
        continue
    fi
 
    mkdir -p $l
    echo Converting $l $p
    python semeeval_to_bts_rnc_convert.py  dwug_${l} ${p} ${l}/${p}.tsv
  done
done
