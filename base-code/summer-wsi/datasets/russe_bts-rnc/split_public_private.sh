SRC_DIR=../../russe-wsi-kit/data/main/bts-rnc
cp $SRC_DIR/train.csv ru/train.csv
(head -1 $SRC_DIR/test-solution.csv; for w in `cat $SRC_DIR/public.txt`; do  grep $'\t'$w$'\t' $SRC_DIR/test-solution.csv ; done) > ru/test-public.csv
