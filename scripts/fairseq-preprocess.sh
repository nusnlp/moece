#!/bin/sh

tp=$1
trainpref=$(realpath $tp)
dd=$2
destdir=$(realpath $dd)
cd data

fairseq-preprocess --source-lang ori --target-lang cor --validpref tokenized/bea-dev_tokenized --srcdict tf_vocab.txt --tgtdict tf_vocab.txt --workers 20 --trainpref $trainpref --destdir $destdir --no-specials | tee tmp.txt
mv tmp.txt $destdir/preprocess.log
