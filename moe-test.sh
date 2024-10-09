#!/usr/bin/env bash

MAX_TOKENS=8000
BUFFER_SIZE=200

input_file=$1
ckpt=$2
output_dir=$3

mkdir -p $output_dir

filename="$(basename -- $input_file)"
extension="${filename##*.}"
filename="${filename%.*}"

DATA_WORKERS=12
DATA_DIR=data/fairseq-aux-bin

fairseq-interactive $DATA_DIR --no-specials \
    --num-workers $DATA_WORKERS \
    --remove-bpe=sentencepiece \
    --task translation -s ori -t cor --criterion label_smoothed_cross_entropy \
    --max-tokens $MAX_TOKENS --buffer-size $BUFFER_SIZE \
    --path $ckpt \
    --input $input_file > $output_dir/${filename}.tmp
grep ^H $output_dir/${filename}.tmp | cut -f3 > $output_dir/${filename}.txt
