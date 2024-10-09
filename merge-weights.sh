#!/usr/bin/env bash
TOTAL_UPDATES=20000
PEAK_LR=0.00005
EXPERT_DROPOUT=0.25
SAVE_INTERVAL=1 

RAW_INPUT=$1
CKPT_PATH=$(echo "$RAW_INPUT" | sed 's:/*$::')
FOLDER_NAME=$(basename $(dirname $CKPT_PATH))
COMPS=(${FOLDER_NAME//-/ })
moe_type=${COMPS[1]}
size=${COMPS[2]}
size=${COMPS[2]}

if [ "$moe_type" = "gs" ]; then
	v_moe_type="aux_gshard"
	top_k=2
elif [ "$moe_type" = "st" ]; then
	v_moe_type="aux_switch2"
	top_k=1
else
	echo "Run the code with moe-train.sh {gs/sh} {base/large}"; exit 128;
fi

if [ "$size" = "base" ]; then
	MAX_TOKENS=21845
	UPDATE_FREQ=96
elif [ "$size" = "large" ]; then
	MAX_TOKENS=8192
	UPDATE_FREQ=256
else
	echo "Run the code with moe-train.sh {gs/sh} {base/large}"; exit 128;
fi

SEED=77
SAVE_PATH="models/merged-${moe_type}-${size}"
mkdir -p $SAVE_PATH
cp $CKPT_PATH $SAVE_PATH/checkpoint_last.pt

ARCH="t5-v1.1-${size}"
MOE_ARCH="t5-moe-v1.1-${size}"
mkdir -p $SAVE_PATH

DATA_WORKERS=0
DATA_DIR=data/fairseq-aux-bin

timestamp=`date "+%Y%0m%0d_%T"`
mkdir -p $SAVE_PATH/logs
me=$(basename "$0")
cp $me $SAVE_PATH/logs/

python train.py $DATA_DIR \
    --seed $SEED \
    --num-workers $DATA_WORKERS \
    --max-epoch 3 --freeze-non-MoE --share-expert-gelu --merge-backbone \
    --moe-freq 1 --moe-type $v_moe_type --aux-weight 0.1 \
    --gate-logits --gate-hidden-dims "[384]" --gate-class-dim 28 \
    --moe-location decoder --expert-dropout $EXPERT_DROPOUT \
    --num-experts 7 --share-gate --gate-top-n $top_k --gate-capacity "[4,6]" \
    --skip-invalid-size-inputs-valid-test \
    --task aux_translation -s ori -t cor -x edit \
    --criterion label_smoothed_cross_entropy_for_moe --moe-weight 1 \
    --arch $MOE_ARCH \
    --reset-optimizer --reset-lr-scheduler --reset-dataloader --reset-meters \
    --max-source-positions 128 \
    --max-target-positions 128 \
    --optimizer adafactor \
    --lr $PEAK_LR \
    --update-freq $UPDATE_FREQ \
    --max-tokens $MAX_TOKENS \
    --save-dir $SAVE_PATH \
    --disable-validation \
    --max-tokens-valid $MAX_TOKENS \
    --save-interval-updates $SAVE_INTERVAL \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 2>&1 | tee $SAVE_PATH/logs/train-${timestamp}.log
