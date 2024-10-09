#!/usr/bin/env bash
moe_type=$1
size=$2

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
	UPDATE_FREQ=12
elif [ "$size" = "large" ]; then
	MAX_TOKENS=6554
	UPDATE_FREQ=40
else
	echo "Run the code with moe-train.sh {gs/sh} {base/large}"; exit 128;
fi

if [ "$size" = "large" ] && [ "$moe_type" = "gs" ]; then
        GATE_CAPACITY="[2,3]"
else
        GATE_CAPACITY="[4,6]"
fi

TOTAL_UPDATES=20000
PEAK_LR=0.0002
EXPERT_DROPOUT=0.25
SAVE_INTERVAL=100 

SEED=77
SAVE_PATH="models/moece-${moe_type}-${size}"
PRETRAINED_DIR=models/pretrained
ARCH="t5-v1.1-${size}"
PRETRAINED_PATH=$PRETRAINED_DIR/${ARCH}.pt
MOE_ARCH="t5-moe-v1.1-${size}"
mkdir -p $SAVE_PATH

DATA_WORKERS=2

DATA_DIR=data/fairseq-aux-bin

timestamp=`date "+%Y%0m%0d_%T"`
mkdir -p $SAVE_PATH/logs
me=$(basename "$0")
cp $me $SAVE_PATH/logs/

python train.py $DATA_DIR \
    --seed $SEED \
    --num-workers $DATA_WORKERS \
    --moe-freq 1 --moe-type $v_moe_type --aux-weight 0.1 \
    --gate-logits --gate-hidden-dims "[384]" --gate-class-dim 28 \
    --moe-location decoder --expert-dropout $EXPERT_DROPOUT \
    --num-experts 7 --share-gate --gate-top-n $top_k --gate-capacity $GATE_CAPACITY \
    --skip-invalid-size-inputs-valid-test \
    --no-epoch-checkpoints \
    --task aux_translation -s ori -t cor -x edit \
    --criterion label_smoothed_cross_entropy_for_moe --moe-weight 1 \
    --arch $MOE_ARCH \
    --restore-file $PRETRAINED_PATH \
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
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 10 2>&1 | tee $SAVE_PATH/logs/train-${timestamp}.log
