#!/bin/bash

# Define arrays for comparison
MODELS=("FNO")
LOSS_FUNC=("relative_l2")
DATASETS=("hh_step" "hh_poisson")
CONFIGS=("1")

for MODEL in "${MODELS[@]}"; do
    for LOSS in "${LOSS_FUNC[@]}"; do
        for DATA in "${DATASETS[@]}"; do
            for CONFIG in "${CONFIGS[@]}"; do
                echo "Running $MODEL with LOSS=$LOSS, DATASET=$DATA, and CONFIG=$CONFIG"
                python main.py \
                    --model_name "$MODEL" \
                    --dataset_name "$DATA" \
                    --model_config "${MODEL}_config${CONFIG}" \
                    --loss_func_name "$LOSS" \
                    --batch_size 32 \
                    --epochs 500 \
                    --lr 0.001\
                    --resume_path None #"./checkpoints/${MODEL}_config${CONFIG}_${LOSS}_${DATA}_best.pth.tar"
            done
        done
    done
done

