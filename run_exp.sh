#!/bin/bash

# Define arrays for comparison
MODELS=("FNO")
LOSS_FUNC=("relative_l4" "weighted_l2")

for MODEL in "${MODELS[@]}"; do
    for LOSS in "${LOSS_FUNC[@]}"; do
        echo "Running $MODEL with LOSS=$LOSS"
        python main.py \
            --model_name "$MODEL" \
            --dataset_name "multi_hh" \
            --model_config "${MODEL}_config1" \
            --loss_func_name "$LOSS" \
            --batch_size 32 \
            --epochs 1000 \
            --lr 0.001
    done
done