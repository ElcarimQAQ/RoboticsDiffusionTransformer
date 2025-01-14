#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=1

task=("PegInsertionSide-v1" "PickCube-v1" "StackCube-v1" "PlugCharger-v1" "PushCube-v1")
shader=("default" "rt-fast" "rt")

for t in "${task[@]}"; do
    for s in "${shader[@]}"; do
        if [[ -d videos/"$t"_shader="$s" ]]; then
            echo "Already exists, skip task: "$t"_shader="$s""
            continue
        fi
        python -m eval_sim.eval_rdt_maniskill \
            --pretrained_path ./weights/maniskill-model/rdt/mp_rank_00_model_states.pt \
            -e "$t" \
            -n 10 \
            --shader "$s"
    done
done
