export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=1
python -m eval_sim.eval_rdt_maniskill \
    --pretrained_path ./weights/maniskill-model/rdt/mp_rank_00_model_states.pt \
    -e PickCube-v1 \
    -n 5 \
    --shader default