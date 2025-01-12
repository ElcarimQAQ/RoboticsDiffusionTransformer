# export CUDA_VISIBLE_DEVICES=1
# python -m eval_sim.eval_rdt_maniskill \
# --pretrained_path ./weights/maniskill-model/rdt/mp_rank_00_model_states.pt

export CUDA_VISIBLE_DEVICES=1
python test.py \
--pretrained_model_name_or_path ./weights/rdt-1b \
--lang_embeddings_path ./weights/maniskill-model/lang_embeds/text_embed_StackCube-v1.pt