# export HF_ENDPOINT=https://hf-mirror.com
# python -m scripts.agilex_inference \
# --use_actions_interpolation \
# --pretrained_model_name_or_path="./weights/rdt-1b" \
# --lang_embeddings_path="./weights/maniskill-model/lang_embeds/text_embed_StackCube-v1.pt" \
# --ctrl_freq=25 \
# # --disable_puppet_arm

# export HF_ENDPOINT=https://hf-mirror.com
# python -m scripts.agilex_inference \
# --use_actions_interpolation \
# --pretrained_model_name_or_path="./checkpoints/rdt-finetune-1b/checkpoint-2000" \
# --lang_embeddings_path="./weights/maniskill-model/lang_embeds/text_embed_StackCube-v1.pt" \
# --ctrl_freq=25 \
# # --disable_puppet_arm

# export HF_ENDPOINT=https://hf-mirror.com
# python -m scripts.agilex_inference \
# --use_actions_interpolation \
# --pretrained_model_name_or_path="./checkpoints/rdt-finetune-1b-stack-cube_v0/checkpoint-5000" \
# --lang_embeddings_path="./weights/maniskill-model/lang_embeds/text_embed_StackCube-v1.pt" \
# --ctrl_freq=10 \
# --publish_rate=30 \
# # --disable_puppet_arm

# export HF_ENDPOINT=https://hf-mirror.com
# python -m scripts.agilex_inference \
# --use_actions_interpolation \
# --pretrained_model_name_or_path="./checkpoints/rdt-finetune-1b-stack-cube_v0/checkpoint-8000" \
# --lang_embeddings_path="./weights/maniskill-model/lang_embeds/text_embed_StackCube-v1.pt" \
# --ctrl_freq=25 \
# --publish_rate=30 \
# # --disable_puppet_arm

export HF_ENDPOINT=https://hf-mirror.com
python -m scripts.agilex_inference \
--use_actions_interpolation \
--pretrained_model_name_or_path="/home/agilex/cobot_magic/RoboticsDiffusionTransformer/checkpoints/rdt-finetune-1b-fold-cloth/checkpoint-12000" \
--lang_embeddings_path="/mnt/fold_cloth.pt" \
--ctrl_freq=25 \
--publish_rate=30 \
# --disable_puppet_arm
