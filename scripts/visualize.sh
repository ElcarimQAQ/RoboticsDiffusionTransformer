# python visualize_episodes.py \
# --dataset_dir /home/agilex/cobot_magic/data \
# --task_name aloha_mobile_dummy \
# --episode_idx 0


# python visualize_episodes.py \
# --dataset_dir /mnt/datasets/rdt-ft-data/rdt_data \
# --task_name $1 \
# --episode_idx $2

python visualize_episodes.py \
--dataset_dir /home/agilex/cobot_magic/data \
--task_name $1 \
--episode_idx $2

# sh visualize.sh fold-cloth 
# sh visualize.sh tian_fold_cloth 0