#!/bin/bash

# # ----------
# VIT
# # ----------
CUDA_VISIBLE_DEVICES=1 python accuracy.py --subset_size 500 --config_retrieval './configs/Retrieval_flickr_server.yaml' --batch_size 32

CUDA_VISIBLE_DEVICES=1 python attack_acc_ind_choice.py --subset_size 500 --config_retrieval './configs/Retrieval_flickr_server.yaml' --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --batch_size 1 --num_to_pois 30

# ----------
CUDA_VISIBLE_DEVICES=1 python accuracy.py --subset_size 500 --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32

CUDA_VISIBLE_DEVICES=1 python attack_acc_ind_choice.py --subset_size 500 --config_retrieval './configs/Retrieval_coco_server.yaml' --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --batch_size 1 --num_to_pois 30


# # ----------
# CNN
# # ----------
CUDA_VISIBLE_DEVICES=1 python accuracy.py --subset_size 500 --config_retrieval './configs/Retrieval_flickr_server.yaml' --batch_size 32 --source_model 'RN101'

CUDA_VISIBLE_DEVICES=1 python attack_acc_ind_choice.py --subset_size 500 --config_retrieval './configs/Retrieval_flickr_server.yaml' --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --batch_size 1 --num_to_pois 30 --source_model 'RN101'

# ----------
CUDA_VISIBLE_DEVICES=1 python accuracy.py --subset_size 500 --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32 --source_model 'RN101'

CUDA_VISIBLE_DEVICES=1 python attack_acc_ind_choice.py --subset_size 500 --config_retrieval './configs/Retrieval_coco_server.yaml' --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --batch_size 1 --num_to_pois 30 --source_model 'RN101'

# # ----------
# FARE2
# # ----------
CUDA_VISIBLE_DEVICES=1 python accuracy.py --subset_size 500 --config_retrieval './configs/Retrieval_flickr_server.yaml' --batch_size 32 --source_model 'FARE2'

CUDA_VISIBLE_DEVICES=1 python attack_acc_ind_choice.py --subset_size 500 --config_retrieval './configs/Retrieval_flickr_server.yaml' --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --batch_size 1 --num_to_pois 30 --source_model 'FARE2'


# ----------
CUDA_VISIBLE_DEVICES=1 python accuracy.py --subset_size 500 --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32 --source_model 'FARE2'

CUDA_VISIBLE_DEVICES=1 python attack_acc_ind_choice.py --subset_size 500 --config_retrieval './configs/Retrieval_coco_server.yaml' --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --batch_size 1 --num_to_pois 30 --source_model 'FARE2'

# # ----------
# FARE4
# # ----------
CUDA_VISIBLE_DEVICES=1 python accuracy.py --subset_size 500 --config_retrieval './configs/Retrieval_flickr_server.yaml' --batch_size 32 --source_model 'FARE4'

CUDA_VISIBLE_DEVICES=1 python attack_acc_ind_choice.py --subset_size 500 --config_retrieval './configs/Retrieval_flickr_server.yaml' --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --batch_size 1 --num_to_pois 30 --source_model 'FARE4'

# ----------
CUDA_VISIBLE_DEVICES=1 python accuracy.py --subset_size 500 --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32 --source_model 'FARE4'

CUDA_VISIBLE_DEVICES=1 python attack_acc_ind_choice.py --subset_size 500 --config_retrieval './configs/Retrieval_coco_server.yaml' --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --batch_size 1 --num_to_pois 30 --source_model 'FARE4'

# # ----------
# JPG99
# # ----------
CUDA_VISIBLE_DEVICES=1 python accuracy.py --subset_size 500 --config_retrieval './configs/Retrieval_flickr_server.yaml' --batch_size 32 --jpg_quality 99

CUDA_VISIBLE_DEVICES=1 python attack_acc_ind_choice.py --subset_size 500 --config_retrieval './configs/Retrieval_flickr_server.yaml' --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --batch_size 1 --num_to_pois 30 --jpg_quality 99

# ----------
CUDA_VISIBLE_DEVICES=1 python accuracy.py --subset_size 500 --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32 --jpg_quality 99

CUDA_VISIBLE_DEVICES=1 python attack_acc_ind_choice.py --subset_size 500 --config_retrieval './configs/Retrieval_coco_server.yaml' --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --batch_size 1 --num_to_pois 30 --jpg_quality 99

# # ----------
# JPG94
# # ----------
CUDA_VISIBLE_DEVICES=1 python accuracy.py --subset_size 500 --config_retrieval './configs/Retrieval_flickr_server.yaml' --batch_size 32 --jpg_quality 94

CUDA_VISIBLE_DEVICES=1 python attack_acc_ind_choice.py --subset_size 500 --config_retrieval './configs/Retrieval_flickr_server.yaml' --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --batch_size 1 --num_to_pois 30 --jpg_quality 94

# ----------
CUDA_VISIBLE_DEVICES=1 python accuracy.py --subset_size 500 --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32 --jpg_quality 94

CUDA_VISIBLE_DEVICES=1 python attack_acc_ind_choice.py --subset_size 500 --config_retrieval './configs/Retrieval_coco_server.yaml' --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --batch_size 1 --num_to_pois 30 --jpg_quality 94

# # ----------
# JPG89
# # ----------
CUDA_VISIBLE_DEVICES=1 python accuracy.py --subset_size 500 --config_retrieval './configs/Retrieval_flickr_server.yaml' --batch_size 32 --jpg_quality 89

CUDA_VISIBLE_DEVICES=1 python attack_acc_ind_choice.py --subset_size 500 --config_retrieval './configs/Retrieval_flickr_server.yaml' --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --batch_size 1 --num_to_pois 30 --jpg_quality 89

# ----------
CUDA_VISIBLE_DEVICES=1 python accuracy.py --subset_size 500 --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32 --jpg_quality 89

CUDA_VISIBLE_DEVICES=1 python attack_acc_ind_choice.py --subset_size 500 --config_retrieval './configs/Retrieval_coco_server.yaml' --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --batch_size 1 --num_to_pois 30 --jpg_quality 89

# # ----------
# BLIP-2
# # ----------
CUDA_VISIBLE_DEVICES=1 python accuracy.py --subset_size 500 --config_retrieval './configs/Retrieval_flickr_server.yaml' --batch_size 24 --source_model 'BLIP-2'

CUDA_VISIBLE_DEVICES=1 python attack_acc_ind_choice.py --subset_size 500 --config_retrieval './configs/Retrieval_flickr_server.yaml' --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --batch_size 1 --num_to_pois 30 --source_model 'BLIP-2'

# ----------
CUDA_VISIBLE_DEVICES=1 python accuracy.py --subset_size 500 --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 24 --source_model 'BLIP-2'

CUDA_VISIBLE_DEVICES=1 python attack_acc_ind_choice.py --subset_size 500 --config_retrieval './configs/Retrieval_coco_server.yaml' --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --batch_size 1 --num_to_pois 30 --source_model 'BLIP-2'