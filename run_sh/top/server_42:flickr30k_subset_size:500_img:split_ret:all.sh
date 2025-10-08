#!/bin/bash

# # ----------
# VIT
# # ----------
CUDA_VISIBLE_DEVICES=1 python accuracy.py --subset_size 500 --config_retrieval './configs/Retrieval_flickr_server.yaml' --batch_size 32

CUDA_VISIBLE_DEVICES=1 python attack_top.py --subset_size 500 --config_retrieval './configs/Retrieval_flickr_server.yaml' --config_attack './configs/pred_ic_ret5/absolute_configs_4-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --batch_size 32 --poison_insertion 1
CUDA_VISIBLE_DEVICES=1 python attack_top.py --subset_size 500 --config_retrieval './configs/Retrieval_flickr_server.yaml' --config_attack './configs/pred_ic_ret5/absolute_configs_4-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --batch_size 32 --poison_insertion 5
CUDA_VISIBLE_DEVICES=1 python attack_top.py --subset_size 500 --config_retrieval './configs/Retrieval_flickr_server.yaml' --config_attack './configs/pred_ic_ret5/absolute_configs_4-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --batch_size 32 --poison_insertion 10