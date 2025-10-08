#!/bin/bash

# 0. accuracy (create clean_files)
CUDA_VISIBLE_DEVICES=0 python accuracy.py --subset_size 500 --config_retrieval './configs/Retrieval_flickr_server.yaml' --batch_size 32 --source_model 'FARE2'

# ----------
# IC PRED RET 5
# ----------

# Cosine context
# # Pred Augs Llms
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/pred_ic_ret5/absolute_configs_4-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --config_retrieval './configs/Retrieval_flickr_server.yaml' --batch_size 32 --source_model 'FARE2'

CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/pred_ic_ret5/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --config_retrieval './configs/Retrieval_flickr_server.yaml' --batch_size 32 --source_model 'FARE2'

CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/pred_ic_ret5/absolute_configs_16-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --config_retrieval './configs/Retrieval_flickr_server.yaml' --batch_size 32 --source_model 'FARE2'

CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/pred_ic_ret5/absolute_configs_24-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --config_retrieval './configs/Retrieval_flickr_server.yaml' --batch_size 32 --source_model 'FARE2'

# ----------