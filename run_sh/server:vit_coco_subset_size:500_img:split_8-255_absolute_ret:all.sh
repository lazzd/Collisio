#!/bin/bash

# 0. accuracy (create clean_files)
# CUDA_VISIBLE_DEVICES=0 python accuracy.py --subset_size 500 --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32

# ----------

# Base (|S| = 0)
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/no_aug/absolute_configs_8-255/ret:all/config_attack_exp1.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32

# ----------

# Query-to-query (|S| = 0)
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/no_aug_query_to_query/absolute_configs_8-255/config_attack_exp1.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32

# ----------
# SYN
# ----------

# Cosine context
# # Syn
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/syn/absolute_configs_8-255/ret:all/aug:least/choice:cosine_context/final:loss/config_attack_exp1.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/syn/absolute_configs_8-255/ret:all/aug:least/choice:cosine_context/final:loss/config_attack_exp2.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/syn/absolute_configs_8-255/ret:all/aug:least/choice:cosine_context/final:loss/config_attack_exp3.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/syn/absolute_configs_8-255/ret:all/aug:least/choice:cosine_context/final:loss/config_attack_exp4.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32

# ----------

# Random
# # Syn
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/syn/absolute_configs_8-255/ret:all/aug:least/choice:random/config_attack_exp1.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/syn/absolute_configs_8-255/ret:all/aug:least/choice:random/config_attack_exp2.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/syn/absolute_configs_8-255/ret:all/aug:least/choice:random/config_attack_exp3.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/syn/absolute_configs_8-255/ret:all/aug:least/choice:random/config_attack_exp4.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32

# ----------
# CAPTION MISTRAL
# ----------

# Cosine context
# # Caption Llms
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp1.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp2.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp3.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32

# ----------

# Random
# # Caption Llms
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:random/config_attack_exp1.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:random/config_attack_exp2.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:random/config_attack_exp3.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/caption_llms/absolute_configs_8-255/ret:all/choice:random/config_attack_exp4.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32

# ----------
# IC PRED RET 1
# ----------

# Cosine context
# # Pred Augs Llms
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/pred_ic/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp1.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/pred_ic/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp2.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/pred_ic/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp3.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/pred_ic/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32

# ----------

# Random
# # Pred Augs Llms
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/pred_ic/absolute_configs_8-255/ret:all/choice:random/config_attack_exp1.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/pred_ic/absolute_configs_8-255/ret:all/choice:random/config_attack_exp2.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/pred_ic/absolute_configs_8-255/ret:all/choice:random/config_attack_exp3.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/pred_ic/absolute_configs_8-255/ret:all/choice:random/config_attack_exp4.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32

# ----------
# IC PRED RET 5
# ----------

# Cosine context
# # Pred Augs Llms
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/pred_ic_ret5/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp1.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/pred_ic_ret5/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp2.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/pred_ic_ret5/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp3.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/pred_ic_ret5/absolute_configs_8-255/ret:all/choice:cosine_context/config_attack_exp4.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32

# ----------

# Random
# # Pred Augs Llms
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/pred_ic_ret5/absolute_configs_8-255/ret:all/choice:random/config_attack_exp1.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/pred_ic_ret5/absolute_configs_8-255/ret:all/choice:random/config_attack_exp2.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/pred_ic_ret5/absolute_configs_8-255/ret:all/choice:random/config_attack_exp3.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32
CUDA_VISIBLE_DEVICES=0 python attack.py --subset_size 500 --config_attack './configs/pred_ic_ret5/absolute_configs_8-255/ret:all/choice:random/config_attack_exp4.yaml' --config_retrieval './configs/Retrieval_coco_server.yaml' --batch_size 32