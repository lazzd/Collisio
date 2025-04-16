import os
import time
from argparse import Namespace

import json

import random

from tqdm import tqdm

import statistics

import torch
import torch.nn.functional as F

from torchvision import transforms

from sim_attack import SIMAttacker, PGD_linfSimImageAttacker, TextCopium, TextCaptionLlmsAug, TextPredLlmsAug

from utils.utils_path import get_model_name, get_dataset_name, verify_and_create_path, get_subset_size_str, get_config_attack_path_str, get_date, get_uuid

from utils.utils_statistic import AttackManager, safe_variance


# -----

IMAGE_ATTACK_DICT = {
    "PGD_linf": PGD_linfSimImageAttacker
}

TEXT_SIMILARITY_DICT = {
    "TextCopium": TextCopium,
    "TextCaptionLlmsAug": TextCaptionLlmsAug,
    "TextPredLlmsAug": TextPredLlmsAug
}

# -----

def save_configurations_to_json(config_retrieval, config_attack, args, path_save_results):
    try:
        retrieval_path = os.path.join(path_save_results, "config_retrieval.json")
        with open(retrieval_path, 'w') as f:
            json.dump(config_retrieval, f, indent=4)
        
        attack_path = os.path.join(path_save_results, "config_attack.json")
        with open(attack_path, 'w') as f:
            json.dump(config_attack, f, indent=4)
        
        if isinstance(args, Namespace):
            args = vars(args)
        args_path = os.path.join(path_save_results, "args.json")
        with open(args_path, 'w') as f:
            json.dump(args, f, indent=4)
    
    except Exception as e:
        print(f"Error: {e}")

# -----

def retrieval_eval(clean_sims_dict, clean_poisoning_image_dict, adv_poisoning_image_dict, caption_id, ret_idx_in_list, true_image_id, pois_image_id, top_k=10):
    clean_sims_caption2i = clean_sims_dict["clean_sims_t2i"][caption_id]

    sim_caption2adv_i = adv_poisoning_image_dict["adv_sims_caption2adv_img"][ret_idx_in_list]
    
    # ----------
    ir_1_top_k_results = {}
    # ----------

    # ----------
    # for clean eval
    if pois_image_id is None:
        clean_sim_caption2adv_i = clean_poisoning_image_dict["clean_sims_caption2adv_img"][ret_idx_in_list]
        eval_clean_sims_caption2i = clean_sims_caption2i.clone()
        eval_clean_sims_caption2i = torch.cat([eval_clean_sims_caption2i, torch.tensor([clean_sim_caption2adv_i], device=eval_clean_sims_caption2i.device)])
        pois_clean_adv_id = len(eval_clean_sims_caption2i) - 1
    else:
        eval_clean_sims_caption2i = clean_sims_caption2i
        pois_clean_adv_id = pois_image_id
    
    clean_rank_sims = torch.argsort(eval_clean_sims_caption2i, descending=True)

    clean_ir_list = []
    for i in range(1, top_k+1):
        top_n_clean_indices = clean_rank_sims[:i]
        clean_ir = pois_clean_adv_id in top_n_clean_indices
        clean_ir_list.append(clean_ir)
    ir_1_top_k_results["clean_ir_list"] = clean_ir_list
    
    clean_acc_ir_list = []
    for i in range(1, top_k+1):
        top_n_clean_acc_indices = clean_rank_sims[:i]
        clean_ir_acc = true_image_id in top_n_clean_acc_indices
        clean_acc_ir_list.append(clean_ir_acc)
    ir_1_top_k_results["clean_acc_ir_list"] = clean_acc_ir_list
    # ----------

    # ----------
    # for adv eval
    adv_sims_caption2i = clean_sims_caption2i.clone()
    adv_sims_caption2i = torch.cat([adv_sims_caption2i, torch.tensor([sim_caption2adv_i], device=adv_sims_caption2i.device)])
    pois_adv_id = len(adv_sims_caption2i) - 1

    adv_rank_sims = torch.argsort(adv_sims_caption2i, descending=True)

    adv_ir_list = []
    for i in range(1, top_k+1):
        top_n_adv_indices = adv_rank_sims[:i]
        adv_ir = pois_adv_id in top_n_adv_indices
        adv_ir_list.append(adv_ir)
    ir_1_top_k_results["adv_ir_list"] = adv_ir_list

    adv_acc_ir_list = []
    for i in range(1, top_k+1):
        top_n_adv_acc_indices = adv_rank_sims[:i]
        adv_acc_ir = true_image_id in top_n_adv_acc_indices
        adv_acc_ir_list.append(adv_acc_ir)
    ir_1_top_k_results["adv_acc_ir_list"] = adv_acc_ir_list
    # ----------

    return ir_1_top_k_results

# -----

def retrieval_eval_itm(model, clean_sims_dict, clean_poisoning_image_dict, adv_poisoning_image_dict, caption_id, true_image_id, pois_image_id, device, k_test=128, batch_size=128, top_k=10):
    clean_sims_caption2i_itm = clean_sims_dict["clean_sims_t2i_itm"][caption_id]

    # -----
    clean_text_features_dict = clean_sims_dict["clean_text_features_dict"]

    text_embeds_caption_id = clean_text_features_dict["text_embeds"][caption_id].unsqueeze(0).to(device) # roba dei testi, dovrebbero essere sempre fermi...
    text_ids_caption_id = clean_text_features_dict["text_ids"][caption_id].unsqueeze(0).to(device) # roba dei testi, dovrebbero essere sempre fermi...
    text_atts_caption_id = clean_text_features_dict["text_atts"][caption_id].unsqueeze(0).to(device) # roba dei testi, dovrebbero essere sempre fermi...
    # -----

    # -----
    clean_image_features_dict = clean_sims_dict["clean_image_features_dict"]
    # -----

    # ----------
    ir_1_top_k_itm_results = {}
    # ----------

    # ----------
    # for clean eval
    if pois_image_id is None:
        eval_clean_vit_feat = clean_poisoning_image_dict["clean_pois_vit_feat"]
        eval_clean_image_embed = clean_poisoning_image_dict["clean_pois_image_embed"]

        eval_clean_vit_feats = clean_image_features_dict["clean_vit_feats"].clone()
        eval_clean_vit_feats = torch.cat([eval_clean_vit_feats, eval_clean_vit_feat.to(eval_clean_vit_feats.device).unsqueeze(0)], dim=0)

        eval_clean_images_embeds = clean_image_features_dict["clean_image_embeds"].clone()

        eval_clean_images_embeds = torch.cat([eval_clean_images_embeds, eval_clean_image_embed.to(eval_clean_images_embeds.device).unsqueeze(0)], dim=0)

        assert len(eval_clean_vit_feats) == len(eval_clean_images_embeds)
        pois_clean_adv_id = len(eval_clean_vit_feats) - 1

        clean_sims_matrix = []
        for image_embed in eval_clean_images_embeds:
            image_embed = image_embed.to(device)
            clean_sim_q2t = image_embed @ text_embeds_caption_id.t()
            clean_sim_i2t, _ = clean_sim_q2t.max(0)
            clean_sims_matrix.append(clean_sim_i2t)
        clean_sims_matrix = torch.stack(clean_sims_matrix, dim=0)

        clean_sims_matrix = clean_sims_matrix.t()
        clean_score_matrix_t2i = torch.full((1, len(eval_clean_images_embeds)), -100.0, device=device)

        for i, clean_sims in enumerate(clean_sims_matrix):
            topk_sim, topk_idx = clean_sims.topk(k=k_test, dim=0)

            for batch_start in range(0, k_test, batch_size):
                batch_end = min(batch_start + batch_size, k_test)
                topk_idx_batch = topk_idx[batch_start:batch_end].cpu()

                image_inputs = eval_clean_vit_feats[topk_idx_batch].to(device)

                with torch.no_grad():
                    score = model.compute_itm(
                        image_inputs=image_inputs,
                        text_ids=text_ids_caption_id[i].repeat(batch_end - batch_start, 1),
                        text_atts=text_atts_caption_id[i].repeat(batch_end - batch_start, 1),
                    ).float()
                clean_score_matrix_t2i[i, topk_idx_batch] = score + topk_sim[batch_start:batch_end]

                del score, image_inputs
                torch.cuda.empty_cache()

        eval_clean_sims_caption2i_itm = clean_score_matrix_t2i.squeeze(0)
        
    else:
        eval_clean_sims_caption2i_itm = clean_sims_caption2i_itm
        pois_clean_adv_id = pois_image_id
    # ----------

    clean_rank_sims = torch.argsort(eval_clean_sims_caption2i_itm, descending=True)

    clean_ir_list_itm = []
    for i in range(1, top_k+1):
        top_n_clean_indices = clean_rank_sims[:i]
        clean_ir = pois_clean_adv_id in top_n_clean_indices
        clean_ir_list_itm.append(clean_ir)
    ir_1_top_k_itm_results["clean_ir_list_itm"] = clean_ir_list_itm
    
    clean_acc_ir_list_itm = []
    for i in range(1, top_k+1):
        top_n_clean_acc_indices = clean_rank_sims[:i]
        clean_acc_ir = true_image_id in top_n_clean_acc_indices
        clean_acc_ir_list_itm.append(clean_acc_ir)
    ir_1_top_k_itm_results["clean_acc_ir_list_itm"] = clean_acc_ir_list_itm

    # ----------

    # ----------
    # for adv eval
    adv_image_vit_feat = adv_poisoning_image_dict["vit_feat"]
    adv_image_embed = adv_poisoning_image_dict["image_embed"]

    adv_vit_feats = clean_image_features_dict["clean_vit_feats"].clone()
    adv_vit_feats = torch.cat([adv_vit_feats, adv_image_vit_feat.to(adv_vit_feats.device).unsqueeze(0)], dim=0)

    adv_images_embeds = clean_image_features_dict["clean_image_embeds"].clone()
    
    adv_images_embeds = torch.cat([adv_images_embeds, adv_image_embed.to(adv_images_embeds.device).unsqueeze(0)], dim=0)

    assert len(adv_vit_feats) == len(adv_images_embeds)
    pois_adv_id = len(adv_vit_feats) - 1

    adv_sims_matrix = []
    for image_embed in adv_images_embeds:
        image_embed = image_embed.to(device)
        adv_sim_q2t = image_embed @ text_embeds_caption_id.t()
        adv_sim_i2t, _ = adv_sim_q2t.max(0)
        adv_sims_matrix.append(adv_sim_i2t)
    adv_sims_matrix = torch.stack(adv_sims_matrix, dim=0)

    adv_sims_matrix = adv_sims_matrix.t()
    adv_score_matrix_t2i = torch.full((1, len(adv_images_embeds)), -100.0, device=device)

    for i, adv_sims in enumerate(adv_sims_matrix):
        topk_sim, topk_idx = adv_sims.topk(k=k_test, dim=0)

        for batch_start in range(0, k_test, batch_size):
            batch_end = min(batch_start + batch_size, k_test)
            topk_idx_batch = topk_idx[batch_start:batch_end].cpu()

            image_inputs = adv_vit_feats[topk_idx_batch].to(device)

            with torch.no_grad():
                score = model.compute_itm(
                    image_inputs=image_inputs,
                    text_ids=text_ids_caption_id[i].repeat(batch_end - batch_start, 1),
                    text_atts=text_atts_caption_id[i].repeat(batch_end - batch_start, 1),
                ).float()
            adv_score_matrix_t2i[i, topk_idx_batch] = score + topk_sim[batch_start:batch_end]

            del score, image_inputs
            torch.cuda.empty_cache()

    adv_score_matrix_t2i_id = adv_score_matrix_t2i.squeeze(0)
    # ----------
    
    adv_rank_sims = torch.argsort(adv_score_matrix_t2i_id, descending=True)

    adv_ir_list_itm = []
    for i in range(1, top_k+1):
        top_n_adv_indices = adv_rank_sims[:i]
        adv_ir = pois_adv_id in top_n_adv_indices
        adv_ir_list_itm.append(adv_ir)
    ir_1_top_k_itm_results["adv_ir_list_itm"] = adv_ir_list_itm
    
    adv_acc_ir_list_itm = []
    for i in range(1, top_k+1):
        top_n_adv_acc_indices = adv_rank_sims[:i]
        adv_acc_ir = true_image_id in top_n_adv_acc_indices
        adv_acc_ir_list_itm.append(adv_acc_ir)
    ir_1_top_k_itm_results["adv_acc_ir_list_itm"] = adv_acc_ir_list_itm

    return ir_1_top_k_itm_results

# -----

def add_in_ir_1_top_k_results_dict_all(ir_1_top_k_results_dict, ir_1_top_k_results_dict_all):
    for key, val_list in ir_1_top_k_results_dict.items():
        all_key = f"{key}"
        if all_key not in ir_1_top_k_results_dict_all:
            ir_1_top_k_results_dict_all[all_key] = []
        ir_1_top_k_results_dict_all[all_key].append(val_list)

def compute_means_for_each_key(ir_1_top_k_results_dict_all):
    means_dict = {}
    for key, lists in ir_1_top_k_results_dict_all.items():
        transposed = zip(*lists)
        means_dict[key] = [sum(column) / len(column) for column in transposed]
    return means_dict

# -----

def retrieval_attack(model, m_tokenizer, ref_model, tokenizer, data_loader, img_preprocess, clean_sims_dict, device, args, config, path_save_results, top_k=10):
    model.eval()

    to_pil = transforms.ToPILImage()

    model_name = get_model_name(args)
    dataset_name = get_dataset_name(args, config)

    config_retrieval = config["config_retrieval"]
    config_attack = config["config_attack"]
    
    subset_size_val = get_subset_size_str(args)

    config_attack_upperpath_str, config_attack_exp_name_str = get_config_attack_path_str(args)

    date = get_date()
    uuid = get_uuid()

    test_file_name = os.path.basename(config["config_retrieval"]["test_file"])

    path_results_adv = os.path.join(path_save_results, f"model:{model_name}", "valuations", "adversarial", f"dataset:{dataset_name}", f"seed:{args.seed}", f"{subset_size_val}", f"(config::{config_attack_upperpath_str})", f"{config_attack_exp_name_str}.{date}_{uuid}")

    path_results_adv_img = os.path.join(path_results_adv, "adv_imgs")
    verify_and_create_path(path_results_adv_img)

    attack_manager = AttackManager(
        args, 
        config_retrieval,
        config_attack,
        top_k=top_k, 
        use_itm=config.get("use_itm", False)
    )

    # ----------
    image_attacker = IMAGE_ATTACK_DICT[config_attack['image_attack_name']](preprocess=img_preprocess, verbose=args.verbose, **config_attack['image_attack_params'])

    if config_attack['txt_similarity_name'] == 'TextPredLlmsAug':
        if config["use_itm"]:
            clean_sims_t2i_mat = clean_sims_dict['clean_sims_t2i_itm']
        else:
            clean_sims_t2i_mat = clean_sims_dict['clean_sims_t2i']
        image_list_name = data_loader.dataset.image
        txt_attacker = TEXT_SIMILARITY_DICT[config_attack['txt_similarity_name']](clean_sims_t2i_mat=clean_sims_t2i_mat, image_list_name=image_list_name, test_file_name=test_file_name, ref_net=ref_model, tokenizer=tokenizer, cls=args.cls, device=device, **config_attack['txt_similarity_params'])
    elif config_attack['txt_similarity_name'] == 'TextCaptionLlmsAug':
        txt_attacker = TEXT_SIMILARITY_DICT[config_attack['txt_similarity_name']](test_file_name=test_file_name, dataset_name=dataset_name, seed=args.seed, subset_size_val=subset_size_val, ref_net=ref_model, tokenizer=tokenizer, cls=args.cls, device=device, **config_attack['txt_similarity_params'])
    else:
        txt_attacker = TEXT_SIMILARITY_DICT[config_attack['txt_similarity_name']](ref_net=ref_model, tokenizer=tokenizer, cls=args.cls, device=device, **config_attack['txt_similarity_params'])

    attacker = SIMAttacker(model, tokenizer, image_attacker, txt_attacker)
    # ----------

    start_time = time.time()

    for batch_idx, (captions_group, ret_captions_list_group, true_images, pois_images, captions_ids, ret_captions_list_group_ids, true_images_ids, pois_images_ids, pois_images_internal_ids, true_images_names) in enumerate(tqdm(data_loader)):
        print(f'--------------------> batch:{batch_idx}/{len(data_loader)}')

        # ---------------------------------
        # ---------------------------------

        images_adv, texts_args, texts_init = attacker.sim_attack(pois_images, captions_group, num_iters=config_attack['image_attack_num_iter'], true_images=true_images, images_names=true_images_names, indexes_captions=captions_ids)

        # ---------------------------------
        # Embeds images_adv
        # ---------------------------------
        with torch.no_grad():
            images_adv_norm = img_preprocess(images_adv).to(device)
            images_adv_output = model.inference_image(images_adv_norm)

        # ---------------------------------

        for idx in range(images_adv.shape[0]):

            # ----------
            caption = captions_group[idx]
            caption_id = captions_ids[idx]

            text_init = texts_init[idx]
            texts_arg = texts_args[idx]

            true_image_id = true_images_ids[idx]

            if config_attack.get("query-to-query", False):
                ret_captions_list = [caption]
                ret_captions_ids_list = [caption_id]
            else:
                ret_captions_list_complete = ret_captions_list_group[idx]
                ret_captions_ids_list_complete = ret_captions_list_group_ids[idx]
                num_ret_captions = config_attack.get("num_ret_captions", 1)
                if num_ret_captions == 'All' or num_ret_captions >= len(ret_captions_list_complete):
                    ret_captions_list = ret_captions_list_complete
                    ret_captions_ids_list = ret_captions_ids_list_complete
                else:
                    ret_captions_idx_sampled = random.sample(range(len(ret_captions_list_complete)), num_ret_captions)
                    ret_captions_list = [ret_captions_list_complete[i] for i in ret_captions_idx_sampled]
                    ret_captions_ids_list = [ret_captions_ids_list_complete[i] for i in ret_captions_idx_sampled]

            image_adv = images_adv[idx].unsqueeze(0)
            # ----------
            image_adv_feat = images_adv_output['image_feat'][idx].float().detach()
            # ----------
            pois_image = pois_images[idx].unsqueeze(0)
            pois_image_id = pois_images_ids[idx]
            pois_image_internal_id = pois_images_internal_ids[idx]

            # ----------
            # RET CAPTIONS FEATS
            # ----------

            with torch.no_grad():
                captions_m_input = m_tokenizer(ret_captions_list).to(device)
                captions_m_output = model.encode_text(captions_m_input)
                if model_name == "BLIP-2":
                    captions_feats = F.normalize(model.text_proj(captions_m_output), dim=-1)
                else:
                    captions_feats = F.normalize(captions_m_output, dim=-1).float()

            # ----------
            # ADV POISONING IMAGE
            adv_poisoning_image_dict = {}

            if model_name == 'BLIP-2':
                all_adv_sims_caption2adv_img = image_adv_feat @ captions_feats.t()
                adv_sims_caption2adv_img, _ = all_adv_sims_caption2adv_img.max(dim=0)
            else:
                adv_sims_caption2adv_img = image_adv_feat @ captions_feats.t()
            
            adv_poisoning_image_dict["adv_sims_caption2adv_img"] = adv_sims_caption2adv_img

            if model_name == 'BLIP-2':
                adv_poisoning_image_dict["image_embed"] = images_adv_output['image_feat'][idx].float().detach()
                adv_poisoning_image_dict["vit_feat"] = images_adv_output['vit_feat'][idx].float().detach()
            # ----------
        
            # ----------
            # CLEAN POISONING IMAGE
            clean_poisoning_image_dict = {}

            if pois_image_internal_id is None:
                with torch.no_grad():
                    clean_image_adv_norm = img_preprocess(pois_image).to(device)
                    clean_images_adv_output = model.inference_image(clean_image_adv_norm)
                    
                clean_image_adv_feat = clean_images_adv_output['image_feat'].float().detach()
                if model_name == 'BLIP-2':
                    all_clean_sims_caption2adv_img = (clean_image_adv_feat @ captions_feats.t()).squeeze(0)
                    clean_sims_caption2adv_img, _ = all_clean_sims_caption2adv_img.max(dim=0)
                else:
                    clean_sims_caption2adv_img = (clean_image_adv_feat @ captions_feats.t()).squeeze(0)

                clean_poisoning_image_dict["clean_sims_caption2adv_img"] = clean_sims_caption2adv_img

                if model_name == 'BLIP-2':
                    clean_poisoning_image_dict["clean_pois_image_embed"] = clean_images_adv_output['image_feat'].float().detach().squeeze(0)
                    clean_poisoning_image_dict["clean_pois_vit_feat"] = clean_images_adv_output['vit_feat'].float().detach().squeeze(0)

            # --------------
            # EMDEDDINGS TEXTS ARG
            # --------------
            with torch.no_grad():
                texts_arg_input = m_tokenizer(texts_arg).to(device)
                texts_arg_output = model.encode_text(texts_arg_input)
                if model_name == "BLIP-2":
                    texts_arg_feats = F.normalize(model.text_proj(texts_arg_output), dim=-1)
                else:
                    texts_arg_feats = F.normalize(texts_arg_output, dim=-1).float()
            # --------------

            # ----------
            adv_image_name = f"TxtID-{caption_id}_ImgTrue-{true_image_id}_ImgAdv-{pois_image_id}.jpg"
            adv_image_name_tensor_dict = f"TxtID-{caption_id}_ImgTrue-{true_image_id}_ImgAdv-{pois_image_id}.pt"
            # ----------

            adv_tensor_dict = {
                "caption": caption,
                "caption_id": caption_id,
                "text_init":text_init,
                "texts_arg": texts_arg, # list
                "ret_captions": []
            }

            ret_captions_centroid_dist_list = []
            ret_captions_texts_arg_cosine_sim_centroid_list = []
            ret_captions_centroid_cosine_sim_list = []

            ir_1_top_k_results_dict_all = {}
            if config["use_itm"] and model_name == "BLIP-2":
                ir_1_top_k_itm_results_dict_all = {}

            for ret_idx in range(len(ret_captions_list)):

                ret_caption = ret_captions_list[ret_idx]
                ret_caption_id = ret_captions_ids_list[ret_idx]
                caption_feat = captions_feats[ret_idx]

                # ----------
                ret_caption_dict = {
                    "ret_caption": ret_caption,
                    "ret_caption_id": ret_caption_id
                }

                caption_texts_arg_distances = torch.norm(caption_feat - texts_arg_feats, p=2, dim=1)
                ret_caption_dict["caption_texts_arg_distances"] = caption_texts_arg_distances.cpu()

                texts_arg_feats_centroid = torch.mean(texts_arg_feats, dim=0)
                ret_caption_dict["texts_arg_feats_centroid"] = texts_arg_feats_centroid.cpu()
                
                caption_centroid_dist = torch.norm(caption_feat - texts_arg_feats_centroid, p=2)
                ret_caption_dict["caption_centroid_dist"] = caption_centroid_dist.cpu()
                ret_captions_centroid_dist_list.append(caption_centroid_dist.item())

                caption_texts_arg_cosine_sim = texts_arg_feats @ caption_feat
                ret_caption_dict["caption_texts_arg_cosine_sim"] = caption_texts_arg_cosine_sim.cpu()

                caption_texts_arg_cosine_sim_centroid = torch.mean(caption_texts_arg_cosine_sim, dim=0)
                ret_caption_dict["caption_texts_arg_cosine_sim_centroid"] = caption_texts_arg_cosine_sim_centroid.cpu()
                ret_captions_texts_arg_cosine_sim_centroid_list.append(caption_texts_arg_cosine_sim_centroid.item())

                caption_centroid_cosine_sim = (texts_arg_feats_centroid @ caption_feat)
                ret_caption_dict["caption_centroid_cosine_sim"] = caption_centroid_cosine_sim.cpu()
                ret_captions_centroid_cosine_sim_list.append(caption_centroid_cosine_sim.item())

                adv_tensor_dict["ret_captions"].append(ret_caption_dict)
                # ----------

                ir_1_top_k_results_dict = retrieval_eval(clean_sims_dict, clean_poisoning_image_dict, adv_poisoning_image_dict, ret_caption_id, ret_idx, true_image_id, pois_image_internal_id, top_k=top_k)
                add_in_ir_1_top_k_results_dict_all(ir_1_top_k_results_dict, ir_1_top_k_results_dict_all)

                if config["use_itm"] and model_name == "BLIP-2":
                    ir_1_top_k_itm_results_dict = retrieval_eval_itm(model, clean_sims_dict, clean_poisoning_image_dict, adv_poisoning_image_dict, ret_caption_id, true_image_id, pois_image_internal_id, device, k_test=128, batch_size=128, top_k=top_k)
                    add_in_ir_1_top_k_results_dict_all(ir_1_top_k_itm_results_dict, ir_1_top_k_itm_results_dict_all)
            
            adv_tensor_dict["ret_captions_centroid_dist_list"] = ret_captions_centroid_dist_list
            adv_tensor_dict["ret_captions_centroid_dist_mean"] = statistics.mean(ret_captions_centroid_dist_list)
            adv_tensor_dict["ret_captions_centroid_dist_variance"] = safe_variance(ret_captions_centroid_dist_list)
            adv_tensor_dict["ret_captions_texts_arg_cosine_sim_centroid_list"] = ret_captions_texts_arg_cosine_sim_centroid_list
            adv_tensor_dict["ret_captions_texts_arg_cosine_sim_centroid_mean"] = statistics.mean(ret_captions_texts_arg_cosine_sim_centroid_list)
            adv_tensor_dict["ret_captions_texts_arg_cosine_sim_centroid_variance"] = safe_variance(ret_captions_texts_arg_cosine_sim_centroid_list)
            adv_tensor_dict["ret_captions_centroid_cosine_sim_list"] = ret_captions_centroid_cosine_sim_list
            adv_tensor_dict["ret_captions_centroid_cosine_sim_mean"] = statistics.mean(ret_captions_centroid_cosine_sim_list)
            adv_tensor_dict["ret_captions_centroid_cosine_sim_variance"] = safe_variance(ret_captions_centroid_cosine_sim_list)

            ir_1_top_k_results_dict_mean = compute_means_for_each_key(ir_1_top_k_results_dict_all)
            if config["use_itm"] and model_name == "BLIP-2":
                ir_1_top_k_itm_results_dict_mean = compute_means_for_each_key(ir_1_top_k_itm_results_dict_all)
                attack_manager.add_attack_result(
                    caption=caption,
                    caption_id=caption_id,
                    ret_captions=ret_captions_list,
                    ret_captions_ids=ret_captions_ids_list,
                    true_image_id=true_image_id,
                    adv_image_id=pois_image_id,
                    adv_image_id_internal=pois_image_internal_id,
                    adv_image_name=adv_image_name,
                    text_init=text_init,
                    texts_arg=texts_arg,
                    adv_sims=adv_sims_caption2adv_img.tolist(),
                    ir_1_top_k_results_dict=ir_1_top_k_results_dict_mean,
                    ir_1_top_k_results_dict_all=ir_1_top_k_results_dict_all,
                    ir_1_top_k_itm_results_dict=ir_1_top_k_itm_results_dict_mean,
                    ir_1_top_k_itm_results_dict_all=ir_1_top_k_itm_results_dict_all)
            else:
                attack_manager.add_attack_result(
                    caption=caption,
                    caption_id=caption_id,
                    ret_captions=ret_captions_list,
                    ret_captions_ids=ret_captions_ids_list,
                    true_image_id=true_image_id,
                    adv_image_id=pois_image_id,
                    adv_image_id_internal=pois_image_internal_id,
                    adv_image_name=adv_image_name,
                    text_init=text_init,
                    texts_arg=texts_arg,
                    adv_sims=adv_sims_caption2adv_img.tolist(),
                    ir_1_top_k_results_dict=ir_1_top_k_results_dict_mean,
                    ir_1_top_k_results_dict_all=ir_1_top_k_results_dict_all)

            path_adv_image = os.path.join(path_results_adv_img, adv_image_name)
            adv_image = to_pil(image_adv.squeeze(0))
            adv_image.save(path_adv_image)
    
            path_adv_image_tensor_dict = os.path.join(path_results_adv_img, adv_image_name_tensor_dict)
            torch.save(adv_tensor_dict, path_adv_image_tensor_dict)

    end_time = time.time()
    total_time = end_time - start_time
    attack_manager.set_time(total_time)

    attack_manager.compute_averages()

    adv_results = attack_manager.get_results()

    path_adv_results = os.path.join(path_results_adv, "adv_results.json")
    with open(path_adv_results, 'w') as json_file:
        json.dump(adv_results, json_file, indent=4)
    
    save_configurations_to_json(config_retrieval, config_attack, args, path_results_adv)

# -----

def eval_attack(model, m_tokenizer, ref_model, tokenizer, data_loader, img_preprocess, clean_sims_dict, device, args, config, path_save_results):
    quant_source_model = getattr(args, 'quant_source_model', None)
    if quant_source_model is None:
        model = model.to(device)
    
    ref_model = ref_model.to(device)

    retrieval_attack(model, m_tokenizer, ref_model, tokenizer, data_loader, img_preprocess, clean_sims_dict, device, args, config, path_save_results)