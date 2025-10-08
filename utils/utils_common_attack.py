import os
from argparse import Namespace

import json

import torch

from sim_attack import SIMAttacker, PGD_linfSimImageAttacker, TextCopium, TextCaptionLlmsAug, TextPredLlmsAug


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
    rank_results = {}
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

    if true_image_id in clean_rank_sims:
        clean_rank = (clean_rank_sims == true_image_id).nonzero(as_tuple=True)[0].item() + 1
    else:
        clean_rank = None
    rank_results["clean_rank"] = clean_rank

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

    if true_image_id in adv_rank_sims:
        adv_rank = (adv_rank_sims == true_image_id).nonzero(as_tuple=True)[0].item() + 1
    else:
        adv_rank = None
    rank_results["adv_rank"] = adv_rank

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

    return ir_1_top_k_results, rank_results

# -----

def retrieval_eval_itm(model, clean_sims_dict, clean_poisoning_image_dict, adv_poisoning_image_dict, caption_id, true_image_id, pois_image_id, device, k_test=128, batch_size=128, top_k=10):
    clean_sims_caption2i_itm = clean_sims_dict["clean_sims_t2i_itm"][caption_id]

    # -----
    clean_text_features_dict = clean_sims_dict["clean_text_features_dict"]

    text_embeds_caption_id = clean_text_features_dict["text_embeds"][caption_id].unsqueeze(0).to(device)
    text_ids_caption_id = clean_text_features_dict["text_ids"][caption_id].unsqueeze(0).to(device)
    text_atts_caption_id = clean_text_features_dict["text_atts"][caption_id].unsqueeze(0).to(device)
    # -----

    # -----
    clean_image_features_dict = clean_sims_dict["clean_image_features_dict"]
    # -----

    # ----------
    rank_itm_results = {}
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

    if true_image_id in clean_rank_sims:
        clean_rank = (clean_rank_sims == true_image_id).nonzero(as_tuple=True)[0].item() + 1
    else:
        clean_rank = None
    rank_itm_results["clean_rank_itm"] = clean_rank

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

    if true_image_id in adv_rank_sims:
        adv_rank = (adv_rank_sims == true_image_id).nonzero(as_tuple=True)[0].item() + 1
    else:
        adv_rank = None
    rank_itm_results["adv_rank_itm"] = adv_rank

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

    return ir_1_top_k_itm_results, rank_itm_results

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

def retrieval_eval_top(clean_sims_dict, clean_poisoning_image_dict_sample_list, adv_poisoning_image_dict_sample_list, caption_id, ret_idx_in_list, true_image_id, pois_images_ids, top_k=10):
    clean_sims_caption2i = clean_sims_dict["clean_sims_t2i"][caption_id]

    sim_caption2adv_i_list = [adv_poisoning_image_dict["adv_sims_caption2adv_img"][ret_idx_in_list] for adv_poisoning_image_dict in adv_poisoning_image_dict_sample_list]
    
    # ----------
    rank_results = {}
    # ----------
    ir_1_top_k_results = {}
    # ----------

    # ----------
    # for clean eval
    pois_clean_adv_id_list = []
    eval_clean_sims_caption2i = clean_sims_caption2i.clone()
    idx_in = 0
    for pois_image_id in pois_images_ids:
        if pois_image_id is None:
            clean_sim_caption2adv_i = clean_poisoning_image_dict_sample_list[idx_in]["clean_sims_caption2adv_img"][ret_idx_in_list]
            eval_clean_sims_caption2i = torch.cat([eval_clean_sims_caption2i, torch.tensor([clean_sim_caption2adv_i], device=eval_clean_sims_caption2i.device)])
            pois_clean_adv_id = len(eval_clean_sims_caption2i) - 1
            idx_in += 1
        else:
            pois_clean_adv_id = pois_image_id
        pois_clean_adv_id_list.append(pois_clean_adv_id)
    
    clean_rank_sims = torch.argsort(eval_clean_sims_caption2i, descending=True)

    if true_image_id in clean_rank_sims:
        clean_rank = (clean_rank_sims == true_image_id).nonzero(as_tuple=True)[0].item() + 1
    else:
        clean_rank = None
    rank_results["clean_rank"] = clean_rank

    clean_not_ir_list = []
    for i in range(1, top_k+1):
        top_n_clean_indices = clean_rank_sims[:i]
        clean_not_ir = true_image_id not in top_n_clean_indices
        clean_not_ir_list.append(clean_not_ir)
    ir_1_top_k_results["clean_ir_list"] = clean_not_ir_list
    
    clean_acc_ir_list = []
    for i in range(1, top_k+1):
        top_n_clean_acc_indices = clean_rank_sims[:i]
        clean_ir_acc = true_image_id in top_n_clean_acc_indices
        clean_acc_ir_list.append(clean_ir_acc)
    ir_1_top_k_results["clean_acc_ir_list"] = clean_acc_ir_list

    clean_fault_ir_list = []
    for i in range(1, top_k + 1):
        top_n_clean_fault_indices = clean_rank_sims[:i]
        pois_in_top = sum(1 for pid in pois_clean_adv_id_list if pid in top_n_clean_fault_indices)
        frac = pois_in_top / i
        clean_fault_ir_list.append(frac)
    ir_1_top_k_results["clean_fault_ir_list"] = clean_fault_ir_list
    # ----------

    # ----------
    # for adv eval
    pois_adv_id_list = []
    adv_sims_caption2i = clean_sims_caption2i.clone()
    for sim_caption2adv_i in sim_caption2adv_i_list:
        adv_sims_caption2i = torch.cat([adv_sims_caption2i, torch.tensor([sim_caption2adv_i], device=adv_sims_caption2i.device)])
        pois_adv_id = len(adv_sims_caption2i) - 1
        pois_adv_id_list.append(pois_adv_id)

    adv_rank_sims = torch.argsort(adv_sims_caption2i, descending=True)

    if true_image_id in adv_rank_sims:
        adv_rank = (adv_rank_sims == true_image_id).nonzero(as_tuple=True)[0].item() + 1
    else:
        adv_rank = None
    rank_results["adv_rank"] = adv_rank

    adv_not_ir_list = []
    for i in range(1, top_k+1):
        top_n_adv_indices = adv_rank_sims[:i]
        adv_ir = true_image_id not in top_n_adv_indices
        adv_not_ir_list.append(adv_ir)
    ir_1_top_k_results["adv_ir_list"] = adv_not_ir_list

    adv_acc_ir_list = []
    for i in range(1, top_k+1):
        top_n_adv_acc_indices = adv_rank_sims[:i]
        adv_acc_ir = true_image_id in top_n_adv_acc_indices
        adv_acc_ir_list.append(adv_acc_ir)
    ir_1_top_k_results["adv_acc_ir_list"] = adv_acc_ir_list

    adv_fault_ir_list = []
    for i in range(1, top_k + 1):
        top_n_adv_fault_indices = adv_rank_sims[:i]
        pois_in_top = sum(1 for pid in pois_adv_id_list if pid in top_n_adv_fault_indices)
        frac = pois_in_top / i
        adv_fault_ir_list.append(frac)
    ir_1_top_k_results["adv_fault_ir_list"] = adv_fault_ir_list
    # ----------

    return ir_1_top_k_results, rank_results

def retrieval_eval_itm_top(model, clean_sims_dict, clean_poisoning_image_dict_sample_list, adv_poisoning_image_dict_sample_list, caption_id, true_image_id, pois_images_ids, device, k_test=128, batch_size=128, top_k=10):
    clean_sims_caption2i_itm = clean_sims_dict["clean_sims_t2i_itm"][caption_id]

    # -----
    clean_text_features_dict = clean_sims_dict["clean_text_features_dict"]

    text_embeds_caption_id = clean_text_features_dict["text_embeds"][caption_id].unsqueeze(0).to(device)
    text_ids_caption_id = clean_text_features_dict["text_ids"][caption_id].unsqueeze(0).to(device)
    text_atts_caption_id = clean_text_features_dict["text_atts"][caption_id].unsqueeze(0).to(device)
    # -----

    # -----
    clean_image_features_dict = clean_sims_dict["clean_image_features_dict"]
    # -----

    # ----------
    rank_itm_results = {}
    # ----------
    ir_1_top_k_itm_results = {}
    # ----------

    # ----------
    # for clean eval
    pois_clean_adv_id_list = []
    if any(p is None for p in pois_images_ids):
        eval_clean_vit_feats = clean_image_features_dict["clean_vit_feats"].clone()
        eval_clean_images_embeds = clean_image_features_dict["clean_image_embeds"].clone()
        idx_in = 0
        for pois_image_id in pois_images_ids:
            if pois_image_id is None:
                eval_clean_vit_feat = clean_poisoning_image_dict_sample_list[idx_in]["clean_pois_vit_feat"]
                eval_clean_image_embed = clean_poisoning_image_dict_sample_list[idx_in]["clean_pois_image_embed"]

                eval_clean_vit_feats = torch.cat([eval_clean_vit_feats, eval_clean_vit_feat.to(eval_clean_vit_feats.device).unsqueeze(0)], dim=0)

                eval_clean_images_embeds = torch.cat([eval_clean_images_embeds, eval_clean_image_embed.to(eval_clean_images_embeds.device).unsqueeze(0)], dim=0)

                assert len(eval_clean_vit_feats) == len(eval_clean_images_embeds)
                pois_clean_adv_id = len(eval_clean_vit_feats) - 1
                idx_in += 1
            else:
                pois_clean_adv_id = pois_image_id
        pois_clean_adv_id_list.append(pois_clean_adv_id)

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
        pois_clean_adv_id_list = pois_images_ids
    # ----------

    clean_rank_sims = torch.argsort(eval_clean_sims_caption2i_itm, descending=True)

    if true_image_id in clean_rank_sims:
        clean_rank = (clean_rank_sims == true_image_id).nonzero(as_tuple=True)[0].item() + 1
    else:
        clean_rank = None
    rank_itm_results["clean_rank_itm"] = clean_rank

    clean_not_ir_list_itm = []
    for i in range(1, top_k+1):
        top_n_clean_indices = clean_rank_sims[:i]
        clean_ir = true_image_id not in top_n_clean_indices
        clean_not_ir_list_itm.append(clean_ir)
    ir_1_top_k_itm_results["clean_ir_list_itm"] = clean_not_ir_list_itm
    
    clean_acc_ir_list_itm = []
    for i in range(1, top_k+1):
        top_n_clean_acc_indices = clean_rank_sims[:i]
        clean_acc_ir = true_image_id in top_n_clean_acc_indices
        clean_acc_ir_list_itm.append(clean_acc_ir)
    ir_1_top_k_itm_results["clean_acc_ir_list_itm"] = clean_acc_ir_list_itm

    clean_fault_ir_list_itm = []
    for i in range(1, top_k + 1):
        top_n_clean_fault_indices = clean_rank_sims[:i]
        pois_in_top = sum(1 for pid in pois_clean_adv_id_list if pid in top_n_clean_fault_indices)
        frac = pois_in_top / i
        clean_fault_ir_list_itm.append(frac)
    ir_1_top_k_itm_results["clean_fault_ir_list_itm"] = clean_fault_ir_list_itm
    # ----------

    # ----------
    # for adv eval
    pois_adv_id_list = []
    adv_vit_feats = clean_image_features_dict["clean_vit_feats"].clone()
    adv_images_embeds = clean_image_features_dict["clean_image_embeds"].clone()
    for adv_poisoning_image_dict in adv_poisoning_image_dict_sample_list:
        adv_image_vit_feat = adv_poisoning_image_dict["vit_feat"]
        adv_image_embed = adv_poisoning_image_dict["image_embed"]

        adv_vit_feats = torch.cat([adv_vit_feats, adv_image_vit_feat.to(adv_vit_feats.device).unsqueeze(0)], dim=0)
        
        adv_images_embeds = torch.cat([adv_images_embeds, adv_image_embed.to(adv_images_embeds.device).unsqueeze(0)], dim=0)

        assert len(adv_vit_feats) == len(adv_images_embeds)
        pois_adv_id = len(adv_vit_feats) - 1
        pois_adv_id_list.append(pois_adv_id)

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

    if true_image_id in adv_rank_sims:
        adv_rank = (adv_rank_sims == true_image_id).nonzero(as_tuple=True)[0].item() + 1
    else:
        adv_rank = None
    rank_itm_results["adv_rank_itm"] = adv_rank

    adv_not_ir_list_itm = []
    for i in range(1, top_k+1):
        top_n_adv_indices = adv_rank_sims[:i]
        adv_ir = true_image_id not in top_n_adv_indices
        adv_not_ir_list_itm.append(adv_ir)
    ir_1_top_k_itm_results["adv_ir_list_itm"] = adv_not_ir_list_itm
    
    adv_acc_ir_list_itm = []
    for i in range(1, top_k+1):
        top_n_adv_acc_indices = adv_rank_sims[:i]
        adv_acc_ir = true_image_id in top_n_adv_acc_indices
        adv_acc_ir_list_itm.append(adv_acc_ir)
    ir_1_top_k_itm_results["adv_acc_ir_list_itm"] = adv_acc_ir_list_itm

    adv_fault_ir_list_itm = []
    for i in range(1, top_k + 1):
        top_n_adv_fault_indices = adv_rank_sims[:i]
        pois_in_top = sum(1 for pid in pois_adv_id_list if pid in top_n_adv_fault_indices)
        frac = pois_in_top / i
        adv_fault_ir_list_itm.append(frac)
    ir_1_top_k_itm_results["adv_fault_ir_list_itm"] = adv_fault_ir_list_itm

    return ir_1_top_k_itm_results, rank_itm_results