import time

from tqdm import tqdm

import torch

from utils.utils_statistic import AccuracyManager

from utils.utils_accuracy import compute_accuracy
from utils.utils_blip_2 import compute_sim_matrix


# ----------

def retrieval_accuracy_for_attack(data_loader, sims_dict, args, config, top_k=10, clean_val=False, on_qt=False):
    config_retrieval = config["config_retrieval"]

    if clean_val:
        key_sims_t2i = "clean_sims_t2i"
        key_sims_t2i_itm = "clean_sims_t2i_itm"
    else:
        key_sims_t2i = "adv_sims_t2i"
        key_sims_t2i_itm = "adv_sims_t2i_itm"

    accuracy_manager = AccuracyManager(
        args, 
        config_retrieval,
        top_k=top_k, 
        use_itm=config.get("use_itm", False)
    )

    start_time = time.time()

    for batch_idx, (captions_group, ret_captions_group, true_images, adv_images, captions_ids, ret_captions_group_ids, true_images_ids, adv_images_ids, adv_images_internal, true_images_names) in enumerate(tqdm(data_loader)):
        print(f'--------------------> inner batch:{batch_idx}/{len(data_loader)}')

        for idx, (caption, caption_id, ret_captions, ret_caption_ids, true_image_id) in enumerate(zip(captions_group, captions_ids, ret_captions_group, ret_captions_group_ids, true_images_ids)):
            if on_qt:
                accuracy_ir_1_top_k, rank = compute_accuracy(sims_dict[key_sims_t2i][caption_id], true_image_id, top_k=top_k)
                if config["use_itm"]:
                    accuracy_ir_1_top_k_itm, rank_itm = compute_accuracy(sims_dict[key_sims_t2i_itm][caption_id], true_image_id, top_k=top_k)
                    accuracy_manager.add_accuracy(caption, caption_id, true_image_id, accuracy_ir_1_top_k, rank, accuracy_ir_1_top_k_itm, rank_itm)
                else:
                    accuracy_manager.add_accuracy(caption, caption_id, true_image_id, accuracy_ir_1_top_k, rank)
            else:
                for inner_idx, (ret_caption, ret_caption_id) in enumerate(zip(ret_captions, ret_caption_ids)):
                    accuracy_ir_1_top_k, rank = compute_accuracy(sims_dict[key_sims_t2i][ret_caption_id], true_image_id, top_k=top_k)
                    if config["use_itm"]:
                        accuracy_ir_1_top_k_itm, rank_itm = compute_accuracy(sims_dict[key_sims_t2i_itm][ret_caption_id], true_image_id, top_k=top_k)
                        accuracy_manager.add_accuracy(ret_caption, ret_caption_id, true_image_id, accuracy_ir_1_top_k, rank, accuracy_ir_1_top_k_itm, rank_itm)
                    else:
                        accuracy_manager.add_accuracy(ret_caption, ret_caption_id, true_image_id, accuracy_ir_1_top_k, rank)

    end_time = time.time()
    total_time = end_time - start_time
    accuracy_manager.set_time(total_time)
    accuracy_manager.compute_averages()
    accuracy_results = accuracy_manager.get_results()

    return accuracy_results

def retrieval_accuracy_for_attack_choice(data_loader, sims_dict, args, config, idx_to_eval, top_k=10, clean_val=False, on_qt=False):
    config_retrieval = config["config_retrieval"]

    if clean_val:
        key_sims_t2i = "clean_sims_t2i"
        key_sims_t2i_itm = "clean_sims_t2i_itm"
    else:
        key_sims_t2i = "adv_sims_t2i"
        key_sims_t2i_itm = "adv_sims_t2i_itm"

    accuracy_manager = AccuracyManager(
        args, 
        config_retrieval,
        top_k=top_k, 
        use_itm=config.get("use_itm", False)
    )

    start_time = time.time()

    all_idx = 0

    for batch_idx, (captions_group, ret_captions_group, true_images, adv_images, captions_ids, ret_captions_group_ids, true_images_ids, adv_images_ids, adv_images_internal, true_images_names) in enumerate(tqdm(data_loader)):
        print(f'--------------------> inner batch:{batch_idx}/{len(data_loader)}')

        for idx, (caption, caption_id, ret_captions, ret_caption_ids, true_image_id) in enumerate(zip(captions_group, captions_ids, ret_captions_group, ret_captions_group_ids, true_images_ids)):

            if all_idx in idx_to_eval:
                if on_qt:
                    accuracy_ir_1_top_k, rank = compute_accuracy(sims_dict[key_sims_t2i][caption_id], true_image_id, top_k=top_k)
                    if config["use_itm"]:
                        accuracy_ir_1_top_k_itm, rank_itm = compute_accuracy(sims_dict[key_sims_t2i_itm][caption_id], true_image_id, top_k=top_k)
                        accuracy_manager.add_accuracy(caption, caption_id, true_image_id, accuracy_ir_1_top_k, rank, accuracy_ir_1_top_k_itm, rank_itm)
                    else:
                        accuracy_manager.add_accuracy(caption, caption_id, true_image_id, accuracy_ir_1_top_k, rank)
                else:
                    for inner_idx, (ret_caption, ret_caption_id) in enumerate(zip(ret_captions, ret_caption_ids)):
                        accuracy_ir_1_top_k, rank = compute_accuracy(sims_dict[key_sims_t2i][ret_caption_id], true_image_id, top_k=top_k)
                        if config["use_itm"]:
                            accuracy_ir_1_top_k_itm, rank_itm = compute_accuracy(sims_dict[key_sims_t2i_itm][ret_caption_id], true_image_id, top_k=top_k)
                            accuracy_manager.add_accuracy(ret_caption, ret_caption_id, true_image_id, accuracy_ir_1_top_k, rank, accuracy_ir_1_top_k_itm, rank_itm)
                        else:
                            accuracy_manager.add_accuracy(ret_caption, ret_caption_id, true_image_id, accuracy_ir_1_top_k, rank)
            all_idx += 1

    end_time = time.time()
    total_time = end_time - start_time
    accuracy_manager.set_time(total_time)
    accuracy_manager.compute_averages()
    accuracy_results = accuracy_manager.get_results()

    return accuracy_results

def retrieval_accuracy_for_attack_sub(data_loader, sims_dict, args, config, idx_to_pois, top_k=10, clean_val=False, on_qt=False):
    config_retrieval = config["config_retrieval"]

    if clean_val:
        key_sims_t2i = "clean_sims_t2i"
        key_sims_t2i_itm = "clean_sims_t2i_itm"
    else:
        key_sims_t2i = "adv_sims_t2i"
        key_sims_t2i_itm = "adv_sims_t2i_itm"

    accuracy_manager = AccuracyManager(
        args, 
        config_retrieval,
        top_k=top_k, 
        use_itm=config.get("use_itm", False)
    )

    start_time = time.time()

    all_idx = 0

    for batch_idx, (captions_group, ret_captions_group, true_images, adv_images, captions_ids, ret_captions_group_ids, true_images_ids, adv_images_ids, adv_images_internal, true_images_names) in enumerate(tqdm(data_loader)):
        print(f'--------------------> inner batch:{batch_idx}/{len(data_loader)}')

        for idx, (caption, caption_id, ret_captions, ret_caption_ids, true_image_id) in enumerate(zip(captions_group, captions_ids, ret_captions_group, ret_captions_group_ids, true_images_ids)):

            if all_idx not in idx_to_pois:
                if on_qt:
                    accuracy_ir_1_top_k, rank = compute_accuracy(sims_dict[key_sims_t2i][caption_id], true_image_id, top_k=top_k)
                    if config["use_itm"]:
                        accuracy_ir_1_top_k_itm, rank_itm = compute_accuracy(sims_dict[key_sims_t2i_itm][caption_id], true_image_id, top_k=top_k)
                        accuracy_manager.add_accuracy(caption, caption_id, true_image_id, accuracy_ir_1_top_k, rank, accuracy_ir_1_top_k_itm, rank_itm)
                    else:
                        accuracy_manager.add_accuracy(caption, caption_id, true_image_id, accuracy_ir_1_top_k, rank)
                else:
                    for inner_idx, (ret_caption, ret_caption_id) in enumerate(zip(ret_captions, ret_caption_ids)):
                        accuracy_ir_1_top_k, rank = compute_accuracy(sims_dict[key_sims_t2i][ret_caption_id], true_image_id, top_k=top_k)
                        if config["use_itm"]:
                            accuracy_ir_1_top_k_itm, rank_itm = compute_accuracy(sims_dict[key_sims_t2i_itm][ret_caption_id], true_image_id, top_k=top_k)
                            accuracy_manager.add_accuracy(ret_caption, ret_caption_id, true_image_id, accuracy_ir_1_top_k, rank, accuracy_ir_1_top_k_itm, rank_itm)
                        else:
                            accuracy_manager.add_accuracy(ret_caption, ret_caption_id, true_image_id, accuracy_ir_1_top_k, rank)
            all_idx += 1

    end_time = time.time()
    total_time = end_time - start_time
    accuracy_manager.set_time(total_time)
    accuracy_manager.compute_averages()
    accuracy_results = accuracy_manager.get_results()

    return accuracy_results

# ----------

def build_adv_sims(clean_feats_dict, image_adv_feat):
    clean_text_feats = clean_feats_dict["clean_text_feats"]
    clean_image_feats = clean_feats_dict["clean_image_feats"]

    image_adv_feat_to_add = image_adv_feat.clone().unsqueeze(0)
    adv_image_feats = clean_image_feats.clone()
    adv_image_feats = torch.cat([adv_image_feats, image_adv_feat_to_add], dim=0)

    adv_sims_matrix = clean_text_feats @ adv_image_feats.t()

    adv_sims_dict = {
        "adv_sims_t2i": adv_sims_matrix
    }

    return adv_sims_dict

def build_adv_sims_itm(model, clean_feats_dict, adv_poisoning_image_dict):
    device_model = model.device

    adv_image_vit_feat = adv_poisoning_image_dict["vit_feat"]
    adv_image_embed = adv_poisoning_image_dict["image_embed"]

    adv_vit_feats = clean_feats_dict["clean_image_features_dict"]["clean_vit_feats"].clone()
    adv_vit_feats = torch.cat([adv_vit_feats, adv_image_vit_feat.to(adv_vit_feats.device).unsqueeze(0)], dim=0)

    adv_images_embeds = clean_feats_dict["clean_image_features_dict"]["clean_image_embeds"].clone()
    adv_images_embeds = torch.cat([adv_images_embeds, adv_image_embed.to(adv_images_embeds.device).unsqueeze(0)], dim=0)

    clean_text_features_dict = clean_feats_dict["clean_text_features_dict"]

    adv_images_embeds = adv_images_embeds.to(device_model)
    adv_vit_feats = adv_vit_feats.to(device_model)

    clean_text_embeds = clean_text_features_dict["text_embeds"].to(device_model)
    clean_text_ids = clean_text_features_dict["text_ids"].to(device_model)
    clean_text_atts = clean_text_features_dict["text_atts"].to(device_model)

    adv_sims_matrix, adv_sims_matrix_itm = compute_sim_matrix(
        adv_images_embeds,
        clean_text_embeds,
        adv_vit_feats,
        clean_text_embeds,
        clean_text_ids,
        clean_text_atts,
        128,
        model,
        device_model,
        batch_size=128,
        top_sim_k_cpu=True
    )

    return {
        "adv_sims_t2i": adv_sims_matrix,
        "adv_sims_t2i_itm": adv_sims_matrix_itm
    }

# ----------

def build_adv_seq_image_feats(adv_image_feats_dict, image_adv_feat):
    adv_image_feats = adv_image_feats_dict["image_feats"]
    image_adv_feat_to_add = image_adv_feat.clone().unsqueeze(0)
    adv_image_feats = torch.cat([adv_image_feats, image_adv_feat_to_add], dim=0)

    adv_image_feats_dict.update({
        "image_feats": adv_image_feats
    })

    return adv_image_feats_dict

def build_adv_seq_image_feats_itm(adv_image_feats_dict, adv_poisoning_image_dict):
    adv_image_vit_feat = adv_poisoning_image_dict["vit_feat"]
    adv_image_embed = adv_poisoning_image_dict["image_embed"]

    adv_vit_feats = adv_image_feats_dict["vit_feats"].clone()
    adv_vit_feats = torch.cat([adv_vit_feats, adv_image_vit_feat.to(adv_vit_feats.device).unsqueeze(0)], dim=0)

    adv_images_embeds = adv_image_feats_dict["image_embeds"].clone()
    adv_images_embeds = torch.cat([adv_images_embeds, adv_image_embed.to(adv_images_embeds.device).unsqueeze(0)], dim=0)

    adv_image_feats_dict.update({
        "vit_feats": adv_vit_feats,
        "image_embeds": adv_images_embeds
    })

    return adv_image_feats_dict

def compute_adv_sims(clean_feats_dict, adv_image_feats_dict):
    clean_text_feats = clean_feats_dict["clean_text_feats"]

    adv_image_feats = adv_image_feats_dict["image_feats"]

    adv_sims_matrix = clean_text_feats @ adv_image_feats.t()

    adv_sims_dict = {
        "adv_sims_t2i": adv_sims_matrix
    }

    return adv_sims_dict

def compute_adv_sims_itm(model, clean_feats_dict, adv_image_feats_dict):
    device_model = model.device

    clean_text_features_dict = clean_feats_dict["clean_text_features_dict"]

    adv_images_embeds = adv_image_feats_dict["image_embeds"].to(device_model)
    adv_vit_feats = adv_image_feats_dict["vit_feats"].to(device_model)

    clean_text_embeds = clean_text_features_dict["text_embeds"].to(device_model)
    clean_text_ids = clean_text_features_dict["text_ids"].to(device_model)
    clean_text_atts = clean_text_features_dict["text_atts"].to(device_model)

    adv_sims_matrix, adv_sims_matrix_itm = compute_sim_matrix(
        adv_images_embeds,
        clean_text_embeds,
        adv_vit_feats,
        clean_text_embeds,
        clean_text_ids,
        clean_text_atts,
        128,
        model,
        device_model,
        batch_size=128,
        top_sim_k_cpu=True
    )

    return {
        "adv_sims_t2i": adv_sims_matrix,
        "adv_sims_t2i_itm": adv_sims_matrix_itm
    }