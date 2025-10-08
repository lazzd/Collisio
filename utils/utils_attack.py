import os
import time

import json

import random

from tqdm import tqdm

import statistics

import torch
import torch.nn.functional as F

from torchvision import transforms

from utils.utils import compress_images_jpeg

from utils.utils_path import get_model_name, get_dataset_name, verify_and_create_path, get_subset_size_str, get_config_attack_path_str, get_date, get_uuid

from utils.utils_statistic import AttackManager, safe_variance

from utils.utils_common_attack import IMAGE_ATTACK_DICT, TEXT_SIMILARITY_DICT, SIMAttacker, retrieval_eval, retrieval_eval_itm, add_in_ir_1_top_k_results_dict_all, compute_means_for_each_key, save_configurations_to_json


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

    jpg_quality = args.jpg_quality

    if jpg_quality is not None:
        path_results_adv = os.path.join(path_save_results, f"jpg_quality:{jpg_quality}", f"model:{model_name}", "valuations", "adversarial", f"dataset:{dataset_name}", f"seed:{args.seed}", f"{subset_size_val}", f"(config::{config_attack_upperpath_str})", f"{config_attack_exp_name_str}.{date}_{uuid}")
    else:
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

        if jpg_quality is not None:
            images_adv = compress_images_jpeg(images_adv, jpg_quality)

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
                if jpg_quality is not None:
                    pois_image = compress_images_jpeg(pois_image, jpg_quality)

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
                "texts_arg": texts_arg,
                "ret_captions": []
            }

            ret_captions_centroid_dist_list = []
            ret_captions_texts_arg_cosine_sim_centroid_list = []
            ret_captions_centroid_cosine_sim_list = []

            ir_1_top_k_results_dict_all = {}
            rank_results_dict_all = {}
            if config["use_itm"] and model_name == "BLIP-2":
                ir_1_top_k_itm_results_dict_all = {}
                rank_itm_results_dict_all = {}

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

                ir_1_top_k_results_dict, rank_results_dict = retrieval_eval(clean_sims_dict, clean_poisoning_image_dict, adv_poisoning_image_dict, ret_caption_id, ret_idx, true_image_id, pois_image_internal_id, top_k=top_k)
                add_in_ir_1_top_k_results_dict_all(ir_1_top_k_results_dict, ir_1_top_k_results_dict_all)
                add_in_ir_1_top_k_results_dict_all(rank_results_dict, rank_results_dict_all)

                if config["use_itm"] and model_name == "BLIP-2":
                    ir_1_top_k_itm_results_dict, rank_itm_results_dict = retrieval_eval_itm(model, clean_sims_dict, clean_poisoning_image_dict, adv_poisoning_image_dict, ret_caption_id, true_image_id, pois_image_internal_id, device, k_test=128, batch_size=128, top_k=top_k)
                    add_in_ir_1_top_k_results_dict_all(ir_1_top_k_itm_results_dict, ir_1_top_k_itm_results_dict_all)
                    add_in_ir_1_top_k_results_dict_all(rank_itm_results_dict, rank_itm_results_dict_all)
            
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
                    rank_results_dict_all=rank_results_dict_all,
                    ir_1_top_k_itm_results_dict=ir_1_top_k_itm_results_dict_mean,
                    ir_1_top_k_itm_results_dict_all=ir_1_top_k_itm_results_dict_all,
                    rank_itm_results_dict_all=rank_itm_results_dict_all)
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
                    ir_1_top_k_results_dict_all=ir_1_top_k_results_dict_all,
                    rank_results_dict_all=rank_results_dict_all)

            path_adv_image = os.path.join(path_results_adv_img, adv_image_name)
            adv_image = to_pil(image_adv.squeeze(0))
            adv_image.save(path_adv_image)

            # ----------

            try:
                base_img = pois_image.squeeze(0).detach().cpu()
                adv_img  = image_adv.squeeze(0).detach().cpu()

                amplify_amplified = 10.0
                amplify_difference = 10.0
                delta = adv_img - base_img

                # grayscale
                pert_mag = delta.abs().amax(dim=0)
                pert_mag_norm = pert_mag / (pert_mag.max() + 1e-8)
                adv_image_name_pert = adv_image_name.replace('.jpg', '_perturbation.png')
                path_adv_image_pert = os.path.join(path_results_adv_img, adv_image_name_pert)
                to_pil(pert_mag_norm).save(path_adv_image_pert)

                # amplified
                amplified = (base_img + amplify_amplified * delta).clamp(0.0, 1.0)
                adv_image_name_ampl = adv_image_name.replace('.jpg', f'_amplified_x{amplify_amplified}.png')
                path_adv_image_ampl = os.path.join(path_results_adv_img, adv_image_name_ampl)
                to_pil(amplified).save(path_adv_image_ampl)

                # difference
                diff_abs = (amplify_difference * (adv_img - base_img).abs()).clamp(0.0, 1.0)
                adv_image_name_diff_abs = adv_image_name.replace('.jpg', f'_diff_abs_x{amplify_difference}.png')
                path_adv_image_diff_abs = os.path.join(path_results_adv_img, adv_image_name_diff_abs)
                to_pil(diff_abs).save(path_adv_image_diff_abs)

                # original
                adv_image_name_orig = adv_image_name.replace('.jpg', '_original.png')
                path_adv_image_orig = os.path.join(path_results_adv_img, adv_image_name_orig)
                to_pil(base_img).save(path_adv_image_orig)

                # adv
                adv_image_name_adv = adv_image_name.replace('.jpg', '_adv.png')
                path_adv_image_adv = os.path.join(path_results_adv_img, adv_image_name_adv)
                to_pil(adv_img).save(path_adv_image_adv)
            except Exception as e:
                print(f"Failed to save perturbation visualizations for {adv_image_name}: {e}")
    
            # ----------

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