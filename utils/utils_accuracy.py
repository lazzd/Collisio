import os
import json
import time

import torch

from tqdm import tqdm

from utils.utils_path import get_model_name, get_dataset_name, get_subset_size_str, verify_and_create_path

from utils.utils_statistic import AccuracyManager


def compute_accuracy(clean_sims_caption2i, true_image_id, top_k=10):
    clean_rank_sims = torch.argsort(clean_sims_caption2i, descending=True)

    if true_image_id in clean_rank_sims:
        rank = (clean_rank_sims == true_image_id).nonzero(as_tuple=True)[0].item() + 1
    else:
        rank = None

    clean_accuracy_ir_list = []
    for i in range(1, top_k+1):
        top_n_clean_indices = clean_rank_sims[:i]
        clean_ir = true_image_id in top_n_clean_indices
        clean_accuracy_ir_list.append(clean_ir)

    return clean_accuracy_ir_list, rank

# -----

def retrieval_accuracy(data_loader, clean_sims_dict, args, config, path_save_results, top_k=10, on_qt=False):
    model_name = get_model_name(args)
    dataset_name = get_dataset_name(args, config)

    config_retrieval = config["config_retrieval"]
    subset_size_val = get_subset_size_str(args)

    jpg_quality = args.jpg_quality

    if jpg_quality is not None:
        path_results = os.path.join(path_save_results, f"jpg_quality:{jpg_quality}", f"model:{model_name}", "valuations", f"clean_accuracy-on_qt:{on_qt}", f"dataset:{dataset_name}", f"seed:{args.seed}", f"{subset_size_val}")
    else:
        path_results = os.path.join(path_save_results, f"model:{model_name}", "valuations", f"clean_accuracy-on_qt:{on_qt}", f"dataset:{dataset_name}", f"seed:{args.seed}", f"{subset_size_val}")
    verify_and_create_path(path_results)

    accuracy_manager = AccuracyManager(
        args, 
        config_retrieval,
        top_k=top_k, 
        use_itm=config.get("use_itm", False)
    )

    start_time = time.time()

    for batch_idx, (captions_group, ret_captions_group, true_images, adv_images, captions_ids, ret_captions_group_ids, true_images_ids, adv_images_ids, adv_images_internal, true_images_names) in enumerate(tqdm(data_loader)):
        print(f'--------------------> batch:{batch_idx}/{len(data_loader)}')
        
        for idx, (caption, caption_id, ret_captions, ret_caption_ids, true_image_id) in enumerate(zip(captions_group, captions_ids, ret_captions_group, ret_captions_group_ids, true_images_ids)):
            if on_qt:
                accuracy_ir_1_top_k, rank = compute_accuracy(clean_sims_dict["clean_sims_t2i"][caption_id], true_image_id, top_k=top_k)
                if config["use_itm"]:
                    accuracy_ir_1_top_k_itm, rank_itm = compute_accuracy(clean_sims_dict["clean_sims_t2i_itm"][caption_id], true_image_id, top_k=top_k)
                    accuracy_manager.add_accuracy(caption, caption_id, true_image_id, accuracy_ir_1_top_k, rank, accuracy_ir_1_top_k_itm, rank_itm)
                else:
                    accuracy_manager.add_accuracy(caption, caption_id, true_image_id, accuracy_ir_1_top_k, rank)
            else:
                for inner_idx, (ret_caption, ret_caption_id) in enumerate(zip(ret_captions, ret_caption_ids)):
                    accuracy_ir_1_top_k, rank = compute_accuracy(clean_sims_dict["clean_sims_t2i"][ret_caption_id], true_image_id, top_k=top_k)
                    if config["use_itm"]:
                        accuracy_ir_1_top_k_itm, rank_itm = compute_accuracy(clean_sims_dict["clean_sims_t2i_itm"][ret_caption_id], true_image_id, top_k=top_k)
                        accuracy_manager.add_accuracy(ret_caption, ret_caption_id, true_image_id, accuracy_ir_1_top_k, rank, accuracy_ir_1_top_k_itm, rank_itm)
                    else:
                        accuracy_manager.add_accuracy(ret_caption, ret_caption_id, true_image_id, accuracy_ir_1_top_k, rank)

    end_time = time.time()
    total_time = end_time - start_time
    accuracy_manager.set_time(total_time)

    accuracy_manager.compute_averages()

    accuracy_results = accuracy_manager.get_results()

    path_adv_results = os.path.join(path_results, "accuracy_results.json")
    with open(path_adv_results, 'w') as json_file:
        json.dump(accuracy_results, json_file, indent=4)
    
    final_indexes_dict = data_loader.dataset.final_indexes_dict
    considered_text_ids_list = []
    for sampled_id, real_txt_id in final_indexes_dict.items():
        considered_text_ids_list.append(real_txt_id)
    
    path_considered_text_ids_list = os.path.join(path_results, "considered_text_ids_list.json")
    with open(path_considered_text_ids_list, 'w') as json_file:
        json.dump(considered_text_ids_list, json_file, indent=4)


def eval_accuracy(data_loader, clean_sims_dict, args, config, path_save_results):
    retrieval_accuracy(data_loader, clean_sims_dict, args, config, path_save_results)