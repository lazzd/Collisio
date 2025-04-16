import os

import torch

from tqdm import tqdm

from torch.utils.data import DataLoader
from dataset import text_dataset, img_dataset

from utils.utils_path import get_model_name, get_dataset_name, get_dataset_name_clean_files, verify_and_create_path, get_subset_size_str, get_date, get_uuid
from utils.utils_blip_2 import compute_text_embeds, compute_image_embeds, compute_sim_matrix


def clean_eval(model, m_tokenizer, test_transform, img_preprocess, device, args, config, path_save_results, external_imgs=False):
    quant_source_model = getattr(args, 'quant_source_model', None)
    if quant_source_model is None:
        model = model.to(device)
    
    model.eval()

    # ---------------------
    model_name = get_model_name(args)
    dataset_name = get_dataset_name_clean_files(args, config, external_imgs=external_imgs)

    config_retrieval = config["config_retrieval"]

    path_clean_eval_model_dataset = os.path.join(path_save_results, f"model:{model_name}", "clean_files", f"dataset:{dataset_name}")
    verify_and_create_path(path_clean_eval_model_dataset)
    # ---------------------

    print("Clean evaluation text")
    test_text_dataset = text_dataset(config_retrieval['test_file'])
    test_text_data_loader = DataLoader(test_text_dataset, batch_size=args.batch_size,
                             num_workers=4, collate_fn=test_text_dataset.collate_fn)
    
    if model_name == 'BLIP-2':
        text_embeds, text_ids, text_atts = compute_text_embeds(model, m_tokenizer, test_text_data_loader, device)

        torch.cuda.empty_cache()

        clean_text_features_dict = {
            "text_embeds": text_embeds,
            "text_ids": text_ids,
            "text_atts": text_atts
        }
        path_clean_text_feats = os.path.join(path_clean_eval_model_dataset, "clean_text_features_dict.pt")
        torch.save(clean_text_features_dict, path_clean_text_feats)
    else:
        num_text = len(test_text_data_loader.dataset.text)
        clean_text_feats = torch.zeros(num_text, model.visual.output_dim, device=device)

        for batch_idx, (captions_group, captions_ids) in enumerate(tqdm(test_text_data_loader)):
            with torch.no_grad():
                text_input = m_tokenizer(captions_group).to(model.logit_scale.device)
                text_features = model.encode_text(text_input)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                clean_text_feats[captions_ids] = text_features.float().detach()
        
        del test_text_data_loader
        del test_text_dataset

        path_clean_text_feats = os.path.join(path_clean_eval_model_dataset, "clean_text_feats.pt")
        torch.save(clean_text_feats, path_clean_text_feats)
    
    print("Clean evaluation images")
    if (args.retrieval_mode_img == 'all') or (external_imgs and args.retrieval_mode_target_img == 'all'):
        test_img_dataset = img_dataset(config_retrieval['img_list_file'], test_transform, config_retrieval['image_root'])
    else:
        test_img_dataset = img_dataset(config_retrieval['test_file'], test_transform, config_retrieval['image_root'])
    test_img_data_loader = DataLoader(test_img_dataset, batch_size=args.batch_size,
                             num_workers=4, collate_fn=test_img_dataset.collate_fn)
    
    if model_name == 'BLIP-2':
        clean_image_embeds, clean_vit_feats = compute_image_embeds(model, img_preprocess, test_img_data_loader, device)

        torch.cuda.empty_cache()

        clean_image_features_dict = {
            "clean_image_embeds": clean_image_embeds,
            "clean_vit_feats": clean_vit_feats
        }
        path_clean_image_feats = os.path.join(path_clean_eval_model_dataset, "clean_image_features_dict.pt")
        torch.save(clean_image_features_dict, path_clean_image_feats)
    else:
        num_image = len(test_img_data_loader.dataset.image)
        clean_image_feats = torch.zeros(num_image, model.visual.output_dim, device=device)

        for batch_idx, (images, images_ids) in enumerate(tqdm(test_img_data_loader)):
            with torch.no_grad():
                images_norm = img_preprocess(images).to(device)
                image_features = model.encode_image(images_norm)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                clean_image_feats[images_ids] = image_features.float().detach()
        
        del test_img_data_loader
        del test_img_dataset

        path_clean_image_feats = os.path.join(path_clean_eval_model_dataset, "clean_image_feats.pt")
        torch.save(clean_image_feats, path_clean_image_feats)
    
    if model_name == 'BLIP-2':
        cos_sim_matrix_t2i, score_matrix_t2i = compute_sim_matrix(clean_image_embeds, text_embeds, clean_vit_feats, text_embeds, text_ids, text_atts, 128, model, device, batch_size=128)

        path_clean_sims_t2i = os.path.join(path_clean_eval_model_dataset, "clean_sims_t2i.pt")
        torch.save(cos_sim_matrix_t2i, path_clean_sims_t2i)

        path_clean_sims_t2i_itm = os.path.join(path_clean_eval_model_dataset, "clean_sims_t2i_itm.pt")
        torch.save(score_matrix_t2i, path_clean_sims_t2i_itm)
    else:
        clean_sims_matrix = clean_text_feats @ clean_image_feats.t()

        path_clean_sims = os.path.join(path_clean_eval_model_dataset, "clean_sims_t2i.pt")
        torch.save(clean_sims_matrix, path_clean_sims)

# -----

def load_clean_sims_t2i(args, config, path_results, external_imgs=False):
    model_name = get_model_name(args)
    dataset_name = get_dataset_name_clean_files(args, config, external_imgs=external_imgs)

    path_clean_eval_model_dataset = os.path.join(path_results, f"model:{model_name}", "clean_files", f"dataset:{dataset_name}")

    sims_dict = {}
    
    path_clean_sims = os.path.join(path_clean_eval_model_dataset, "clean_sims_t2i.pt")
    clean_sims_t2i = torch.load(path_clean_sims)
    sims_dict["clean_sims_t2i"] = clean_sims_t2i

    if model_name == 'BLIP-2':
        path_clean_image_feats = os.path.join(path_clean_eval_model_dataset, "clean_image_features_dict.pt")
        clean_image_features_dict = torch.load(path_clean_image_feats)
        sims_dict["clean_image_features_dict"] = clean_image_features_dict

        path_clean_text_feats = os.path.join(path_clean_eval_model_dataset, "clean_text_features_dict.pt")
        clean_text_features_dict = torch.load(path_clean_text_feats)
        sims_dict["clean_text_features_dict"] = clean_text_features_dict

        path_clean_sims_itm = os.path.join(path_clean_eval_model_dataset, "clean_sims_t2i_itm.pt")
        clean_sims_t2i_itm = torch.load(path_clean_sims_itm)
        sims_dict["clean_sims_t2i_itm"] = clean_sims_t2i_itm

    return sims_dict