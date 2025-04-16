import argparse

from ruamel.yaml import YAML

import torch
from torch.utils.data import DataLoader

from torchvision import transforms

import utils.utils as utils

from dataset import paired_caption_adv_img_dataset

from utils.utils_model import load_model, load_transformation, get_m_tokenizer
from utils.utils_path import exist_clean_files
from utils.utils_clean import clean_eval, load_clean_sims_t2i
from utils.utils_accuracy import eval_accuracy


# -----

PATH_EXP = "./experiments"

# -----

DO_CLEAN_EVAL = True

# -----

def main(args, config):
    device = torch.device('cuda')

    utils.set_all_seed(args.seed)

    print("Creating Source Model")
    # load source model trained...
    model, ref_model, tokenizer = load_model(args.source_model, args.source_ckpt, args.source_text_encoder, config["config_retrieval"], device)

    #### Dataset ####
    print("Creating dataset")    
    s_test_transform = load_transformation(args.source_model, model)

    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    m_tokenizer = get_m_tokenizer(args.source_model, model)

    if DO_CLEAN_EVAL:
        if not exist_clean_files(PATH_EXP, args, config, external_imgs=False):
            clean_eval(model, m_tokenizer, s_test_transform, images_normalize, device, args, config, PATH_EXP)

        if not exist_clean_files(PATH_EXP, args, config, external_imgs=True):
            clean_eval(model, m_tokenizer, s_test_transform, images_normalize, device, args, config, PATH_EXP, external_imgs=True)

    # ----------

    clean_sims_dict = load_clean_sims_t2i(args, config, PATH_EXP)
    
    if args.retrieval_mode_img == 'all' or args.retrieval_mode_target_img == 'all':
        imgs_external_list = config_retrieval['img_list_file']
    else:
        imgs_external_list = None

    test_dataset = paired_caption_adv_img_dataset(config_retrieval['test_file'], s_test_transform, config_retrieval['image_root'], retrieval_mode_img=args.retrieval_mode_img, imgs_external_list=imgs_external_list, subset_size=args.subset_size, seed_val=args.seed)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, collate_fn=test_dataset.collate_fn)

    eval_accuracy(test_loader, clean_sims_dict, args, config, PATH_EXP)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_retrieval', default='./configs/Retrieval_flickr_server.yaml')
    parser.add_argument('--subset_size', default=float("inf"), type=int)
    parser.add_argument('--retrieval_mode_img', default='split', type=str, choices=['split', 'all'], help="For retrieved images. 'split': Karpathy split, 'all': all images in eval set.")
    parser.add_argument('--retrieval_mode_target_img', default='split', type=str, choices=['all', 'split'], help="To search best poisoning image. 'split': inside Karpathy split, 'all': in all images in eval set.")
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--source_model', default='ViT-B/16', type=str, choices=['ViT-B/16', 'RN101', 'BLIP-2'])
    parser.add_argument('--source_text_encoder', default='bert-base-uncased', type=str)
    parser.add_argument('--source_ckpt', default='', type=str)

    parser.add_argument('--cls', default=False, type=bool)
    parser.add_argument('--verbose', default=False, type=bool)
    
    args = parser.parse_args()

    yaml = YAML(typ='rt')

    # Load your YAML file
    with open(args.config_retrieval, 'r') as config_file:
        config_retrieval = yaml.load(config_file)
    
    if args.source_model == 'BLIP-2':
        use_itm = True
    else:
        use_itm = False

    config = {
        "config_retrieval": config_retrieval,
        "use_itm": use_itm
    }

    main(args, config)