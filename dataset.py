import os
import re
import json
import random

from PIL import Image

import torch
from torch.utils.data import Dataset

from utils.utils import set_all_seed


MAX_WORDS = 30

def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

class paired_caption_adv_img_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=MAX_WORDS, retrieval_mode_img='split', imgs_external_list=None, subset_size = float("inf"), seed_val=42):
        loaded_ann = json.load(open(ann_file, 'r'))

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        # ----------
        self.seed_val = seed_val
        # ----------

        # ----------
        # for 'external' images
        self.external_image = []
        self.external_image_dict = {}

        if imgs_external_list is not None:
            loaded_external_imgs_list = json.load(open(imgs_external_list, 'r'))
        else:
            loaded_external_imgs_list = loaded_ann
        
        for i, ann in enumerate(loaded_external_imgs_list):
            image_path = ann["image"]
            self.external_image_dict[image_path] = i
            self.external_image.append(image_path)

        # ----------
        # for image
        if retrieval_mode_img!='split':
            self.image = self.external_image
            self.image_dict = self.external_image_dict
        else:
            self.image = []
            self.image_dict = {}
            for i, ann in enumerate(loaded_ann):
                image_path = ann["image"]
                self.image_dict[image_path] = i
                self.image.append(image_path)

        # ----------
        # mapping_inside
        self.adv_img_internal = {}
        
        # ----------
        # for text

        self.text = []
        
        self.img2txt = {}
        self.txt2img = {}

        self.txt2adv_img =  {}
        self.ann = []

        # ----------
        set_all_seed(seed_val)
        # ----------

        txt_id = 0
        for i, ann in enumerate(loaded_ann):
            image_path = ann["image"]
            idx_image = self.image_dict[image_path]
            external_idx_image = self.external_image_dict.get(image_path, None)
            self.img2txt[idx_image] = []

            for j, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, self.max_words))
                self.txt2img[txt_id] = idx_image    
                img_adv_idx = random.choice([u for u in range(len(loaded_external_imgs_list)) if u != external_idx_image])
                
                img_adv_idx_path = self.external_image[img_adv_idx]
                self.adv_img_internal[txt_id] = self.image_dict.get(img_adv_idx_path, None)

                self.txt2adv_img[txt_id] = img_adv_idx

                self.img2txt[idx_image].append(txt_id)
                txt_id += 1
            self.ann.append(ann)

        # ----------
        # for retrieval queries
        self.txt2ret_txts =  {}
        # ----------
        for txt_idx, img_idx in self.txt2img.items():
            all_associated_txt_ids = self.img2txt[img_idx]
            self.txt2ret_txts[txt_idx] = [item for item in all_associated_txt_ids if item != txt_idx]
        
        # ----------
        set_all_seed(seed_val)
        # ----------

        if subset_size >= len(self.txt2img):
            final_idxs = list(range(len(self.txt2img)))
        else:
            final_idxs = random.sample(range(len(self.txt2img)), subset_size)
        
        random.shuffle(final_idxs)

        self.final_indexes_dict = {}
        for i, final_idx in enumerate(final_idxs):
            self.final_indexes_dict[i] = final_idx

    def __len__(self):
        return len(self.final_indexes_dict)

    def __getitem__(self, final_index):
        index = self.final_indexes_dict[final_index]

        text = self.text[index]

        ret_text_idxs = self.txt2ret_txts[index]
        ret_texts = [self.text[ret_text_idx] for ret_text_idx in ret_text_idxs]

        true_image_idx = self.txt2img[index]
        true_image_name = self.image[true_image_idx]
        true_image_path = os.path.join(self.image_root, true_image_name)
        true_image = Image.open(true_image_path).convert('RGB')
        true_image = self.transform(true_image)

        adv_image_idx = self.txt2adv_img[index]
        adv_image_path = os.path.join(self.image_root, self.external_image[adv_image_idx])
        adv_image = Image.open(adv_image_path).convert('RGB')
        adv_image = self.transform(adv_image)
        adv_image_internal = self.adv_img_internal[index]

        return text, ret_texts, true_image, adv_image, index, ret_text_idxs, true_image_idx, adv_image_idx, adv_image_internal, true_image_name

    def collate_fn(self, batch):
        txt_groups, ret_txt_groups, true_imgs, adv_imgs, txt_ids, ret_txt_ids, true_imgs_idx, adv_imgs_idx, adv_imgs_internal, true_image_names = list(zip(*batch))
        true_imgs = torch.stack(true_imgs, 0)
        adv_imgs = torch.stack(adv_imgs, 0)
        return txt_groups, ret_txt_groups, true_imgs, adv_imgs, list(txt_ids), list(ret_txt_ids), list(true_imgs_idx), list(adv_imgs_idx), list(adv_imgs_internal), true_image_names

class img_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root):
        loaded_ann = json.load(open(ann_file, 'r'))

        self.transform = transform
        self.image_root = image_root

        self.image = []

        for i, ann in enumerate(loaded_ann):
            self.image.append(ann['image'])
        
    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image[index])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index

    def collate_fn(self, batch):
        imgs, imgs_idx = list(zip(*batch))        
        imgs = torch.stack(imgs, 0)
        return imgs, list(imgs_idx)

class text_dataset(Dataset):
    def __init__(self, ann_file, max_words=MAX_WORDS):
        loaded_ann = json.load(open(ann_file, 'r'))

        self.max_words = max_words

        self.text = []

        for i, ann in enumerate(loaded_ann):
            for j, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, self.max_words))

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]

        return text, index

    def collate_fn(self, batch):
        txt_groups, txt_ids = list(zip(*batch))
        return txt_groups, list(txt_ids)