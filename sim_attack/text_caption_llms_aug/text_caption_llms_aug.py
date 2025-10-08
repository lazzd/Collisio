import os

import json

import random

import torch

from dataset import pre_caption, MAX_WORDS


FOLDER_AUG_LLMS_PATH = './data_annotation/caption_mistral_llms_aug'
SUFFIX_FILE = '.seed:42.json'

class TextCaptionLlmsAug():
    def __init__(self, test_file_name, dataset_name, seed, subset_size_val, ref_net, tokenizer, cls=True, loss_type="Cosine", max_length=30, add_original_captions=True, num_copium=1, batch_size=32, device="cpu", use_caption_preprocessing=True):
        assert loss_type == "Cosine" or loss_type == "L2" or loss_type == "Random"
        # ----------
        self.test_file_name = test_file_name
        self.dataset_name = dataset_name
        self.seed = seed
        self.subset_size_val = subset_size_val
        # ----------
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_original_caption = add_original_captions
        self.num_copium = num_copium
        self.batch_size = batch_size
        self.cls = cls
        # -----
        self.loss_type = loss_type
        # -----
        self.device = device
        
        # -----
        self.use_caption_preprocessing = use_caption_preprocessing
        # -----

        self.caption_aug_folder_path_name = os.path.join(FOLDER_AUG_LLMS_PATH, f"dataset:{dataset_name}", f"seed:{seed}", f"{subset_size_val}", f"{test_file_name.split('.json')[0]}{SUFFIX_FILE}")

        try:
            with open(self.caption_aug_folder_path_name, "r", encoding="utf-8") as f:
                self.caption_aug_texts_json = json.load(f)
        except FileNotFoundError:
            print(f"Error: Aug texts file '{self.caption_aug_folder_path_name}' not exist.")
            exit()

    def embed_normalize(self, embed, dim=-1):
        return embed / embed.norm(dim=dim, keepdim=True)
    
    def get_text_embeds(self, net, texts):
        with torch.no_grad():
            texts_arg_input = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
            texts_arg_output = net.inference_text(texts_arg_input)['text_feat']

        if self.cls:
            texts_arg_output = texts_arg_output[:, 0, :].detach()
        else:
            texts_arg_output = texts_arg_output.detach()

        return texts_arg_output
    
    def run_txt_llms_aug(self, net, texts, indexes_captions):
        assert indexes_captions is not None

        if self.num_copium > 0 and self.loss_type != "Random":
            output_feats = self.get_text_embeds(net, texts)

        final_adverse = []
        final_texts_copium = []

        for i, (text, idx_caption) in enumerate(zip(texts, indexes_captions)):
            if self.num_copium > 0:

                assert str(idx_caption) in self.caption_aug_texts_json
                aug_dict = self.caption_aug_texts_json[str(idx_caption)]

                aug_arg_texts = aug_dict["caption"]

                if self.use_caption_preprocessing:
                    aug_arg_texts = [pre_caption(aug_arg_text, MAX_WORDS) for aug_arg_text in aug_arg_texts]
                aug_arg_texts = [aug_arg_text for aug_arg_text in aug_arg_texts if aug_arg_text != text]

                if len(aug_arg_texts) > 0:

                    all_text_adverse = aug_arg_texts

                    num_to_select = min(self.num_copium, len(aug_arg_texts))

                    if self.loss_type != "Random":
                        output_feat_ori = output_feats[i]

                        replace_embeds = self.get_text_embeds(net, aug_arg_texts)

                        loss = self.loss_func(replace_embeds, output_feat_ori)
                        sorted_loss_indices = torch.argsort(loss, descending=True)

                        top_indices = sorted_loss_indices[:num_to_select]
                        text_adverse = [aug_arg_texts[idx] for idx in top_indices]
                    else:
                        text_adverse = random.sample(aug_arg_texts, num_to_select) if num_to_select > 0 else []
                else:
                    text_adverse = []
                    all_text_adverse = []
                
            else:
                text_adverse = []
                all_text_adverse = []

            if self.add_original_caption:
                text_adverse.append(text)
                all_text_adverse.append(text)

            final_adverse.append(text_adverse)
            final_texts_copium.append(all_text_adverse)
        
        return final_adverse, final_texts_copium

    def loss_func(self, replace_embeds, original_model_actual_original_feat):
        if self.loss_type == "Cosine":
            loss = original_model_actual_original_feat @ replace_embeds.t()
        else:
            loss = -torch.norm(replace_embeds - original_model_actual_original_feat, p=2, dim=1)
        
        return loss