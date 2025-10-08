import types

import torch
import torch.nn as nn

from lavis.models import load_model_and_preprocess

import torch.nn.functional as F


class VisualConfig:
    def __init__(self, output_dim):
        self.output_dim = output_dim

class DeviceConfig:
    def __init__(self, device):
        self.device = device

class ExtendedMethodsBlip2:
    def __init__(self, blip2_tokenizer, vis_processors, txt_processors, tokenizer, output_dim=768, device="cpu"):
        self.blip2_tokenizer = blip2_tokenizer
        self.vis_processors = vis_processors
        self.txt_processors = txt_processors["eval"]

        self.tokenizer = tokenizer
        self.visual = VisualConfig(output_dim)
        self.logit_scale = DeviceConfig(device)

    def inference_image(self, image):
        image_feat, vit_feat = self.forward_image(image)
        image_feat_norm = F.normalize(self.vision_proj(image_feat), dim=-1)
        return {'image_embed': image_feat,
            'image_feat': image_feat_norm,
            'vit_feat': vit_feat}

    def inference_text(self, text_input):
        text = []
        for input_ids in text_input.input_ids:
            t = self.tokenizer.decode(input_ids).replace('[PAD]', '').replace('[CLS]', '').replace('[SEP]', '').strip()
            text.append(t)
        
        text_input = self.blip2_tokenizer(text, padding="max_length", truncation=True, max_length=35, return_tensors="pt").to(self.device)
        text_feat = self.forward_text(text_input)
        text_feat_norm = F.normalize(self.text_proj(text_feat), dim=-1)
        return {'text_embed': text_feat,
                'text_feat': text_feat_norm}
    
    def encode_text(self, text_ids):
        text_feat = self.forward_text(text_ids)
        return text_feat

    def encode_image(self, image):
        image_feat, vit_feat = self.forward_image(image)
        return image_feat
    
    def inference(self, image, text):
        text_input = self.blip2_tokenizer(text, padding="max_length", truncation=True, max_length=35, return_tensors="pt").to(self.device)
        text_feat = self.forward_text(text_input)
        text_feat_norm = F.normalize(self.text_proj(text_feat), dim=-1)

        image_feat, vit_feat = self.forward_image(image)
        image_feat_norm = F.normalize(self.vision_proj(image_feat), dim=-1)
        return {'text_feat': text_feat_norm, 'image_feat': image_feat_norm, "vit_feat": vit_feat}
    
    def get_tokenizer(self):
        def tokenizer(texts):
            return self.blip2_tokenizer(texts, padding="max_length", truncation=True, max_length=35, return_tensors="pt")
        return tokenizer

def extend_blip2model(instance, methods_class, *args, **kwargs):
    output_dim=256
    methods_instance = methods_class(output_dim=output_dim, *args, **kwargs)

    # add fields
    for attr in vars(methods_instance):
        setattr(instance, attr, getattr(methods_instance, attr))

    # add methods
    for attr in dir(methods_class):
        if callable(getattr(methods_class, attr)) and not attr.startswith("__"):
            method = getattr(methods_class, attr)
            setattr(instance, attr, types.MethodType(method, instance))

def load_blip2(device):
    model, vis_processors, txt_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device=device, is_eval=True)
    blip2_tokenizer = model.tokenizer

    return model, vis_processors, txt_processors, blip2_tokenizer