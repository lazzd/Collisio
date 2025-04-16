import torch

from torchvision import transforms
from PIL import Image

from transformers import BertForMaskedLM

from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

from models.blip2_model.blip2 import load_blip2, extend_blip2model, ExtendedMethodsBlip2

from models import clip


def load_model(model_name, model_ckpt, text_encoder, config, device, quant_source_model=None):
    tokenizer = BertTokenizer.from_pretrained(text_encoder)
    ref_model = BertForMaskedLM.from_pretrained(text_encoder)    
    if model_name in ['ALBEF', 'TCL']:
        model = ALBEF(config=config, text_encoder=text_encoder, tokenizer=tokenizer)
        checkpoint = torch.load(model_ckpt, map_location='cpu')
    ### load checkpoint
    elif model_name in ["BLIP-2"]:
        model, vis_processors, txt_processors, blip2_tokenizer = load_blip2(device=device)
        extend_blip2model(model, ExtendedMethodsBlip2, blip2_tokenizer, vis_processors=vis_processors, txt_processors=txt_processors, tokenizer=tokenizer, device=device)
        return model, ref_model, tokenizer
    else:
        # load directly CLIP
        model, preprocess = clip.load(model_name, device=device)
        model.set_tokenizer(tokenizer)
        return model, ref_model, tokenizer
    
    try:
        state_dict = checkpoint['model']
    except:
        state_dict = checkpoint

    if model_name == 'TCL':
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 

    for key in list(state_dict.keys()):
        if 'bert' in key:
            encoder_key = key.replace('bert.', '')
            state_dict[encoder_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    
    return model, ref_model, tokenizer

# -----

def load_transformation(model_name, model):
    if model_name == 'BLIP-2':
        n_px = 364

        def blip2_transform(n_px):

            transform = transforms.Compose([
                transforms.Resize((n_px, n_px), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor()])
            
            def bl_transform(image):
                tr_image = transform(image)
                return tr_image
            return bl_transform
        
        s_test_transform = blip2_transform(n_px)
    else:
        n_px = model.visual.input_resolution
        s_test_transform = transforms.Compose([
            transforms.Resize(n_px, interpolation=Image.BICUBIC),
            transforms.CenterCrop(n_px),
            transforms.ToTensor(),       
        ])
    
    return s_test_transform

# -----

def get_m_tokenizer(model_name, model):
    if model_name == 'BLIP-2':
        m_tokenizer = model.get_tokenizer()
    else:
        def get_clip_tokenizer():
            def clip_tokenizer(text):
                return clip.tokenize(text, 77, True)
            return clip_tokenizer

        m_tokenizer = get_clip_tokenizer()
    
    return m_tokenizer