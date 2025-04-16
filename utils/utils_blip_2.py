import torch

import torch.nn.functional as F

from tqdm import tqdm


def compute_text_embeds(model, m_tokenizer, text_dataset, device):
    text_embeds = []
    text_ids = []
    text_atts = []

    for batch_idx, (captions_group, captions_ids) in enumerate(tqdm(text_dataset)):
        with torch.no_grad():
            text_input = m_tokenizer(captions_group).to(device)
            text_feat = model.forward_text(text_input)
            text_embed = F.normalize(model.text_proj(text_feat), dim=-1)
            
            text_embeds.append(text_embed.cpu())
            text_ids.append(text_input.input_ids.cpu())
            text_atts.append(text_input.attention_mask.cpu())

            del text_input, text_feat, text_embed
            torch.cuda.empty_cache()
    
    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    return text_embeds, text_ids, text_atts

# -----

def compute_image_embeds(model, img_preprocess, img_dataset, device):
    vit_feats = []
    image_embeds = []
    
    for batch_idx, (images, images_ids) in enumerate(tqdm(img_dataset)):
        with torch.no_grad():
            for img, img_id in zip(images, images_ids):
                img_norm = img_preprocess(img).unsqueeze(0).to(device)
                image_feat, vit_feat = model.forward_image(img_norm)
                image_embed = F.normalize(model.vision_proj(image_feat), dim=-1)
                
                vit_feats.append(vit_feat.cpu())
                image_embeds.append(image_embed.cpu())

                del img_norm, image_feat, vit_feat, image_embed
                torch.cuda.empty_cache()

    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    return image_embeds, vit_feats

# -----

def compute_sim_matrix(clean_image_feats, clean_text_feats, vit_feats, text_embeds, text_ids, text_atts, k_test, model, device, batch_size):    
    sims_matrix = []

    for image_embed in clean_image_feats:
        sim_q2t = image_embed @ clean_text_feats.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    cos_sim_matrix_t2i = sims_matrix.clone().t()

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(text_embeds), len(clean_image_feats)), -100.0)

    for i, sims in tqdm(enumerate(sims_matrix), total=len(sims_matrix), desc="Processing Text-to-Image Similarity"):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)

        for batch_start in range(0, k_test, batch_size):
            batch_end = min(batch_start + batch_size, k_test)
            topk_idx_batch = topk_idx[batch_start:batch_end]

            image_inputs = vit_feats[topk_idx_batch].to(device)
            text_ids_batch = text_ids[i].repeat(batch_end - batch_start, 1).to(device)
            text_atts_batch = text_atts[i].repeat(batch_end - batch_start, 1).to(device)

            with torch.no_grad():
                score = model.compute_itm(
                    image_inputs=image_inputs,
                    text_ids=text_ids_batch,
                    text_atts=text_atts_batch,
                ).float()

            score_matrix_t2i[i, topk_idx_batch] = score.cpu() + topk_sim[batch_start:batch_end]

            del score, image_inputs, text_ids_batch, text_atts_batch
            torch.cuda.empty_cache()

    return cos_sim_matrix_t2i, score_matrix_t2i