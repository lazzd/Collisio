import torch

from typing import Union, Optional

from sim_attack.img_attacker.img_attacker_utils import SimImageAttacker
from sim_attack.text_syn.text_copium import TextCopium
from sim_attack.text_caption_llms_aug.text_caption_llms_aug import TextCaptionLlmsAug
from sim_attack.text_pred_llms_aug.text_pred_llms_aug import TextPredLlmsAug

# -----

class SIMAttacker():
    def __init__(self, model, tokenizer, img_attacker: SimImageAttacker, txt_attacker: Union[TextCopium, TextCaptionLlmsAug, TextPredLlmsAug]):
        self.model = model
        self.tokenizer = tokenizer
        self.txt_attacker = txt_attacker
        self.img_attacker = img_attacker
    
    def sim_attack(self, images, texts, num_iters, true_images=None, images_names=None, indexes_captions=None, **kwargs):
        # get argumentation texts
        with torch.no_grad():
            if isinstance(self.txt_attacker, TextCopium):
                texts_args, all = self.txt_attacker.run_txt_copium(self.model, texts)
            else:
                texts_args, all = self.txt_attacker.run_txt_llms_aug(self.model, texts, indexes_captions)

        # to handle the extreme case when a sublist is empty
        adv_indices = [i for i, texts in enumerate(texts_args) if len(texts) > 0]
        
        selected_texts_args = [texts_args[i] for i in adv_indices]

        if len(selected_texts_args) > 0:
            images_mask = torch.zeros(images.shape[0], dtype=torch.bool)
            images_mask[adv_indices] = True
            selected_images = images[images_mask]
            if true_images is not None:
                selected_true_images = true_images[images_mask]

            # -----

            max_num_captions = max([len(sublist) for sublist in selected_texts_args])
            texts_embedes_list = []
            for i, texts_argued in enumerate(selected_texts_args):
                with torch.no_grad():
                    text_input = self.tokenizer(texts_argued, padding='max_length', truncation=True, max_length=self.txt_attacker.max_length, return_tensors='pt')
                    text_embed = self.model.inference_text(text_input)['text_feat']

                remaining_captions = max_num_captions - len(texts_argued)
                if remaining_captions > 0:
                    zeroes_embed = torch.zeros(remaining_captions, text_embed.shape[1], dtype=text_embed.dtype).to(text_embed.device)
                    text_embed = torch.cat((text_embed, zeroes_embed), dim=0)

                texts_embedes_list.append(text_embed)
            
            target_texts_embeds = torch.stack(texts_embedes_list).squeeze(1)

            if len(target_texts_embeds.shape) < 3:
                target_texts_embeds = target_texts_embeds.unsqueeze(1)

            images_adv = self.img_attacker.run_sim_attack(self.model, selected_images, target_texts_embeds, num_iters, selected_true_images)

            # -----

            ret_images_adv = images.clone()
            ret_images_adv[images_mask] = images_adv
        else:
            ret_images_adv = images.clone()
        
        return ret_images_adv, texts_args, texts