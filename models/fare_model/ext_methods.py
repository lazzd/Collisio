import torch.nn.functional as F
from models import clip


class FAREExtMethods:
    def inference_text(self, text_input):
        text = []
        for input_ids in text_input.input_ids:
            t = self.tokenizer.decode(input_ids).replace('[PAD]', '').replace('[CLS]', '').replace('[SEP]', '').strip()
            text.append(t)
        text_input = clip.tokenize(text, 77, True).to(self.logit_scale.device)
        txt_embed = self.encode_text(text_input)
        text_feat = F.normalize(txt_embed, dim=-1)
        return {'text_embed': txt_embed,
                'text_feat': text_feat,}
    
    def inference_image(self, image):
        image_embed = self.encode_image(image)
        image_feat = F.normalize(image_embed, dim=-1)
        return {'image_embed': image_embed,
                'image_feat': image_feat,
                }
    
    def inference(self, image, text):
        text_input = clip.tokenize(text, 77, True).to(self.logit_scale.device)
        image_features = self.encode_image(image)
        text_features = self.encode_text(text_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return {'text_feat': text_features, 'image_feat': image_features}
    
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer