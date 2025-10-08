import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import logging

import random

from sim_attack.text_syn.synonyms_over import Conj_Synonyms

from sim_attack.text_syn.text_syn_utils import filter_words, chars_to_del, OfflineDict


class TextCopium():
    def __init__(self, ref_net, tokenizer, cls=True, use_thesaurus=False, loss_type="Cosine", use_loss_for_final=False, sentence_context=False, max_length=30, add_original_word=False, add_original_captions=True, num_perturbation=1, topk=10, topsyn=None, num_link=1, num_copium=1, threshold_pred_score=0.3, batch_size=32, use_offline_dict=True, device="cpu"):
        assert loss_type == "Cosine" or loss_type == "L2" or loss_type == "Random"
        # ----------
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_original_caption = add_original_captions
        # ----------
        self.add_original_word = add_original_word
        # ----------
        self.num_perturbation = num_perturbation
        self.num_copium = num_copium
        self.num_link = num_link
        if self.num_link == "Inf":
            self.num_link = float('inf')
        self.threshold_pred_score = threshold_pred_score
        self.topk = topk
        self.topsyn = topsyn
        self.batch_size = batch_size
        self.cls = cls
        # -----
        self.loss_type = loss_type
        self.use_loss_for_final = use_loss_for_final
        # -----
        self.sentence_context = sentence_context
        # -----
        self.use_thesaurus = use_thesaurus
        # -----
        self.device = device

        self.use_offline_dict = use_offline_dict
        if self.use_offline_dict:
            self.offline_dict = OfflineDict()
    
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

    def run_txt_copium(self, net, texts):
        ref_device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(ref_device)

        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, dim=-1)  # seq-len k

        # ----------
        # ----------
        if self.num_copium > 0:
            origin_output = net.inference_text(text_inputs)
            if self.cls:
                origin_embeds = origin_output['text_feat'][:, 0, :].detach()
            else:
                origin_embeds = origin_output['text_feat'].detach()
        # ----------
        # ----------

        final_adverse = []
        final_texts_copium = []

        for i, text in enumerate(texts):
            if self.num_copium > 0:

                all_text_copium = []

                # word importance eval
                important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)
                
                list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

                words, sub_words, keys = self._tokenize(text)

                if self.num_perturbation > 0:
                    list_final_words = [(float('inf'), copy.deepcopy(words))]

                    change = 0
                    for iter_index, top_index in enumerate(list_of_index):
                        if change >= self.num_perturbation:
                            break
                        
                        next_list_final_words = []

                        tgt_word = words[top_index[0]]
                        if tgt_word.lower() in filter_words:
                            continue
                        if keys[top_index[0]][0] > self.max_length - 2:
                            continue

                        if self.use_thesaurus:
                            sanitized_tgt_word = tgt_word
                            for char_to_del in chars_to_del:
                                sanitized_tgt_word = sanitized_tgt_word.replace(char_to_del, '')

                            if self.use_offline_dict:
                                substitutes = self.offline_dict.get_value(sanitized_tgt_word)
                            else:
                                substitutes = None

                            if substitutes is None:
                                synonym = Conj_Synonyms()
                                try:
                                    substitutes = synonym.find_synonyms(sanitized_tgt_word, max_number_of_requests=9999999999, rate_limit_timeout_period=999999999)
                                except Exception as e:
                                    logging.error(f"Error occurred while finding synonyms for '{sanitized_tgt_word}': {str(e)}")
                                    substitutes = None
                                
                                if substitutes is not None:
                                    if self.use_offline_dict:
                                        self.offline_dict.add_entry(sanitized_tgt_word, substitutes)

                            if self.topsyn is not None:
                                substitutes = substitutes[:self.topsyn]
                        else:
                            substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k

                            word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                            substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                                        self.threshold_pred_score)

                        for couple_final_words in list_final_words:
                            final_words = couple_final_words[1]
                            replace_texts = []
                            
                            available_substitutes = []
                            for substitute_ in substitutes:
                                substitute = substitute_

                                if substitute == tgt_word:
                                    continue  # filter out original word
                                if '##' in substitute:
                                    continue  # filter out sub-word
                                
                                if substitute.lower() in filter_words:
                                    continue

                                # ----------
                                if self.sentence_context:
                                    temp_replace = copy.deepcopy(final_words)
                                    temp_replace[top_index[0]] = substitute
                                    available_substitutes.append(substitute)
                                    replace_texts.append(' '.join(temp_replace))
                                # ----------
                                # ----------
                                else:
                                    available_substitutes.append(substitute)
                                    replace_texts.append(substitute)
                            
                            if len(replace_texts) == 0:
                                continue

                            if self.loss_type != "Random":
                                # ---------
                                replace_embeds = self.get_text_embeds(net, replace_texts)

                                if self.sentence_context:
                                    output_feat_ori = origin_embeds[i]

                                else:
                                    output_feat_ori = self.get_text_embeds(net, tgt_word).squeeze(0)
                                
                                loss = self.loss_func(replace_embeds, output_feat_ori)
                                # ----------

                                sorted_loss_indices = torch.argsort(loss, descending=True)
                            else:
                                loss = torch.full((len(replace_texts),), float('inf'))
                                sorted_loss_indices = torch.randperm(loss.size(0))

                            idx_loss_indices = 0
                            idx_split = 0

                            while idx_split < self.num_link and idx_split < len(available_substitutes):
                                changed = False
                                while not changed and idx_loss_indices < sorted_loss_indices.shape[0]:
                                    if available_substitutes[sorted_loss_indices[idx_loss_indices]] != tgt_word:
                                        actual_final_words = copy.deepcopy(final_words)
                                        actual_final_words[top_index[0]] = available_substitutes[sorted_loss_indices[idx_loss_indices]]
                                        
                                        next_list_final_words = next_list_final_words + [(loss[sorted_loss_indices[idx_loss_indices]].item(), actual_final_words)]
                                        all_text_copium.append(' '.join(actual_final_words))

                                        changed = True
                                    
                                    idx_loss_indices += 1
                                
                                idx_split += 1

                            if self.add_original_word:
                                actual_final_words = copy.deepcopy(final_words)
                                actual_final_words[top_index[0]] = tgt_word
                                next_list_final_words = next_list_final_words + [(couple_final_words[0], actual_final_words)]

                                all_text_copium.append(' '.join(actual_final_words))

                        if len(next_list_final_words) != 0:
                            list_final_words = copy.deepcopy(next_list_final_words)
                            change += 1

                    # # ----------
                    final_adverse.append([(couple_final_words[0], ' '.join(couple_final_words[1])) for couple_final_words in list_final_words if ' '.join(couple_final_words[1]) != text])
                    final_texts_copium.append(all_text_copium)
                
                else:
                    final_adverse.append([])
                    final_texts_copium.append(all_text_copium)
            else:
                final_adverse.append([])

        filtered_final_adverse = []
        for sublist, text in zip(final_adverse, texts):
            if len(sublist) < self.num_copium:
                random_filtered_sublist = [couple_list[1] for couple_list in sublist]
            else:
                if self.loss_type != "Random" and self.use_loss_for_final:
                    random_filtered_sublist = [couple_list[1] for couple_list in sorted(sublist, key=lambda x: x[0], reverse=True)[:self.num_copium]]
                else:
                    random_filtered_sublist = random.sample([couple_list[1] for couple_list in sublist], self.num_copium)

            if self.add_original_caption:
                random_filtered_sublist.append(text)
            filtered_final_adverse.append(random_filtered_sublist)

        return filtered_final_adverse, final_texts_copium
    
    def loss_func(self, replace_embeds, original_model_actual_original_feat):
        if self.loss_type == "Cosine":
            loss = original_model_actual_original_feat @ replace_embeds.t()
        else:
            loss = -torch.norm(replace_embeds - original_model_actual_original_feat, p=2, dim=1)
        
        return loss
 
    def _tokenize(self, text):
        words = text.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def _get_masked(self, text):
        words = text.split(' ')
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words

    def get_important_scores(self, text, net, origin_embeds, batch_size, max_length):
        device = origin_embeds.device

        masked_words = self._get_masked(text)
        masked_texts = [' '.join(words) for words in masked_words]

        masked_embeds = []

        for i in range(0, len(masked_texts), batch_size):
            masked_text_input = self.tokenizer(masked_texts[i:i+batch_size], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt').to(device)
            masked_output = net.inference_text(masked_text_input)
            if self.cls:
                masked_embed = masked_output['text_feat'][:, 0, :].detach()
            else:
                masked_embed = masked_output['text_feat'].flatten(1).detach()
            masked_embeds.append(masked_embed)
        
        masked_embeds = torch.cat(masked_embeds, dim=0)
        
        criterion = torch.nn.KLDivLoss(reduction='none')

        import_scores = criterion(masked_embeds.log_softmax(dim=-1), origin_embeds.softmax(dim=-1).repeat(len(masked_texts), 1))

        return import_scores.sum(dim=-1)

def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words

    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    return words

def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    device = mlm_model.device
    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to(device)
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words