import argparse

import os
import json

import re
import ast

from transformers import pipeline

import torch

import numpy as np

import random

from tqdm import tqdm


# ----------

HF_TOKEN = ""

# ----------

MAX_ATTEMPTS = 3
SAVE_PATH = './caption_mistral_llms_aug'

# ----------


def set_all_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def verify_and_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def add_relative_path(path):
    if not path.startswith(f'.{os.path.sep}'):
        path = f'.{os.path.sep}' + path

def define_message(caption):
    messages = [
        {"role": "user", "content":
            f"""Given the following caption, generate exactly 15 variations that preserve the original meaning while rephrasing the wording. Ensure diversity in structure and style while keeping the core message intact.\nReturn the output as a Python list format, where each variation is a string inside a list.\nExample output format: ```python\ncaptions = ["First variation here", "Second variation here", "Third variation here", ...]```\nHere is the original caption: '{caption}'"""},
    ]

    return messages

def main(args):
    seed = args.seed
    set_all_seed(seed=seed)

    chatbot = pipeline("text-generation", model="mistralai/Mistral-Small-24B-Instruct-2501", torch_dtype=torch.bfloat16, token=HF_TOKEN, device_map="auto")

    # ----------
    ann_filename_path = args.ann_filename_path
    add_relative_path(ann_filename_path)
    sampled_txt_ids_filename_path = args.sampled_txt_ids_filename_path
    add_relative_path(sampled_txt_ids_filename_path)

    ann_dataset_name = ann_filename_path.split(os.path.sep)[-1].split(".json")[0]

    if sampled_txt_ids_filename_path is None:
        save_path = os.path.join(*ann_filename_path.split(os.path.sep)[:-1])
        save_filename = f"complete_mistral_seed:{seed}_{ann_filename_path.split(os.path.sep)[-1]}"
        considered_text_ids_list = None
    else:
        save_path = os.path.join(*sampled_txt_ids_filename_path.split(os.path.sep)[2:-1])
        save_path = os.path.join(SAVE_PATH, save_path)
        verify_and_create_path(save_path)
        save_filename = f"{ann_dataset_name}.seed:{seed}.json"
        with open(sampled_txt_ids_filename_path, 'r') as f:
            considered_text_ids_list = json.load(f)
        
    # ----------
    
    loaded_ann = json.load(open(ann_filename_path, 'r'))

    final_list_output = {}

    idx_txt = 0
    for i, ann in enumerate(tqdm(loaded_ann, desc="Processing annotations")):
        for j, caption in enumerate(ann['caption']):

            if considered_text_ids_list is None or (idx_txt in considered_text_ids_list):
                result_dict = {"true_caption": caption}

                messages = define_message(caption)

                aug_caption = []

                num_try = 1
                max_new_tokens=512

                while len(aug_caption) == 0 and num_try <= MAX_ATTEMPTS:
                    set_all_seed(seed=seed)
                    new_tokens = max_new_tokens*num_try
                    output = chatbot(messages, num_return_sequences=1, max_new_tokens=new_tokens)

                    response = output[0]['generated_text'][1]['content']

                    cleaned_response = re.sub(r"```python|```", "", response).strip()

                    match = re.search(r"captions\s*=\s*(\[[\s\S]*\])", cleaned_response)

                    if match:
                        captions_list_str = match.group(1)
                        aug_caption = ast.literal_eval(captions_list_str)
                    else:
                        print("Error")
                        print(cleaned_response)
                        num_try += 1
                
                if len(aug_caption) == 0:
                    print("Fallback")
                    fallback_captions = re.findall(r'"([^"]+)"(?:,|\])', cleaned_response)
                    if fallback_captions:
                        aug_caption = fallback_captions
                    else:
                        print("Fallback failed")

                result_dict["caption"] = aug_caption

                final_list_output[idx_txt] = result_dict

            idx_txt += 1
    
    path_results = os.path.join(save_path, save_filename)
    with open(path_results, 'w') as json_file:
        json.dump(final_list_output, json_file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_filename_path', default='./data_annotation/flickr30k_test.json')
    parser.add_argument('--sampled_txt_ids_filename_path', default=None)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    main(args)