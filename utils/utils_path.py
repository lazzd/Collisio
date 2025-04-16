import os

from datetime import datetime
import uuid


def verify_and_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

# -----

def get_model_name(args):
    model_name = args.source_model
    model_name = model_name.replace("/", "-")

    return model_name

def get_dataset_name_clean_files(args, config, external_imgs=False):
    dataset_name = config["config_retrieval"]["dataset_name"]
    if external_imgs:
        if args.retrieval_mode_target_img == 'all':
            dataset_name += "-retrieval_mode_img:all"
        else:
            dataset_name += "-retrieval_mode_img:split"
    else:
        if args.retrieval_mode_img == 'all':
            dataset_name += "-retrieval_mode_img:all"
        else:
            dataset_name += "-retrieval_mode_img:split"
    
    return dataset_name

def get_dataset_name(args, config):
    dataset_name = config["config_retrieval"]["dataset_name"]
    if args.retrieval_mode_img == 'all':
        dataset_name += "-retrieval_mode_img:all"
    else:
        dataset_name += "-retrieval_mode_img:split"
    if args.retrieval_mode_target_img == 'all':
        dataset_name += "-retrieval_mode_target_img:all"
    else:
        dataset_name += "-retrieval_mode_target_img:split"
    
    return dataset_name

def get_subset_size_str(args):
    if args.subset_size == float('inf'):
        subset_size_val = "total"
    else:
        subset_size_val = f"subset_size:{args.subset_size}"
    
    return subset_size_val

def get_config_attack_path_str(args):
    config_attack_path = args.config_attack
    if not config_attack_path.startswith(f'.{os.path.sep}'):
        config_attack_path = f'.{os.path.sep}' + config_attack_path

    config_attack_path_str = config_attack_path.split(f"{os.path.sep}configs{os.path.sep}")[-1]

    config_attack_exp_name_str = config_attack_path_str.split(os.path.sep)[-1]
    config_attack_exp_name_str = config_attack_exp_name_str.replace(".yaml", "")
    config_attack_exp_name_str = config_attack_exp_name_str.replace(os.path.sep, ".")

    config_attack_upperpath_str = os.path.join(*config_attack_path_str.split(os.path.sep)[:-1])
    config_attack_upperpath_str = config_attack_upperpath_str.replace(os.path.sep, ".")

    return config_attack_upperpath_str, config_attack_exp_name_str

# -----

def exist_clean_files(path_exp, args, config, external_imgs=False):
    model_name = get_model_name(args)
    dataset_name = get_dataset_name_clean_files(args, config, external_imgs=external_imgs)

    path_clean_files = os.path.join(path_exp, f"model:{model_name}", "clean_files", f"dataset:{dataset_name}")

    return os.path.exists(path_clean_files)

# ----

def get_date():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def get_uuid():
    return uuid.uuid4().hex[:8]