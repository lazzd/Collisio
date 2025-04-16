import statistics


def safe_variance(data_list):
    return statistics.variance(data_list) if len(data_list) > 1 else 0

class AccuracyManager:
    def __init__(self, args, config_retrieval, top_k=10, use_itm=False):
        self.results = {
            "args": vars(args),
            "config_retrieval": config_retrieval,
            "accuracy_list": [],
            "time": 0,
            "top_k": top_k,
            "accuracy": {}
        }
        self.top_k = top_k
        self.use_itm = use_itm

        self.accuracy_ir_lists = [[] for _ in range(self.top_k)]
        if self.use_itm:
            self.accuracy_ir_lists_itm = [[] for _ in range(self.top_k)]

    def add_accuracy(self, caption, caption_id, true_image_id, accuracy_ir_1_top_k, accuracy_ir_1_top_k_itm=None):
        accuracy_obj = {
            "caption": caption,
            "caption_id": caption_id,
            "true_image_id": true_image_id,
            "accuracy_rate": {
                f"accuracy_ir_1_{self.top_k}": accuracy_ir_1_top_k
            }
        }

        if self.use_itm and accuracy_ir_1_top_k_itm is not None:
            accuracy_obj["accuracy_rate"][f"accuracy_ir_1_{self.top_k}_itm"] = accuracy_ir_1_top_k_itm

        for i_ir, acc_ir_val in enumerate(accuracy_ir_1_top_k):
            self.accuracy_ir_lists[i_ir].append(acc_ir_val)

        if self.use_itm and accuracy_ir_1_top_k_itm is not None:
            for i_ir, acc_ir_val_itm in enumerate(accuracy_ir_1_top_k_itm):
                self.accuracy_ir_lists_itm[i_ir].append(acc_ir_val_itm)

        self.results["accuracy_list"].append(accuracy_obj)

    def set_time(self, time_elapsed):
        self.results["time"] = time_elapsed

    def compute_averages(self):
        accs_ir_1_top_k = [sum(acc_ir_vals) / len(acc_ir_vals) for acc_ir_vals in self.accuracy_ir_lists]
        self.results["accuracy"][f"accs_ir_1_{self.top_k}"] = accs_ir_1_top_k

        if self.use_itm:
            accs_ir_1_top_k_itm = [sum(acc_ir_vals) / len(acc_ir_vals) for acc_ir_vals in self.accuracy_ir_lists_itm]
            self.results["accuracy"][f"accs_ir_1_{self.top_k}_itm"] = accs_ir_1_top_k_itm

    def get_results(self):
        return self.results

# -----

class AttackManager:
    def __init__(self, args, config_retrieval, config_attack, top_k=10, use_itm=False):
        self.results = {
            "args": vars(args),
            "config_retrieval": config_retrieval,
            "config_attack": config_attack,
            "adv_list": [],
            "time": 0,
            "asr": {}
        }
        self.top_k = top_k
        self.use_itm = use_itm

        self.adv_ir_lists = [[] for _ in range(self.top_k)]
        self.adv_acc_ir_lists = [[] for _ in range(self.top_k)]
        self.clean_ir_lists = [[] for _ in range(self.top_k)]
        self.clean_acc_ir_lists = [[] for _ in range(self.top_k)]

        if self.use_itm:
            self.adv_ir_lists_itm = [[] for _ in range(self.top_k)]
            self.adv_acc_ir_lists_itm = [[] for _ in range(self.top_k)]
            self.clean_ir_lists_itm = [[] for _ in range(self.top_k)]
            self.clean_acc_ir_lists_itm = [[] for _ in range(self.top_k)]

    def add_attack_result(self, caption, caption_id, ret_captions, ret_captions_ids, true_image_id, adv_image_id, adv_image_id_internal, adv_image_name, text_init, texts_arg, adv_sims, ir_1_top_k_results_dict, ir_1_top_k_results_dict_all, ir_1_top_k_itm_results_dict=None, ir_1_top_k_itm_results_dict_all=None):

        adv_ir_1_top_k = ir_1_top_k_results_dict["adv_ir_list"]
        clean_ir_1_top_k = ir_1_top_k_results_dict["clean_ir_list"]
        adv_acc_ir_1_top_k = ir_1_top_k_results_dict["adv_acc_ir_list"]
        clean_acc_ir_1_top_k = ir_1_top_k_results_dict["clean_acc_ir_list"]
        
        if self.use_itm and ir_1_top_k_itm_results_dict is not None:
            adv_ir_1_top_k_itm = ir_1_top_k_itm_results_dict["adv_ir_list_itm"]
            clean_ir_1_top_k_itm = ir_1_top_k_itm_results_dict["clean_ir_list_itm"]
            adv_acc_ir_1_top_k_itm = ir_1_top_k_itm_results_dict["adv_acc_ir_list_itm"]
            clean_acc_ir_1_top_k_itm = ir_1_top_k_itm_results_dict["clean_acc_ir_list_itm"]
        
        attack_obj = {
            "caption": caption,
            "ret_captions": ret_captions,
            "caption_id": caption_id,
            "ret_captions_ids": ret_captions_ids,
            "true_image_id": true_image_id,
            "adv_image_id": adv_image_id,
            "adv_image_id_internal": adv_image_id_internal,
            "adv_image_name": adv_image_name,
            "text_init": text_init,
            "texts_arg": texts_arg,
            "adv_sims": adv_sims,
            f"all_ir_1_{self.top_k}_results": ir_1_top_k_results_dict_all,
            "attack_rate": {
                f"mean_adv_ir_1_{self.top_k}": adv_ir_1_top_k,
                f"mean_clean_ir_1_{self.top_k}": clean_ir_1_top_k,
                f"mean_adv_acc_ir_1_{self.top_k}": adv_acc_ir_1_top_k,
                f"mean_clean_acc_ir_1_{self.top_k}": clean_acc_ir_1_top_k,
            }
        }

        if self.use_itm and ir_1_top_k_itm_results_dict is not None:
            attack_obj[f"all_ir_1_{self.top_k}_itm_results"] = ir_1_top_k_itm_results_dict_all

            attack_obj["attack_rate"][f"mean_adv_ir_1_{self.top_k}_itm"] = adv_ir_1_top_k_itm
            attack_obj["attack_rate"][f"mean_clean_ir_1_{self.top_k}_itm"] = clean_ir_1_top_k_itm
            attack_obj["attack_rate"][f"mean_adv_acc_ir_1_{self.top_k}_itm"] = adv_acc_ir_1_top_k_itm
            attack_obj["attack_rate"][f"mean_clean_acc_ir_1_{self.top_k}_itm"] = clean_acc_ir_1_top_k_itm

        for i_ir, adv_ir_val in enumerate(adv_ir_1_top_k):
            self.adv_ir_lists[i_ir].append(adv_ir_val)
        for i_ir, adv_acc_ir_val in enumerate(adv_acc_ir_1_top_k):
            self.adv_acc_ir_lists[i_ir].append(adv_acc_ir_val)
        for i_ir, clean_ir_val in enumerate(clean_ir_1_top_k):
            self.clean_ir_lists[i_ir].append(clean_ir_val)
        for i_ir, clean_acc_ir_val in enumerate(clean_acc_ir_1_top_k):
            self.clean_acc_ir_lists[i_ir].append(clean_acc_ir_val)

        if self.use_itm:
            for i_ir, adv_ir_val in enumerate(adv_ir_1_top_k_itm):
                self.adv_ir_lists_itm[i_ir].append(adv_ir_val)
            for i_ir, adv_acc_ir_val in enumerate(adv_acc_ir_1_top_k_itm):
                self.adv_acc_ir_lists_itm[i_ir].append(adv_acc_ir_val)
            for i_ir, clean_ir_val in enumerate(clean_ir_1_top_k_itm):
                self.clean_ir_lists_itm[i_ir].append(clean_ir_val)
            for i_ir, clean_acc_ir_val in enumerate(clean_acc_ir_1_top_k_itm):
                self.clean_acc_ir_lists_itm[i_ir].append(clean_acc_ir_val)

        self.results["adv_list"].append(attack_obj)

    def set_time(self, time_elapsed):
        self.results["time"] = time_elapsed

    def compute_averages(self):
        self.results["asr"] = {
            f"final_adv_ir_1_{self.top_k}": [sum(vals) / len(vals) if vals else 0 for vals in self.adv_ir_lists],
            f"final_clean_ir_1_{self.top_k}": [sum(vals) / len(vals) if vals else 0 for vals in self.clean_ir_lists],
            f"final_adv_acc_ir_1_{self.top_k}": [sum(vals) / len(vals) if vals else 0 for vals in self.adv_acc_ir_lists],
            f"final_clean_acc_ir_1_{self.top_k}": [sum(vals) / len(vals) if vals else 0 for vals in self.clean_acc_ir_lists],
        }

        if self.use_itm:
            self.results["asr"].update({
                f"final_adv_ir_1_{self.top_k}_itm": [sum(vals) / len(vals) if vals else 0 for vals in self.adv_ir_lists_itm],
                f"final_clean_ir_1_{self.top_k}_itm": [sum(vals) / len(vals) if vals else 0 for vals in self.clean_ir_lists_itm],
                f"final_adv_acc_ir_1_{self.top_k}_itm": [sum(vals) / len(vals) if vals else 0 for vals in self.adv_acc_ir_lists_itm],
                f"final_clean_acc_ir_1_{self.top_k}_itm": [sum(vals) / len(vals) if vals else 0 for vals in self.clean_acc_ir_lists_itm],
            })

    def get_results(self):
        return self.results