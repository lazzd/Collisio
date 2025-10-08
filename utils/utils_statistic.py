import statistics


def safe_variance(data_list):
    return statistics.variance(data_list) if len(data_list) > 1 else 0

def mean_dict(dicts):
    result = {}
    keys = dicts[0].keys()
    
    for k in keys:
        values = [d[k] for d in dicts if d[k] is not None]
        
        if not values:
            result[f"mean_{k}"] = None
            continue

        if isinstance(values[0], list):
            result[f"mean_{k}"] = [statistics.mean(v) for v in zip(*values)]
        else:
            result[f"mean_{k}"] = statistics.mean(values)
    
    return result

def reciprocal_rank(rank, k=None):
    if rank is None:
        return 0.0
    r = int(rank)
    if r <= 0:
        return 0.0
    if k is not None and r > k:
        return 0.0
    return 1.0 / r

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

        self.rank_true_images_list = []

        self.accuracy_ir_lists = [[] for _ in range(self.top_k)]
        if self.use_itm:
            self.accuracy_ir_lists_itm = [[] for _ in range(self.top_k)]
            self.rank_true_images_list_itm = []

    def add_accuracy(self, caption, caption_id, true_image_id, accuracy_ir_1_top_k, rank, accuracy_ir_1_top_k_itm=None, rank_itm=None):
        accuracy_obj = {
            "caption": caption,
            "caption_id": caption_id,
            "true_image_id": true_image_id,
            "rank_true_image_id": rank,
            "accuracy_rate": {
                f"accuracy_ir_1_{self.top_k}": accuracy_ir_1_top_k
            }
        }

        self.rank_true_images_list.append(rank)

        if self.use_itm:
            accuracy_obj[f"rank_true_image_id_itm"] = rank_itm
            self.rank_true_images_list_itm.append(rank_itm)

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

        n_total = len(self.rank_true_images_list)
        n_none = sum(1 for r in self.rank_true_images_list if r is None)
        self.results["accuracy"]["rank_None"] = n_none
        if n_total > 0:
            self.results["accuracy"]["clean_mrr"] = (sum(reciprocal_rank(r) for r in self.rank_true_images_list) / n_total)
            self.results["accuracy"][f"clean_mrr_1_{self.top_k}"] = [
                (sum(reciprocal_rank(r, k) for r in self.rank_true_images_list) / n_total)
                for k in range(1, self.top_k + 1)
            ]
        else:
            self.results["accuracy"]["clean_mrr"] = None
            self.results["accuracy"][f"clean_mrr_1_{self.top_k}"] = [None for k in range(1, self.top_k + 1)]

        numeric_ranks = [int(r) for r in self.rank_true_images_list if (r is not None and int(r) > 0)]
        self.results["accuracy"]["clean_mean_rank"] = (
            (sum(numeric_ranks) / len(numeric_ranks)) if numeric_ranks else None
        )
        self.results["accuracy"]["clean_median_rank"] = (
            statistics.median(numeric_ranks) if numeric_ranks else None
        )

        if self.use_itm:
            accs_ir_1_top_k_itm = [sum(acc_ir_vals) / len(acc_ir_vals) for acc_ir_vals in self.accuracy_ir_lists_itm]
            self.results["accuracy"][f"accs_ir_1_{self.top_k}_itm"] = accs_ir_1_top_k_itm

            n_total_itm = len(self.rank_true_images_list_itm)
            n_none_itm = sum(1 for r in self.rank_true_images_list_itm if r is None)
            self.results["accuracy"]["rank_None_itm"] = n_none_itm
            if n_total_itm > 0:
                self.results["accuracy"]["clean_mrr_itm"] = (sum(reciprocal_rank(r) for r in self.rank_true_images_list_itm) / n_total_itm)
                self.results["accuracy"][f"clean_mrr_1_{self.top_k}_itm"] = [
                    (sum(reciprocal_rank(r, k) for r in self.rank_true_images_list_itm) / n_total_itm)
                    for k in range(1, self.top_k + 1)
                ]
            else:
                self.results["accuracy"]["clean_mrr_itm"] = None
                self.results["accuracy"][f"clean_mrr_1_{self.top_k}_itm"] = [None for k in range(1, self.top_k + 1)]
        
            numeric_ranks_itm = [int(r) for r in self.rank_true_images_list_itm if (r is not None and int(r) > 0)]
            self.results["accuracy"]["clean_mean_rank_itm"] = (
                (sum(numeric_ranks_itm) / len(numeric_ranks_itm)) if numeric_ranks_itm else None
            )
            self.results["accuracy"]["clean_median_rank_itm"] = (
                statistics.median(numeric_ranks_itm) if numeric_ranks_itm else None
            )

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

        self.clean_rank_true_images_lists = []
        self.adv_rank_true_images_lists = []

        self.clean_fault_ir_lists = None
        self.adv_fault_ir_lists = None

        if self.use_itm:
            self.adv_ir_lists_itm = [[] for _ in range(self.top_k)]
            self.adv_acc_ir_lists_itm = [[] for _ in range(self.top_k)]
            self.clean_ir_lists_itm = [[] for _ in range(self.top_k)]
            self.clean_acc_ir_lists_itm = [[] for _ in range(self.top_k)]

            self.clean_fault_ir_lists_itm = None
            self.adv_fault_ir_lists_itm = None

            self.clean_rank_true_images_lists_itm = []
            self.adv_rank_true_images_lists_itm = []

    def add_attack_result(self, caption, caption_id, ret_captions, ret_captions_ids, true_image_id, adv_image_id, adv_image_id_internal, adv_image_name, text_init, texts_arg, adv_sims, ir_1_top_k_results_dict, ir_1_top_k_results_dict_all, rank_results_dict_all=None, ir_1_top_k_itm_results_dict=None, ir_1_top_k_itm_results_dict_all=None, rank_itm_results_dict_all=None):

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
            "clean_rank_true_image_id_list": rank_results_dict_all["clean_rank"],
            f"all_ir_1_{self.top_k}_results": ir_1_top_k_results_dict_all,
            "attack_rate": {
                f"mean_adv_ir_1_{self.top_k}": adv_ir_1_top_k,
                f"mean_clean_ir_1_{self.top_k}": clean_ir_1_top_k,
                f"mean_adv_acc_ir_1_{self.top_k}": adv_acc_ir_1_top_k,
                f"mean_clean_acc_ir_1_{self.top_k}": clean_acc_ir_1_top_k,
                "adv_rank_true_image_id_list": rank_results_dict_all["adv_rank"]
            }
        }

        if "clean_fault_ir_list" in ir_1_top_k_results_dict.keys() and "adv_fault_ir_list" in ir_1_top_k_results_dict.keys():
            clean_fault_ir_1_top_k = ir_1_top_k_results_dict["clean_fault_ir_list"]
            attack_obj["attack_rate"][f"mean_clean_fault_ir_1_{self.top_k}"] = clean_fault_ir_1_top_k
            adv_fault_ir_1_top_k = ir_1_top_k_results_dict["adv_fault_ir_list"]
            attack_obj["attack_rate"][f"mean_adv_fault_ir_1_{self.top_k}"] = adv_fault_ir_1_top_k

        if self.use_itm and ir_1_top_k_itm_results_dict is not None:
            attack_obj[f"all_ir_1_{self.top_k}_itm_results"] = ir_1_top_k_itm_results_dict_all

            attack_obj["attack_rate"][f"mean_adv_ir_1_{self.top_k}_itm"] = adv_ir_1_top_k_itm
            attack_obj["attack_rate"][f"mean_clean_ir_1_{self.top_k}_itm"] = clean_ir_1_top_k_itm
            attack_obj["attack_rate"][f"mean_adv_acc_ir_1_{self.top_k}_itm"] = adv_acc_ir_1_top_k_itm
            attack_obj["attack_rate"][f"mean_clean_acc_ir_1_{self.top_k}_itm"] = clean_acc_ir_1_top_k_itm

            if "clean_fault_ir_list_itm" in ir_1_top_k_itm_results_dict.keys() and "adv_fault_ir_list_itm" in ir_1_top_k_itm_results_dict.keys():
                clean_fault_ir_1_top_k_itm = ir_1_top_k_itm_results_dict["clean_fault_ir_list_itm"]
                attack_obj["attack_rate"][f"mean_clean_fault_ir_1_{self.top_k}_itm"] = clean_fault_ir_1_top_k_itm
                adv_fault_ir_1_top_k_itm = ir_1_top_k_itm_results_dict["adv_fault_ir_list_itm"]
                attack_obj["attack_rate"][f"mean_adv_fault_ir_1_{self.top_k}_itm"] = adv_fault_ir_1_top_k_itm

        self.clean_rank_true_images_lists.append(rank_results_dict_all["clean_rank"])
        self.adv_rank_true_images_lists.append(rank_results_dict_all["adv_rank"])

        for i_ir, adv_ir_val in enumerate(adv_ir_1_top_k):
            self.adv_ir_lists[i_ir].append(adv_ir_val)
        for i_ir, adv_acc_ir_val in enumerate(adv_acc_ir_1_top_k):
            self.adv_acc_ir_lists[i_ir].append(adv_acc_ir_val)
        for i_ir, clean_ir_val in enumerate(clean_ir_1_top_k):
            self.clean_ir_lists[i_ir].append(clean_ir_val)
        for i_ir, clean_acc_ir_val in enumerate(clean_acc_ir_1_top_k):
            self.clean_acc_ir_lists[i_ir].append(clean_acc_ir_val)

        if "clean_fault_ir_list" in ir_1_top_k_results_dict.keys() and "adv_fault_ir_list" in ir_1_top_k_results_dict.keys():
            if self.clean_fault_ir_lists is None:
                self.clean_fault_ir_lists = [[] for _ in range(self.top_k)]
            if self.adv_fault_ir_lists is None:
                self.adv_fault_ir_lists = [[] for _ in range(self.top_k)]

            for i_ir, clean_fault_ir_val in enumerate(clean_fault_ir_1_top_k):
                self.clean_fault_ir_lists[i_ir].append(clean_fault_ir_val)
            
            for i_ir, adv_fault_ir_val in enumerate(adv_fault_ir_1_top_k):
                self.adv_fault_ir_lists[i_ir].append(adv_fault_ir_val)

        if self.use_itm:
            attack_obj["clean_rank_true_image_id_list_itm"] = rank_itm_results_dict_all["clean_rank_itm"]
            attack_obj["attack_rate"]["adv_rank_true_image_id_list_itm"] = rank_itm_results_dict_all["adv_rank_itm"]

            self.clean_rank_true_images_lists_itm.append(rank_itm_results_dict_all["clean_rank_itm"])
            self.adv_rank_true_images_lists_itm.append(rank_itm_results_dict_all["adv_rank_itm"])

            for i_ir, adv_ir_val in enumerate(adv_ir_1_top_k_itm):
                self.adv_ir_lists_itm[i_ir].append(adv_ir_val)
            for i_ir, adv_acc_ir_val in enumerate(adv_acc_ir_1_top_k_itm):
                self.adv_acc_ir_lists_itm[i_ir].append(adv_acc_ir_val)
            for i_ir, clean_ir_val in enumerate(clean_ir_1_top_k_itm):
                self.clean_ir_lists_itm[i_ir].append(clean_ir_val)
            for i_ir, clean_acc_ir_val in enumerate(clean_acc_ir_1_top_k_itm):
                self.clean_acc_ir_lists_itm[i_ir].append(clean_acc_ir_val)

            if "clean_fault_ir_list_itm" in ir_1_top_k_itm_results_dict.keys() and "adv_fault_ir_list_itm" in ir_1_top_k_itm_results_dict.keys():
                if self.clean_fault_ir_lists_itm is None:
                    self.clean_fault_ir_lists_itm = [[] for _ in range(self.top_k)]
                if self.adv_fault_ir_lists_itm is None:
                    self.adv_fault_ir_lists_itm = [[] for _ in range(self.top_k)]

                for i_ir, clean_fault_ir_val in enumerate(clean_fault_ir_1_top_k_itm):
                    self.clean_fault_ir_lists_itm[i_ir].append(clean_fault_ir_val)
            
                for i_ir, adv_fault_ir_val in enumerate(adv_fault_ir_1_top_k_itm):
                    self.adv_fault_ir_lists_itm[i_ir].append(adv_fault_ir_val)

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

        if self.clean_fault_ir_lists is not None:
            self.results["asr"].update({
                f"final_clean_fault_ir_1_{self.top_k}": [sum(vals) / len(vals) if vals else 0 for vals in self.clean_fault_ir_lists],
            })
        
        if self.adv_fault_ir_lists is not None:
            self.results["asr"].update({
                f"final_adv_fault_ir_1_{self.top_k}": [sum(vals) / len(vals) if vals else 0 for vals in self.adv_fault_ir_lists],
            })
        
        complete_clean_rank_true_images_list = [r for sublist in self.clean_rank_true_images_lists for r in sublist]
        n_total_clean = len(complete_clean_rank_true_images_list)
        n_none_clean = sum(1 for r in complete_clean_rank_true_images_list if r is None)
        self.results["asr"].update({
            "clean_rank_None": n_none_clean
        })
        if n_total_clean > 0:
            self.results["asr"].update({
                "clean_mrr": (sum(reciprocal_rank(r) for r in complete_clean_rank_true_images_list) / n_total_clean),
                f"clean_mrr_1_{self.top_k}": [
                    (sum(reciprocal_rank(r, k) for r in complete_clean_rank_true_images_list) / n_total_clean)
                    for k in range(1, self.top_k + 1)
                ]
            })
        else:
            self.results["asr"].update({
                "clean_mrr": None,
                f"clean_mrr_1_{self.top_k}": [None for k in range(1, self.top_k + 1)]
            })
        clean_numeric_ranks = [int(r) for r in complete_clean_rank_true_images_list if (r is not None and int(r) > 0)]
        self.results["asr"].update({
            "clean_mean_rank": (
                (sum(clean_numeric_ranks) / len(clean_numeric_ranks)) if clean_numeric_ranks else None
            ),
            "clean_median_rank": (
                statistics.median(clean_numeric_ranks) if clean_numeric_ranks else None
            )
        })

        # ----------

        complete_adv_rank_true_images_list = [r for sublist in self.adv_rank_true_images_lists for r in sublist]
        n_total_adv = len(complete_adv_rank_true_images_list)
        n_none_adv = sum(1 for r in complete_adv_rank_true_images_list if r is None)
        self.results["asr"].update({
            "adv_rank_None": n_none_adv
        })
        if n_total_adv > 0:
            self.results["asr"].update({
                "adv_mrr": (sum(reciprocal_rank(r) for r in complete_adv_rank_true_images_list) / n_total_adv),
                f"adv_mrr_1_{self.top_k}": [
                    (sum(reciprocal_rank(r, k) for r in complete_adv_rank_true_images_list) / n_total_adv)
                    for k in range(1, self.top_k + 1)
                ]
            })
        else:
            self.results["asr"].update({
                "adv_mrr": None,
                f"adv_mrr_1_{self.top_k}": [None for k in range(1, self.top_k + 1)]
            })
        adv_numeric_ranks = [int(r) for r in complete_adv_rank_true_images_list if (r is not None and int(r) > 0)]
        self.results["asr"].update({
            "adv_mean_rank": (
                (sum(adv_numeric_ranks) / len(adv_numeric_ranks)) if adv_numeric_ranks else None
            ),
            "adv_median_rank": (
                statistics.median(adv_numeric_ranks) if adv_numeric_ranks else None
            )
        })

        if self.use_itm:
            complete_clean_rank_true_images_list_itm = [r for sublist in self.clean_rank_true_images_lists_itm for r in sublist]
            n_total_clean_itm = len(complete_clean_rank_true_images_list_itm)
            n_none_clean_itm = sum(1 for r in complete_clean_rank_true_images_list_itm if r is None)
            self.results["asr"].update({
                "clean_rank_None_itm": n_none_clean_itm
            })
            if n_total_clean_itm > 0:
                self.results["asr"].update({
                    "clean_mrr_itm": (sum(reciprocal_rank(r) for r in complete_clean_rank_true_images_list_itm) / n_total_clean_itm),
                    f"clean_mrr_1_{self.top_k}_itm": [
                        (sum(reciprocal_rank(r, k) for r in complete_clean_rank_true_images_list_itm) / n_total_clean_itm)
                        for k in range(1, self.top_k + 1)
                    ]
                })
            else:
                self.results["asr"].update({
                    "clean_mrr_itm": None,
                    f"clean_mrr_1_{self.top_k}_itm": [None for k in range(1, self.top_k + 1)]
                })
            clean_numeric_ranks_itm = [int(r) for r in complete_clean_rank_true_images_list_itm if (r is not None and int(r) > 0)]
            self.results["asr"].update({
                "clean_mean_rank_itm": (
                    (sum(clean_numeric_ranks_itm) / len(clean_numeric_ranks_itm)) if clean_numeric_ranks_itm else None
                ),
                "clean_median_rank_itm": (
                    statistics.median(clean_numeric_ranks_itm) if clean_numeric_ranks_itm else None
                )
            })

            # ----------

            complete_adv_rank_true_images_list_itm = [r for sublist in self.adv_rank_true_images_lists_itm for r in sublist]
            n_total_adv_itm = len(complete_adv_rank_true_images_list_itm)
            n_none_adv_itm = sum(1 for r in complete_adv_rank_true_images_list_itm if r is None)
            self.results["asr"].update({
                "adv_rank_None_itm": n_none_adv_itm
            })
            if n_total_adv_itm > 0:
                self.results["asr"].update({
                    "adv_mrr_itm": (sum(reciprocal_rank(r) for r in complete_adv_rank_true_images_list_itm) / n_total_adv_itm),
                    f"adv_mrr_1_{self.top_k}_itm": [
                        (sum(reciprocal_rank(r, k) for r in complete_adv_rank_true_images_list_itm) / n_total_adv_itm)
                        for k in range(1, self.top_k + 1)
                    ]
                })
            else:
                self.results["asr"].update({
                    "adv_mrr_itm": None,
                    f"adv_mrr_1_{self.top_k}_itm": [None for k in range(1, self.top_k + 1)]
                })
            adv_numeric_ranks_itm = [int(r) for r in complete_adv_rank_true_images_list_itm if (r is not None and int(r) > 0)]
            self.results["asr"].update({
                "adv_mean_rank_itm": (
                    (sum(adv_numeric_ranks_itm) / len(adv_numeric_ranks_itm)) if adv_numeric_ranks_itm else None
                ),
                "adv_median_rank_itm": (
                    statistics.median(adv_numeric_ranks_itm) if adv_numeric_ranks_itm else None
                )
            })

            self.results["asr"].update({
                f"final_adv_ir_1_{self.top_k}_itm": [sum(vals) / len(vals) if vals else 0 for vals in self.adv_ir_lists_itm],
                f"final_clean_ir_1_{self.top_k}_itm": [sum(vals) / len(vals) if vals else 0 for vals in self.clean_ir_lists_itm],
                f"final_adv_acc_ir_1_{self.top_k}_itm": [sum(vals) / len(vals) if vals else 0 for vals in self.adv_acc_ir_lists_itm],
                f"final_clean_acc_ir_1_{self.top_k}_itm": [sum(vals) / len(vals) if vals else 0 for vals in self.clean_acc_ir_lists_itm],
            })

            if self.clean_fault_ir_lists_itm is not None:
                self.results["asr"].update({
                    f"final_clean_fault_ir_1_{self.top_k}_itm": [sum(vals) / len(vals) if vals else 0 for vals in self.clean_fault_ir_lists_itm],
                })
            
            if self.adv_fault_ir_lists_itm is not None:
                self.results["asr"].update({
                    f"final_adv_fault_ir_1_{self.top_k}_itm": [sum(vals) / len(vals) if vals else 0 for vals in self.adv_fault_ir_lists_itm],
                })

    def get_results(self):
        return self.results

# -----

class IndependentAccuracyAttackManager:
    def __init__(self, args, config_retrieval, config_attack, clean_accuracy_attack_results_dict_all, top_k=10, num_sel_dict=None, len_clean_results_acc=None, use_itm=False):
        
        clean_accuracy_attack_results_mean = mean_dict(clean_accuracy_attack_results_dict_all)
        
        self.results = {
            "args": vars(args),
            "config_retrieval": config_retrieval,
            "config_attack": config_attack,
            "adv_list": [],
            "num_sel_dict": num_sel_dict,
            "len_clean_results_acc": len_clean_results_acc,
            "time": 0,
            "clean_accuracy_attack_results_dict_all": clean_accuracy_attack_results_dict_all,
            "mean_original_clean": clean_accuracy_attack_results_mean,
            "asr": {}
        }
        self.top_k = top_k
        self.use_itm = use_itm

        self.len_adv_results_acc_list = []

        self.rank_None_list = []
        self.clean_mrr_list = []
        self.clean_mrr_1_k_lists = [[] for _ in range(self.top_k)]
        self.clean_mean_rank_list = []
        self.clean_median_rank_list = []

        self.adv_abroad_acc_ir_lists = [[] for _ in range(self.top_k)]

        if self.use_itm:
            self.adv_abroad_acc_ir_lists_itm = [[] for _ in range(self.top_k)]

            self.rank_None_list_itm = []
            self.clean_mrr_list_itm = []
            self.clean_mrr_1_k_lists_itm = [[] for _ in range(self.top_k)]
            self.clean_mean_rank_list_itm = []
            self.clean_median_rank_list_itm = []

    def add_attack_result(self, caption, caption_id, true_image_id, adv_image_id, adv_image_id_internal, adv_image_name, text_init, texts_arg, adv_accuracy_attack_results_dict, len_adv_results_acc=None):

        adv_abroad_acc_ir_1_top_k = adv_accuracy_attack_results_dict[f"accs_ir_1_{self.top_k}"]
        
        attack_obj = {
            "caption": caption,
            "caption_id": caption_id,
            "true_image_id": true_image_id,
            "adv_image_id": adv_image_id,
            "adv_image_id_internal": adv_image_id_internal,
            "adv_image_name": adv_image_name,
            "text_init": text_init,
            "texts_arg": texts_arg,
            "len_adv_results_acc": len_adv_results_acc,
            "attack_rate": {
                f"mean_adv_abroad_acc_ir_1_{self.top_k}": adv_abroad_acc_ir_1_top_k
            }
        }

        self.len_adv_results_acc_list.append(len_adv_results_acc)

        self.rank_None_list.append(adv_accuracy_attack_results_dict["rank_None"])
        self.clean_mrr_list.append(adv_accuracy_attack_results_dict["clean_mrr"])
        for i_ir, clean_mrr_val in enumerate(adv_accuracy_attack_results_dict[f"clean_mrr_1_{self.top_k}"]):
            self.clean_mrr_1_k_lists[i_ir].append(clean_mrr_val)
        self.clean_mean_rank_list.append(adv_accuracy_attack_results_dict["clean_mean_rank"])
        self.clean_median_rank_list.append(adv_accuracy_attack_results_dict["clean_median_rank"])

        if self.use_itm:
            self.rank_None_list_itm.append(adv_accuracy_attack_results_dict["rank_None_itm"])
            self.clean_mrr_list_itm.append(adv_accuracy_attack_results_dict["clean_mrr_itm"])
            for i_ir, clean_mrr_val in enumerate(adv_accuracy_attack_results_dict[f"clean_mrr_1_{self.top_k}_itm"]):
                self.clean_mrr_1_k_lists_itm[i_ir].append(clean_mrr_val)
            self.clean_mean_rank_list_itm.append(adv_accuracy_attack_results_dict["clean_mean_rank_itm"])
            self.clean_median_rank_list_itm.append(adv_accuracy_attack_results_dict["clean_median_rank_itm"])

            adv_abroad_acc_ir_1_top_k_itm = adv_accuracy_attack_results_dict[f"accs_ir_1_{self.top_k}_itm"]
            attack_obj["attack_rate"][f"mean_adv_abroad_acc_ir_1_{self.top_k}_itm"] = adv_abroad_acc_ir_1_top_k_itm

        for i_ir, adv_ir_val in enumerate(adv_abroad_acc_ir_1_top_k):
            self.adv_abroad_acc_ir_lists[i_ir].append(adv_ir_val)

        if self.use_itm:
            for i_ir, adv_ir_val in enumerate(adv_abroad_acc_ir_1_top_k_itm):
                self.adv_abroad_acc_ir_lists_itm[i_ir].append(adv_ir_val)

        self.results["adv_list"].append(attack_obj)

    def set_time(self, time_elapsed):
        self.results["time"] = time_elapsed

    def compute_averages(self):
        len_adv_results_acc_list_not_none = [l for l in self.len_adv_results_acc_list if l is not None]

        self.results["asr"] = {
            "all_not_none_in_len_adv_results_acc_list": all(x is not None for x in self.len_adv_results_acc_list),
            "final_mean_len_adv_results_acc": (sum(len_adv_results_acc_list_not_none) / len(len_adv_results_acc_list_not_none)) if len_adv_results_acc_list_not_none else None,
            "final_max_len_adv_results_acc": max(len_adv_results_acc_list_not_none) if len_adv_results_acc_list_not_none else None,
            "final_min_len_adv_results_acc": min(len_adv_results_acc_list_not_none) if len_adv_results_acc_list_not_none else None,
            f"final_adv_abroad_acc_ir_1_{self.top_k}": [sum(vals) / len(vals) if vals else 0 for vals in self.adv_abroad_acc_ir_lists]
        }

        self.results["asr"].update({
            "final_adv_abroad_rank_None_max": None if all(x is None for x in self.rank_None_list) else max(x for x in self.rank_None_list if x is not None),
            "final_adv_abroad_clean_mrr": sum(self.clean_mrr_list)/len(self.clean_mrr_list) if len(self.clean_mrr_list) else None,
            f"final_adv_abroad_clean_mrr_1_{self.top_k}": [sum(vals) / len(vals) if vals else 0 for vals in self.clean_mrr_1_k_lists],
            "final_adv_abroad_clean_mean_rank": sum(self.clean_mean_rank_list)/len(self.clean_mean_rank_list) if len(self.clean_mean_rank_list) else None,
            "final_adv_abroad_clean_median_rank": sum(self.clean_median_rank_list)/len(self.clean_median_rank_list) if len(self.clean_median_rank_list) else None
        })

        if self.use_itm:
            self.results["asr"].update({
                f"final_adv_abroad_acc_ir_1_{self.top_k}_itm": [sum(vals) / len(vals) if vals else 0 for vals in self.adv_abroad_acc_ir_lists_itm]
            })

            self.results["asr"].update({
                "final_adv_abroad_rank_None_max_itm": None if all(x is None for x in self.rank_None_list_itm) else max(x for x in self.rank_None_list_itm if x is not None),
                "final_adv_abroad_clean_mrr_itm": sum(self.clean_mrr_list_itm)/len(self.clean_mrr_list_itm) if len(self.clean_mrr_list_itm) else None,
                f"final_adv_abroad_clean_mrr_1_{self.top_k}_itm": [sum(vals) / len(vals) if vals else 0 for vals in self.clean_mrr_1_k_lists_itm],
                "final_adv_abroad_clean_mean_rank_itm": sum(self.clean_mean_rank_list_itm)/len(self.clean_mean_rank_list_itm) if len(self.clean_mean_rank_list_itm) else None,
                "final_adv_abroad_clean_median_rank_itm": sum(self.clean_median_rank_list_itm)/len(self.clean_median_rank_list_itm) if len(self.clean_median_rank_list_itm) else None
            })

    def get_results(self):
        return self.results