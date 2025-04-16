# COLLISIO

## 📦 Installation

Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed.

1. **Create the environment**

    ```bash
    conda create --name collisio python=3.9.19 -y
    ```

2. **Activate the environment**

    ```bash
    conda activate collisio
    ```

3. **Install dependencies in order** (to avoid version conflicts)

    ```bash
    pip install ruamel-yaml==0.18.6
    pip install torch==2.3.0 torchvision==0.18.0
    pip install ftfy==6.1.1
    pip install timm==0.4.12
    pip install salesforce-lavis==1.0.2
    pip install numpy==1.26.4
    pip install nltk==3.8.1
    pip install wordhoard==1.5.4
    pip install transformers==4.37.2
    ```

---


## 🚀 Usage

### 1. Run accuracy

The script `accuracy.py` computes the initial retrieval accuracies and generates cleaned files that are required for the attack phase.

#### 📌 Example:

```bash
CUDA_VISIBLE_DEVICES=0 python accuracy.py \
    --subset_size 500 \
    --config_retrieval './configs/Retrieval_flickr_server.yaml' \
    --batch_size 32
```

### 2. Run the poisoning attack

The script `attack.py` performs the attack using the clean outputs provided in step 1.

#### 📌 Example:

```bash
CUDA_VISIBLE_DEVICES=0 python attack.py \
    --subset_size 500 \
    --config_attack './configs/no_aug/absolute_configs_4-255/ret:all/config_attack_exp1.yaml' \
    --config_retrieval './configs/Retrieval_flickr_server.yaml' \
    --batch_size 32
```

#### 🔧 Arguments:

- `--config_retrieval`: path to the YAML config file used for the retrieval setup.
- `--subset_size`: limit the number of evaluated samples (use `inf` to evaluate all; in the example: `500`).
- `--retrieval_mode_img`: mode for candidate retrieval images:  
  - `split`: use Karpathy test split (default).  
  - `all`: use all images from the evaluation set.
- `--retrieval_mode_target_img`: mode for selecting da dove poter pescare the poisoning target:  
  - `split`: restrict to the Karpathy split (default).  
  - `all`: search in the full evaluation set.
- `--batch_size`: batch size used during evaluation.
- `--seed`: random seed for reproducibility.
- `--source_model`: vision-language model used for retrieval.  
  Choices:  
  - `ViT-B/16`: CLIP with ViT-B/16 backbone  
  - `RN101`: CLIP with ResNet-101 backbone  
  - `BLIP-2`: multimodal transformer-based model with Q-Former architecture

**Only for `attack.py`:**

- `--config_attack`: path to the attack configuration YAML file (used to control attack strategy and parameters).

#### 🧠 Attack configuration types (`--config_attack`)

The attack strategy is controlled via configuration files located in the `configs/` directory. Below are the available options:

- `no_aug_query_to_query/`: full knowledge of the user query (`qt = qu`).
- `no_aug/`: partial knowledge of the user query (`qt != qu`), **without** Collisio's EoQ.
- `syn/`: Collisio using **TrSyn** as transformation strategy.
- `caption_llms/`: Collisio using **TrLLM** as transformation strategy.
- `pred_ic/`: Collisio using **TrIC-1** transformation strategy.
- `pred_ic_ret5/`: Collisio using **TrIC-5** transformation strategy.

#### 🛠️ Augmentation scripts (`helper_augmentation/`)

In the `helper_augmentation` folder, you can find the scripts used to generate the augmented captions employed in the following transformation strategies:

- `caption_llm/`: uses LLMs to generate captions.
- `pred_ic/`: uses multimodal models to generate captions.

#### 📊 Paper results

In the `paper_complete_results/` folder, you can find the full set of experimental results referenced in the paper (see `Collisio_complete_results.pdf`).