import numpy as np
import scipy.stats as st
import torch


# -----------

def clamp_by_l2(x, max_norm):
    norm = torch.norm(x, dim=(1,2,3), p=2, keepdim=True)
    factor = torch.min(max_norm / norm, torch.ones_like(norm))
    return x * factor

def random_init(x, norm_type, epsilon):
    delta = torch.zeros_like(x)
    assert norm_type == 'Linf' or norm_type == 'L2'
    if norm_type == 'Linf':
        delta.data.uniform_(0.0, 1.0)
        delta.data = delta.data * epsilon
    elif norm_type == 'L2':
        delta.data.uniform_(0.0, 1.0)
        delta.data = delta.data - x
        delta.data = clamp_by_l2(delta.data, epsilon)
    return delta

def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  return kernel


# -----------


class SimImageAttacker():
    def __init__(self, normalize_embeds):
        self.normalize_embeds = normalize_embeds
    
    def run_sim_attack(self, model, images, target_texts_embeds, num_iters, true_images=None):
        pass

    def embed_normalize(self, embed, dim=-1):
        return embed / embed.norm(dim=dim, keepdim=True)