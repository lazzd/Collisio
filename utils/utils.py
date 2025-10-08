import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import torchvision.transforms.v2 as T

import random


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

# -----

def set_all_seed(seed_val):
    seed = seed_val + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

# -----

def compress_images_jpeg(images, quality):
    jpeg_transform = T.JPEG(quality=quality)

    single = False
    if images.dim() == 3:
        images = images.unsqueeze(0)
        single = True

    if images.dtype.is_floating_point:
        min_val, max_val = images.min().item(), images.max().item()
        if max_val > 1.0:
            images = images / 255.0
        elif min_val < 0:
            raise ValueError("Invalid float tensor for JPEG compression.")
        images_uint8 = (images.clamp(0,1) * 255).round().to(torch.uint8).cpu()
    elif images.dtype == torch.uint8:
        images_uint8 = images.cpu()
    else:
        raise TypeError(f"Unsupported dtype {images.dtype}. Use float32 or uint8.")

    compressed = jpeg_transform(images_uint8)

    compressed = compressed.to(torch.float32) / 255.0

    return compressed[0] if single else compressed