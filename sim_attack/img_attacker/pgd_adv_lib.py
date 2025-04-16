from distutils.version import LooseVersion

import numbers
from functools import partial
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.autograd import grad
from torch.nn import functional as F

from sim_attack.img_attacker.img_attacker_utils import SimImageAttacker


use_tensors_in_clamp = False
if LooseVersion(torch.__version__) >= LooseVersion('1.9'):
    use_tensors_in_clamp = True


@torch.no_grad()
def clamp(x: Tensor, lower: Tensor, upper: Tensor, inplace: bool = False) -> Tensor:
    """Clamp based on lower and upper Tensor bounds. Clamping method depends on torch version: clamping with tensors was
    introduced in torch 1.9."""
    δ_clamped = x if inplace else None
    if use_tensors_in_clamp:
        δ_clamped = torch.clamp(x, min=lower, max=upper, out=δ_clamped)
    else:
        δ_clamped = torch.maximum(x, lower, out=δ_clamped)
        δ_clamped = torch.minimum(δ_clamped, upper, out=δ_clamped)
    return δ_clamped


def clamp_(x: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """In-place alias for clamp."""
    return clamp(x=x, lower=lower, upper=upper, inplace=True)


class PGD_linfSimImageAttacker(SimImageAttacker):
    def __init__(self,
                 preprocess=None,
                 normalize_embeds=True,
                 margin_error: float = 0.0,
                 ε: Union[float, Tensor] = 2/255,
                 use_difference: bool = False,
                 random_init: bool = True,
                 scaled_step_size: bool = False,
                 relative_step_size: float = 0.01 / 0.3, # param.
                 absolute_step_size: Optional[float] = None, # param.
                 restarts: int = 1, # param.
                 verbose: bool = False
                ):
        super().__init__(normalize_embeds)
        self.preprocess = preprocess
        self.margin_error = margin_error
        if '/' in ε:
            numer = ε.split("/")[0]
            denumer = ε.split("/")[-1]
            self.ε = float(numer)/float(denumer)
        else:
            self.ε = ε
        self.use_difference = use_difference
        self.random_init = random_init
        self.scaled_step_size = scaled_step_size
        self.relative_step_size = relative_step_size
        self.absolute_step_size = absolute_step_size
        self.restarts = restarts
        self.verbose = verbose
    
    def _pgd_linf(self,
                  model: nn.Module,
                  images: Tensor,
                  target_texts_embeds: Tensor,
                  mask_target_texts_embeds: Tensor,
                  best_cosine_sims: Tensor,
                  ε: Tensor,
                  steps: int = 40) -> Tuple[Tensor, Tensor]:

        def _loss_function_difference(sim, best_sims, mask_inputs):
            diff_sim = best_sims - sim
            diff_sim = diff_sim.clip(0)
            diff_sim_for_sample = diff_sim.sum(1)/(mask_inputs!=False).sum(1)
            diff_sim_for_sample = torch.nan_to_num(diff_sim_for_sample)
            dl_loss = torch.mean(diff_sim_for_sample)

            return dl_loss

        def _loss_function(sim, mask_inputs):
            sim_for_sample = sim.sum(1)/(mask_inputs!=False).sum(1)
            sim_for_sample = torch.nan_to_num(sim_for_sample)

            return sim_for_sample

        device = images.device
        batch_size = len(images)
        batch_view = lambda tensor: tensor.view(batch_size, *[1] * (images.ndim - 1))
        lower, upper = torch.maximum(-images, -batch_view(ε)), torch.minimum(1 - images, batch_view(ε))

        multiplier = 1

        if self.use_difference:
            loss_func = _loss_function_difference
        else:
            loss_func = _loss_function

        if self.scaled_step_size:
            step_size = ε / steps * 1.25
        else:
            step_size: Tensor = ε *  self.relative_step_size if  self.absolute_step_size is None else torch.full_like( ε,  self.absolute_step_size)
        
        if self.use_difference:
            step_size *= -1

        δ = torch.zeros_like(images, requires_grad=True)
        best_adv = images.clone()
        adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if  self.random_init:
            δ.data.uniform_(-1, 1).mul_(batch_view(ε))
            clamp_(δ, lower=lower, upper=upper)

        for i in range(steps):
            adv_images = images + δ

            if self.preprocess is not None:
                adv_inputs = self.preprocess(adv_images)
            else:
                adv_inputs = adv_images

            images_adv_embed = model.inference_image(adv_inputs)['image_feat']
            
            if images_adv_embed.dim() == 3:
                images_adv_embed = torch.mean(images_adv_embed, dim=1)

            if self.normalize_embeds:
                images_adv_embed = self.embed_normalize(images_adv_embed)
            images_adv_embed = images_adv_embed.unsqueeze(1)

            images_adv_embed_transposed = images_adv_embed.transpose(1, 2)

            sim = (target_texts_embeds @ images_adv_embed_transposed).squeeze(-1)

            if self.use_difference:
                loss = multiplier * loss_func(sim, best_cosine_sims, mask_target_texts_embeds)
            else:
                loss = multiplier * loss_func(sim, mask_target_texts_embeds)

            loss_sum = loss.sum()
            loss_sum.backward()

            δ_grad = δ.grad.clone()
            δ_grad = δ_grad.sign_()

            is_adv = (sim >= best_cosine_sims).all(dim=1)

            best_adv = torch.where(batch_view(is_adv), adv_images.detach(), best_adv)
            
            adv_found.logical_or_(is_adv)

            δ.data.addcmul_(batch_view(step_size), δ_grad)
            clamp_(δ, lower=lower, upper=upper)

            δ.grad.data.zero_()

        return adv_found, best_adv
    
    def run_sim_attack(self, model: nn.Module, images: Tensor, target_texts_embeds: Tensor, num_iters: int = 40, true_images=None):
        model.eval()

        if self.use_difference:
            assert true_images is not None

        device = model.logit_scale.device
        batch_size = len(images)

        images = images.to(device)
        true_images = true_images.to(device)
        target_texts_embeds = target_texts_embeds.to(device)
        if self.normalize_embeds:
            target_texts_embeds = self.embed_normalize(target_texts_embeds)
            target_texts_embeds = torch.nan_to_num(target_texts_embeds)

        mask_target_texts_embeds = ~torch.any(torch.eq(target_texts_embeds, 0), dim=2)

        if self.use_difference:
            with torch.no_grad():
                if self.preprocess is not None:
                    true_images = self.preprocess(true_images)
                original_true_images_embed = model.inference_image(true_images)['image_feat']

                if original_true_images_embed.dim() == 3:
                    original_true_images_embed = torch.mean(original_true_images_embed, dim=1)

                if self.normalize_embeds:
                    original_true_images_embed = self.embed_normalize(original_true_images_embed)
                original_true_images_embed = original_true_images_embed.unsqueeze(1)

                original_true_images_embed = original_true_images_embed.transpose(1, 2)

                best_cosine_sims = (target_texts_embeds @ original_true_images_embed).squeeze(-1)
                best_cosine_sims = torch.abs(best_cosine_sims)
                best_cosine_sims += self.margin_error
        else:
            best_cosine_sims = torch.full((target_texts_embeds.shape[0],target_texts_embeds.shape[1],), -float('inf'), device=device)

        adv_inputs = images.clone()
        adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if isinstance(self.ε, numbers.Real):
            ε = torch.full_like(adv_found, self.ε, dtype=images.dtype)
        else:
            ε = self.ε
        
        pgd_attack = partial(self._pgd_linf, model=model, steps=num_iters)

        for i in range(self.restarts):
            adv_found_run, adv_inputs_run = pgd_attack(images=images[~adv_found],
                                                       target_texts_embeds=target_texts_embeds[~adv_found],
                                                       mask_target_texts_embeds=mask_target_texts_embeds[~adv_found],
                                                       best_cosine_sims=best_cosine_sims[~adv_found],
                                                       ε=ε[~adv_found])

            adv_inputs[~adv_found] = adv_inputs_run
            adv_found[~adv_found] = adv_found_run

            if self.verbose:
                print('Success', i + 1, adv_found.float().mean())
                print("Adv Found", adv_found)

            if adv_found.all():
                break

        return adv_inputs.cpu().detach()