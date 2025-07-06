#!/usr/bin/env python3
"""Class for handling thor inference
"""

import torch
import numpy as np
from torchvision import transforms
from get_anomaly_masks_optuna import one_img_functional, diff_fns_dict

import matplotlib.pyplot as plt


class ThorItem():
    def __init__(
            self, 
            timesteps, 
            scheduler, 
            cfg_dict, 
            clean_img=None, 
            noise=None, 
            dbg=False,
            use_x0=False
            ):

        self.timesteps = timesteps  # list of int timesteps to inject harmonisations
        self.scheduler = scheduler
        self.blur_k = cfg_dict["blur_k"]
        self.threshold = cfg_dict["threshold"]
        self.noise_removal_thresh = cfg_dict["min_area"]
        self.get_diff_str = cfg_dict["diff_metric"]
        self.use_x0 = use_x0

        self.is_dbg = dbg

        if not clean_img is None:
            self.set_clean_img(clean_img)

        if not noise is None:
            self.noise = self.set_noise(noise)

    def set_noise(self, noise):
        self.noise = noise

    def set_clean_img(self, clean_img):
        self.clean_img = clean_img

    def step(self, xt, x0, t):
        # wow i hate this so much
        x0_batch = x0 * 0.5 + 0.5
        ci_batch = self.clean_img * 0.5 + 0.5
        diff_list = []
        for x0_unnorm, ci in zip(x0_batch, ci_batch):
            _, diff = one_img_functional(
                        transforms.functional.to_pil_image(x0_unnorm.cpu()),
                        transforms.functional.to_pil_image(ci.cpu()),
                        self.blur_k,
                        self.threshold,
                        self.noise_removal_thresh,
                        get_diff_str=self.get_diff_str
                        )
            diff_list.append(diff)
        diff_batch = torch.as_tensor(np.array(diff_list) , device=xt.device)
        diff_batch = diff_batch / diff_batch.max()

        diff = diff_batch.to(xt.device).type(xt.dtype)
        diff = torch.unsqueeze(diff, 1)
        
        t_list = torch.tensor([t]).long()
        t_list = torch.unsqueeze(t_list, 0)
        t_list.expand(self.clean_img.shape[0], -1).to(self.noise.device)

        if self.use_x0:
            dirty_img = self.scheduler.add_noise(self.clean_img, self.noise, t_list)
            interpolated = diff * xt + (1-diff) * dirty_img 
        else:
            interpolated = diff * x0 + (1-diff) * self.clean_img 
            interpolated = self.scheduler.add_noise(interpolated, self.noise, t_list)

        if self.is_dbg:  # dbg vis
            fig, axs = plt.subplots(2, 3)
            axs[0,0].imshow(diff[0][0].cpu())
            axs[0,1].imshow(xt[0].moveaxis(0,-1).cpu())
            axs[0,2].imshow(dirty_img[0].moveaxis(0,-1).cpu())

            dbg_1 =  diff * xt
            axs[1,1].imshow(dbg_1[0].moveaxis(0,-1).cpu())
            dbg_2 = (1-diff) * dirty_img
            axs[1,2].imshow(dbg_2[0].moveaxis(0,-1).cpu())

            axs[1,0].imshow(interpolated[0].moveaxis(0,-1).cpu())
            
            plt.show()

        return interpolated

