#!/usr/bin/env python3


import os
from PIL import Image 
import numpy as np
import cv2
import torch
from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassJaccardIndex
import click
from torchvision import transforms
from tqdm import tqdm

@click.command()
@click.option("--semantics_dir")
@click.option("--gt_dir")
@click.option("--img_size", type=int, default=512)
@click.option("--tag", default="")
@click.option("--center_crop", is_flag=True)
def main(semantics_dir, gt_dir, img_size, tag, center_crop):
    cwsa_dir = semantics_dir 

    ji = MulticlassJaccardIndex(num_classes=3, average=None)
    j_list = []

    for img_fn in tqdm(os.listdir(gt_dir)):
        gt_fp = os.path.join(gt_dir, img_fn)
        gt_pil = Image.open(gt_fp)

        if center_crop:
            gt_torch = transforms.functional.pil_to_tensor(gt_pil)
            gt_torch = transforms.Resize(img_size, transforms.InterpolationMode.NEAREST)(gt_torch)
            gt_torch = transforms.CenterCrop(img_size)(gt_torch)
            gt_np = gt_torch.cpu().numpy()[0]
        else:
            w,h = gt_pil.size
            new_size = min(w,h)
            gt_pil = gt_pil.crop((0,0,new_size, new_size))
            gt_pil = gt_pil.resize((img_size, img_size), resample = Image.NEAREST)
            gt_np = np.array(gt_pil)

        cwsa_fp = os.path.join(cwsa_dir, img_fn)
        cwsa_pil = Image.open(cwsa_fp)
        cwsa_np = np.array(cwsa_pil)

        cwsa_np[cwsa_np == 3] = 0  # map anomalies to background/soil
        gt_np[gt_np == 3] = 1  # partial crops
        gt_np[gt_np == 4] = 2  # partial weeds

        j_list.append(ji(torch.tensor(np.expand_dims(cwsa_np, -1), dtype=torch.uint8), torch.tensor(np.expand_dims(gt_np, -1), dtype=torch.uint8)))


    pp = np.stack(j_list).mean(axis=0) * 100
    print(pp[0], pp[1], pp[2], np.stack(j_list).mean()*100)

if __name__ == '__main__':
    main()
