#!/usr/bin/env python3


import os
from PIL import Image 
import numpy as np
import cv2
from tqdm import tqdm


if __name__ == '__main__':
    seed=0
    # torch.manual_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.cuda.manual_seed_all(seed)

    phenobench_imgs_dir = "/mnt/tmp/waa_experiments_datasets/crop_and_weed_maize/val/images"
    out_dir = "/mnt/tmp/waa_experiments_datasets/crop_and_weed_maize/val/exg_vm"
    exg_thresh = 0.1  # [-2., 2.]
    kernel_size = 5
    os.makedirs(out_dir)

    for img_fn in tqdm(os.listdir(phenobench_imgs_dir)):
        img_fp = os.path.join(phenobench_imgs_dir, img_fn)
        img_pil = Image.open(img_fp)
        
        img_np = np.array(img_pil)
        r = img_np[:,:,0].astype(np.float32)
        g = img_np[:,:,1].astype(np.float32)
        b = img_np[:,:,2].astype(np.float32)
        exg = (g*2. - r - b ) / 255.

        # making things look better  
        kernel = np.ones((kernel_size,kernel_size),np.float32)/(kernel_size * kernel_size)
        exg = cv2.filter2D(exg,-1,kernel)

        vm = exg > exg_thresh

        out_img_fp = os.path.join(out_dir, img_fn)
        vm_pil = Image.fromarray(vm)
        # vm_pil = vm_pil.resize((512, 512), resample = Image.NEAREST)
        vm_pil.save(out_img_fp)

