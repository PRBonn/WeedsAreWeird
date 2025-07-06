#!/usr/bin/env python3


import os
from PIL import Image 
import numpy as np


if __name__ == '__main__':
    seed=0
    # torch.manual_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.cuda.manual_seed_all(seed)

    phenobench_semseg_labels_dir = "/mnt/mon13/phenobench/imgs/dataset_exposed/PhenoBench/train/semantics"
    out_dir = "/mnt/mon13/phenobench/vm_gt_phenobenchtrain"

    for label_fn in os.listdir(phenobench_semseg_labels_dir):
        label_fp = os.path.join(phenobench_semseg_labels_dir, label_fn)
        semseg = Image.open(label_fp)
        vm = np.array(semseg).astype(np.uint8)
        vm[vm != 0] = 255
        out_label_fp = os.path.join(out_dir, label_fn)
        vm_pil = Image.fromarray(vm)
        vm_pil = vm_pil.resize((512, 512), resample = Image.NEAREST)
        vm_pil.save(out_label_fp)

