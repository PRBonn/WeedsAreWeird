from torchvision import transforms
from PIL import Image
import numpy as np


def resize_me(gt_pil, img_size=1024, is_semseg=False):
    gt_torch = transforms.functional.pil_to_tensor(gt_pil)
    if is_semseg:
        gt_torch = transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST)(gt_torch)
    else:
        gt_torch = transforms.Resize(img_size)(gt_torch)
    gt_torch = transforms.CenterCrop(img_size)(gt_torch)
    gt_np = gt_torch.cpu().numpy()
    gt_np = np.moveaxis(gt_np, 0, -1)
    if is_semseg:
        return gt_np
    else:
        gt_pil = Image.fromarray(gt_np)
        return gt_pil


