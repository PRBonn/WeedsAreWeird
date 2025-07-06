#!/usr/bin/env python3

import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import click
import yaml
import shutil
import optuna
import concurrent.futures
import threading
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries

import lpips  # lpips==0.1.4

reward_lock = threading.Lock()

def get_diff_hsvspace(fake_pil, real_pil):
    fake_pil_hsv = fake_pil.convert("HSV")
    real_pil_hsv = real_pil.convert("HSV")

    real_pil_hsv = real_pil_hsv.resize(fake_pil_hsv.size, Image.BILINEAR)
    fake_np = np.array(fake_pil_hsv).astype(float)
    real_np = np.array(real_pil_hsv).astype(float)

    diff_np = (np.absolute(fake_np - real_np)).sum(axis=-1)

    return diff_np / (255. * 3)


def get_diff_rgbspace(fake_pil, real_pil):
    real_pil = real_pil.resize(fake_pil.size, Image.BILINEAR)
    fake_np = np.array(fake_pil).astype(float)
    real_np = np.array(real_pil).astype(float)

    diff_np = (np.absolute(fake_np - real_np)).sum(axis=-1)

    return diff_np / (255. * 3)  


def get_diff_huespace(fake_pil, real_pil):
    fake_pil_hsv = fake_pil.convert("HSV")
    real_pil_hsv = real_pil.convert("HSV")
    real_pil_hsv = real_pil_hsv.resize(fake_pil_hsv.size, Image.BILINEAR)

    fake_np_h = np.array(fake_pil_hsv)[:,:,0].astype(float)
    real_np_h = np.array(real_pil_hsv)[:,:,0].astype(float)

    diff_np = np.absolute(fake_np_h - real_np_h)
    return diff_np / 255.


def get_diff_lpips(fake_pil, real_pil):
    # from THOR code
    # TODO find a way to create the lpips obj and keep using the same one instead of reinitialising each time
    l_pips_sq = lpips.LPIPS(
                                pretrained=True,   # depricated
                                net='squeeze', 
                                use_dropout=True, 
                                eval_mode=True, 
                                spatial=True,
                                lpips=True
                            )
    fake_tensor = transforms.functional.pil_to_tensor(fake_pil) / 255.
    fake_tensor = fake_tensor.unsqueeze(0)
    real_tensor = transforms.functional.pil_to_tensor(real_pil) / 255.
    loss_lpips = l_pips_sq( 
                            fake_tensor,
                            real_tensor, 
                            normalize=True, 
                            retPerLayer=False
                          )
    loss_lpips =  loss_lpips[0][0].detach().cpu().numpy()
    return loss_lpips


def decompose(f_np):
    f_np = f_np.astype(float)
    R = f_np[:,:,0]
    G = f_np[:,:,1]
    B = f_np[:,:,2]
    RGB = R + G + B

    mew = RGB !=0
    r = np.divide(R, RGB,where=mew)
    g = np.divide(G, RGB,where=mew)
    b = np.divide(B, RGB,where=mew)
    r[~mew]=0
    g[~mew]=0
    b[~mew]=0
    return R, G, B, r, g, b, RGB

def vari(f_np, max_clip=1):
    R, G, B, r, g, b, RGB = decompose(f_np)
    denom = (g+r-b)
    numerator = g-r
    mask = denom!=0
    vm = np.divide(numerator, denom, where=mask)
    vm[~mask]=max_clip*np.sign(numerator)[~mask]
    return vm 

def exg_correct(f_np):
    R, G, B, r, g, b, RGB = decompose(f_np)
    vm = 2.*g - r - b
    return vm

def exr(f_np):
    R, G, B, r, g, b, RGB = decompose(f_np)
    vm = (1.4*r - g)
    if r.max() > 1.0:
        import pdb; pdb.set_trace()
    vm = -vm  # because this is for soil seg
    return vm

def exb(f_np):
    R, G, B, r, g, b, RGB = decompose(f_np)
    vm = (1.4*b-g)  
    vm = -vm  # because this is for soil seg
    return vm

def exgr(f_np):
    exg_ = exg_correct(f_np)
    exr_ = exr(f_np)  # alr neg
    vm = exg_ + exr_
    return vm

def grvi(f_np):
    R, G, B, r, g, b, RGB = decompose(f_np)
    denom = G+R
    denom[denom==0] = 1.
    vm = (G-R) / denom
    return vm

def mgrvi(f_np):
    R, G, B, r, g, b, RGB = decompose(f_np)
    G_sq = np.square(G)
    R_sq = np.square(R)

    denom = G_sq + R_sq
    denom[denom==0] = 1.
    vm = (G_sq - R_sq) / denom
    return vm

def gli(f_np, clip_val=10):
    R, G, B, r, g, b, RGB = decompose(f_np)
    denom = -r-b
    eps = 1. / (255*3)
    denom[denom==0]=eps
    vm = (2.*g-r-b) / denom
    vm[vm < -clip_val ]=-clip_val
    vm[vm > clip_val ]=clip_val
    vm = -vm  # because this is for soil seg
    return vm

def rgbvi(f_np):
    R, G, B, r, g, b, RGB = decompose(f_np)
    G_sq = np.square(G)
    denom = G_sq + B*R
    denom[denom==0]=1
    vm = (G_sq - B*R ) / denom
    return vm

def ikaw(f_np):
    R, G, B, r, g, b, RGB = decompose(f_np)
    denom = R+B
    denom[denom==0]=1
    vm = (R-B) / denom 
    return vm

def mean_vi10(f_np):
    vm_l = []
    for fn_name in exg10_dict:
        fn = exg10_dict[fn_name]
        vm = fn(f_np)
        vm = (vm - vm.min()) / (vm.max() - vm.min())
        vm_l.append(vm)

    vm_np = np.array(vm_l)
    vm = vm_np.mean(0)
    return vm 

def mean_vi10_top3(f_np):
    # hand scale it
    vm = exg_correct(f_np) / 2 
    vm = vm + exr(f_np) 
    vm = vm + (exgr(f_np) / 3)
    vm = vm + mgrvi(f_np)
    vm = vm + rgbvi(f_np)
    vm = vm / 5.0
    return vm 

def new_exgr(f_np):
    f_np = f_np.astype(float)
    exg = f_np[:,:,1]* 2 - f_np[:,:,0] - f_np[:,:,2]
    exr = f_np[:,:,0]* 1.4 - f_np[:,:,1] - f_np[:,:,2]
    vm = exg-exr
    return vm

def exg(f_np):
    f_np = f_np.astype(float)
    f_np = f_np[:,:,1]* 2 - f_np[:,:,0] - f_np[:,:,2]
    return f_np


def get_diff_exg(fake_pil, real_pil):
    real_pil = real_pil.resize(fake_pil.size, Image.BILINEAR)
    fake_np = np.array(fake_pil).astype(float)
    real_np = np.array(real_pil).astype(float)
    fake_np = exg(fake_np)
    real_np = exg(real_np)

    diff_np = np.abs(fake_np - real_np)

    diff_np = diff_np / (255. * 4)  
    return diff_np 


def get_diff_lpips_exg(fake_pil, real_pil):
    lpips_loss = get_diff_lpips(fake_pil, real_pil)
    hue_loss = get_diff_exg(fake_pil, real_pil)
    diff = lpips_loss * hue_loss
    return diff


def get_diff_lpips_hsv(fake_pil, real_pil):
    lpips_loss = get_diff_lpips(fake_pil, real_pil)
    hue_loss = get_diff_hsvspace(fake_pil, real_pil)
    diff = lpips_loss * hue_loss
    return diff


def get_diff_lpips_hue(fake_pil, real_pil):
    lpips_loss = get_diff_lpips(fake_pil, real_pil)
    hue_loss = get_diff_huespace(fake_pil, real_pil)
    diff = lpips_loss * hue_loss
    return diff


def get_diff_lpips_rgb(fake_pil, real_pil):
    lpips_loss = get_diff_lpips(fake_pil, real_pil)
    hue_loss = get_diff_rgbspace(fake_pil, real_pil)
    diff = lpips_loss * hue_loss
    return diff

def get_diff_vm(fake_pil, real_pil):
    real_pil = real_pil.resize(fake_pil.size, Image.BILINEAR)
    fake_np = np.array(fake_pil).astype(float)
    real_np = np.array(real_pil).astype(float)
    fake_np = exg(fake_np) / 255
    real_np = exg(real_np) / 255

    fake_vm = fake_np > 0.1
    real_vm = real_np > 0.1

    diff = np.logical_and(np.logical_not(fake_vm), real_vm)
    diff = diff.astype(np.uint8)

    return diff


def get_diff_ssim_exg(fake_pil, real_pil):
    real_pil = real_pil.resize(fake_pil.size, Image.BILINEAR)
    fake_np = np.array(fake_pil).astype(float)
    real_np = np.array(real_pil).astype(float)
    fake_exg = exg(fake_np) / 255 / 4.0 +0.5
    real_exg = exg(real_np) / 255 / 4.0 +0.5

    _, sim = ssim(fake_exg, real_exg, data_range=1.0, full=True)
    diff = 1.0 - sim

    return diff


def get_diff_ssim(fake_pil, real_pil):
    real_pil = real_pil.resize(fake_pil.size, Image.BILINEAR)
    fake_np = np.array(fake_pil).astype(float)
    real_np = np.array(real_pil).astype(float)
    fake_exg = (fake_np) / 255 
    real_exg = (real_np) / 255 

    _, sim = ssim(fake_exg, real_exg, data_range=1.0, full=True, win_size=3)
    diff = 1.0 - sim
    diff=diff.mean(axis=-1)


    return diff

def get_diff_felzenszwalb(fake_pil, real_pil):
    real_pil = real_pil.resize(fake_pil.size, Image.BILINEAR)
    fake_np = np.array(fake_pil).astype(float)
    real_np = np.array(real_pil).astype(float)

    scale=100
    sigma=0.5
    min_size=20
    real_fz = felzenszwalb(real_np, scale=scale, sigma=sigma, min_size=min_size)
    fake_fz = felzenszwalb(fake_np, scale=scale, sigma=sigma, min_size=min_size)
    diff = get_diff_rgbspace(fake_pil, real_pil)

    for real_seg_id in np.unique(real_fz):
        real_seg = real_fz == real_seg_id
        diff[real_seg] = np.median(diff[real_seg])

    return diff

exg10_dict={
        "exg_correct": exg_correct,
        "exr": exr,
        #"exb": exb,
        #"exgr": exgr,
        # "grvi": grvi,
        "mgrvi": mgrvi,
        #"gli": gli,
        "rgbvi": rgbvi,
        #"ikaw": ikaw,
        #"vari": vari,
        }

exg_fns_dict={
        "exg": exg,
        "exg_correct": exg_correct,
        "vari": vari,
        "exr": exr,
        "exb": exb,
        "exgr": exgr,
        "grvi": grvi,
        "mgrvi": mgrvi,
        "gli": gli,
        "rgbvi": rgbvi,
        "ikaw": ikaw,
        "mean_vi10": mean_vi10,
        "mean_vi10_top3": mean_vi10_top3,
        "new_exgr": new_exgr,
        }

diff_fns_dict={
        "get_diff_lpips": get_diff_lpips,
        "get_diff_exg": get_diff_exg,
        "get_diff_rgbspace": get_diff_rgbspace,
        "get_diff_huespace": get_diff_huespace,
        "get_diff_hsvspace": get_diff_hsvspace,
        "get_diff_lpips_exg": get_diff_lpips_exg,
        "get_diff_lpips_rgb": get_diff_lpips_rgb,
        "get_diff_lpips_hue": get_diff_lpips_hue,
        "get_diff_lpips_hsv": get_diff_lpips_hsv,
        "get_diff_vm": get_diff_vm,
        "get_diff_ssim_exg": get_diff_ssim_exg,
        "get_diff_ssim": get_diff_ssim,
        "get_diff_felzenszwalb": get_diff_felzenszwalb
        }
def blur(np_arr, kernel_size = 5):
    kernel = np.ones((kernel_size,kernel_size),np.float32)/(kernel_size * kernel_size)
    np_blur = cv2.filter2D(np_arr,-1,kernel)
    return np_blur


def threshold(np_arr, min_t=50):
    if min_t <= 0:
        bin_mask= np_arr
    else:
        bin_mask = np_arr >= min_t
    return bin_mask


def remove_noise(bin_mask, min_area=100):
    components = cv2.connectedComponentsWithStats(bin_mask.astype(np.uint8), cv2.CV_32S)
    retval, labels, stats, centroids = components

    bad_labels = np.where(stats[:, cv2.CC_STAT_AREA] < min_area)

    for bl in bad_labels[0]:  # im sure there is a better way to do this or?
        labels[labels==bl] = 0

    return labels > 0


def get_reward(ano_mask, weed_mask):
    """return reward
    which is modelled as the number of weed pixels counted as anomalies
    """
    intersection = np.logical_and(ano_mask, weed_mask).sum()
    if intersection == 0:
        local_reward = 0
    else:
        local_reward = intersection / (np.logical_or(ano_mask, weed_mask).sum())
    return local_reward


def one_img_functional(
                    gen_pil, 
                    real_pil, 
                    blur_k, 
                    thresh, 
                    noise_removal_thresh, 
                    get_diff_str=None,  # has priority
                    get_diff_fn=None,
                    thor_normalise=False
        ):

    if not get_diff_str is None:
        get_diff_fn = diff_fns_dict[get_diff_str]
        # Note to self: normalise for thresholding because we want the threshold to be between (0, 1)
    ano_mask = get_diff_fn(gen_pil, real_pil)

    if blur_k > 1:
        ano_mask = blur(ano_mask, blur_k)
    
    no_thres = np.copy(ano_mask)
    ano_mask = threshold(ano_mask, min_t=thresh)

    if noise_removal_thresh > 0:
        ano_mask = remove_noise(ano_mask, noise_removal_thresh)

    if thor_normalise:
        ano_mask = ano_mask / ano_mask.max()
        no_thres = no_thres / no_thres.max()
    return ano_mask, no_thres


def resize_img_linn(img_pil, des_size):
    img_w, img_h = img_pil.size
    img_minwh = min(img_w, img_h)
    if des_size < img_minwh:  # img is bigger than patch required
        img_pil = img_pil.crop((0,0,des_size, des_size))
    else:
        # img is too small in either one direction, need to resize
        img_pil = img_pil.resize((des_size, des_size), box=(0,0,img_minwh, img_minwh))

    return img_pil 

def one_img(
                img_name, 
                gens_dir, 
                real_dir, 
                blur_k, 
                thresh, 
                noise_removal_thresh, 
                semseg_dir, 
                anom_dir, 
                get_diff, 
                is_vis, 
                is_rewarded=True,
                thor_normalise=False
            ):
    gen_fp = os.path.join(gens_dir, img_name)
    gen_pil = Image.open(gen_fp)

    real_fp = os.path.join(real_dir, img_name)
    real_pil = Image.open(real_fp)

    w,h=real_pil.size
    new_size = min(w,h)
    real_pil = real_pil.crop((0,0,new_size, new_size))
    real_pil = real_pil.resize(gen_pil.size, resample=Image.BILINEAR)
    real_np = np.array(real_pil)
    real_pil = Image.fromarray(real_np[:,:,:3])

    ano_mask, diff = one_img_functional(
                    gen_pil, 
                    real_pil, 
                    blur_k, 
                    thresh, 
                    noise_removal_thresh, 
                    get_diff_fn=get_diff,
                    thor_normalise=thor_normalise
            )

    if is_rewarded:
        semseg_fp = os.path.join(semseg_dir, img_name)
        semseg_pil = Image.open(semseg_fp)
        semseg_pil = semseg_pil.resize(ano_mask.shape, Image.Resampling.NEAREST)
        semseg_np = np.asarray(semseg_pil) 

        weed_mask = np.logical_or(semseg_np == 2, semseg_np ==4)
        with reward_lock:
            global reward
            reward += get_reward(ano_mask, weed_mask)
    
        
    if is_vis:
        out_fp = os.path.join(anom_dir, img_name)
        ano_mask = ano_mask * 255
        mask_pil = Image.fromarray(ano_mask.astype(np.uint8))
        mask_pil.save(out_fp)

    return diff


def one_dir(
        thresh, 
        get_diff, 
        blur_k, 
        noise_removal_thresh,
        gens_dir = "", 
        real_dir = "", 
        semseg_dir = "", 
        anom_dir = ".",
        is_vis = False,
        num_threads=20
        ):
    global reward
    reward = 0
    if num_threads > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for img_name in os.listdir(gens_dir):
                futures=[executor.submit(one_img, img_name, gens_dir, real_dir, blur_k, thresh, noise_removal_thresh, semseg_dir, anom_dir, get_diff, is_vis)]
            concurrent.futures.wait(futures)
    else:
        for img_name in os.listdir(gens_dir):
            one_img(img_name, gens_dir, real_dir, blur_k, thresh, noise_removal_thresh, semseg_dir, anom_dir, get_diff, is_vis)
        


def objective(trial):
    thresh = trial.suggest_float("thresh", 0, 1)
    get_diff = get_diff_hsvspace  # TODO
    blur_k = -1 # trial.suggest_int("blur_k", 1, 21, step=2)
    noise_removal_t = 1 # trial.suggest_int("noise_removal_t", 1, 1000)
    one_dir(thresh, get_diff, blur_k, noise_removal_t)
    return reward

@click.command()
def main():
    exp_name = "waw"
    sampler = optuna.samplers.TPESampler(seed=0) 
    study = optuna.create_study(
            sampler = sampler,
            direction="maximize",
            storage="sqlite:///"+exp_name+".sqlite3",  # Specify the storage URL here.
            study_name=exp_name,
            load_if_exists=True
            )
    study.optimize(objective, n_trials=100)

if __name__ == '__main__':
    main()
