
import os
import random
import shutil
import math
import click
import yaml

import matplotlib.pyplot as plt
import open_clip
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.stats import norm
from scipy.optimize import linear_sum_assignment
from datasets import Dataset, load_dataset
from datasets import Image as Image_HF
from torch.utils.data import DataLoader
import time
from scipy.ndimage import gaussian_filter1d


def pre_preprocess_val(pil, preprocess_from_clip):
    """returns a list of square patches so that all of the image is seen
    """
    w, h = pil.size
    size = min(w,h)
    list_patches = []
    for i in range(0,  w, size):
        if w < i + size: # not enough w to make a patch
            continue
        for j in range(0, h, size):
            if h < j + size: # not enough h to make a patch
                continue
            box = (i, j, i + size, j + size)
            patch =pil.crop(box)
            patch_tensor = preprocess_from_clip(patch)
            list_patches.append(patch_tensor)

    return torch.stack(list_patches)

def pre_preprocess_train(pil):
    # the preprocessing of bioclip crops everything to a square,
    # this only gives one patch (in the top left) 
    # so its only useful for when building the bag of features 
    w, h = pil.size
    size = min(w,h)
    pil = pil.crop((0,0,size,size))
    return pil


def get_sizes(
        batch,
        start=0,
        step=50,
        stop=200,
        ):
    vect = range(start, stop, step)
    size_onehot_list = []
    for i in range(len(batch[0])):  # assuming that they are square we just take one of the dim
        size_onehot = torch.zeros(len(vect)+1)  # last col for over stop, technically only needed if stop is nondivisible by step
        if batch[0][i] != batch[1][i]:  # rectangles
            index = -1
        else:
            size = batch[0][i]
            if size > stop:
                index = -2
            else:
                index = math.floor((size - start) / step)
        size_onehot[index]=1
        aaa=gaussian_filter1d(size_onehot, 0.7)
        size_onehot = aaa/aaa.max()
        size_onehot = torch.from_numpy(size_onehot)
        size_onehot_list.append(size_onehot)
    size_onehot_tensor = torch.stack(size_onehot_list, dim=0)
    return size_onehot_tensor

def get_feats(
        fn_list, 
        img_dir, 
        model, 
        preprocesser, 
        pre_preprocesser=pre_preprocess_train,
        batch_size = 1
        ):
    crop_features_list = []
    size_list = []
    device = (next(model.parameters()).device)

    start = time.time()
    if True:  # TODO
        startprep = time.time()
        fp_list = [ os.path.join(img_dir, fn) for fn in fn_list ]
        ds = Dataset.from_dict({"image": fp_list}).cast_column("image", Image_HF())

        def wrapper(batch):
            batch['size'] = [ image.size for image in batch['image']]
            batch['image'] = [preprocesser(image) for image in batch['image']]
            return batch
        ds.set_transform(wrapper)

        dataloader = DataLoader(
                ds, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=min(batch_size,2), 
                )

        for batch_item in dataloader:
            batch = batch_item["image"].to(device, non_blocking=True)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(batch)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            crop_features_list.append(image_features)
            size_list.append(get_sizes(batch_item['size']))
        crop_tensor = torch.cat(crop_features_list, dim=0)
        size_tensor = torch.cat(size_list, dim=0)
    else:
        for crop_fn in fn_list:
            crop_pil = Image.open(os.path.join(img_dir, crop_fn))
            crop_pil = pre_preprocesser(crop_pil)
            crop_image = preprocesser(crop_pil).unsqueeze(0)
            crop_image = crop_image.to(device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(crop_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            crop_features_list.append(image_features)
        crop_tensor = torch.stack(crop_features_list).squeeze(1)
        crop_tensor = crop_tensor.to(device)
    return crop_tensor, size_tensor

def one_iteration(
        it, 
        model_str, 
        img_dir, 
        crop_dir, 
        num_of_candidates, 
        num_feats=100, 
        min_patch_size=50,
        is_size=True
        ):

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_str)
    tokenizer = open_clip.get_tokenizer(model_str)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=model.to(device)

    """
    word_list =[] 
    word_list.append("Plantae Tracheophyta Betsetopsida Caryophyllales Amaranthaceae Beta vulgaris")
    text = tokenizer(word_list)
    text=text.to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    """

    list_of_imgs_tmp = os.listdir(img_dir)
    list_of_imgs = os.listdir(img_dir)

    for fn in list_of_imgs_tmp:
        try:
            img_pil = Image.open(os.path.join(img_dir, fn))
        except:
            print("skipping: cant open ", fn)
            list_of_imgs.remove(fn)
        w,h = img_pil.size
        if w != h:
            # list_of_imgs.remove(fn)
            pass
        else:
            if w < min_patch_size:
                list_of_imgs.remove(fn)

    print("num of patches left:", len(list_of_imgs))

    for iteration in tqdm(range(num_feats)):
        population_subsample = random.sample(list_of_imgs,min(500,len(list_of_imgs)))  # subsample
        candidates = random.sample(list_of_imgs,num_of_candidates)  # subsample


        population_feats, population_sizes = get_feats(population_subsample,img_dir, model, preprocess_val, batch_size=20)

        candidate_feats, candidate_sizes = get_feats(candidates, img_dir, model, preprocess_val, batch_size=20)


        feats = candidate_feats @ population_feats.T

        valid_votes = candidate_sizes @ population_sizes.T  # only can vote if sizes are similar
        valid_votes = valid_votes.to(feats.device)
        if is_size:
            feats = feats * valid_votes
        votes, votes_arg = feats.max(0)
        score = torch.zeros(feats.shape, device=device, dtype=feats.dtype)
        mews = votes_arg, torch.arange(feats.shape[1])
        score[mews] = feats[mews]
        score = score.sum(1)

        crop_arg = score.argmax()
        crop_fn = candidates[crop_arg]
        crop_patch = Image.open(os.path.join(img_dir, crop_fn))
        crop_patch = pre_preprocess_train(crop_patch)
        crop_patch.save(os.path.join(crop_dir,crop_fn))
        list_of_imgs.remove(crop_fn)

    crop_feats, crop_sizes = get_feats(os.listdir(crop_dir), crop_dir, model, preprocess_val)
    for iteration in tqdm(range(num_feats)):
        candidates = random.sample(list_of_imgs,min(500,len(list_of_imgs)))  # subsample
        candidate_feats , candidate_sizes= get_feats(candidates, img_dir, model, preprocess_val)

        feats = candidate_feats @ crop_feats.T
        valid_votes = candidate_sizes @ crop_sizes.T  # only can vote if sizes are similar
        valid_votes = valid_votes.to(feats.device)
        score, _ = torch.topk(feats,round(num_of_candidates/2)) #, largest=False)
        score = score.sum(1)

def get_scores(
        tag, 
        model_str, 
        img_dir, 
        crops_feats_fp, 
        weeds_feats_fp,
        corr_dir,
        wrong_dir
        ):

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_str)
    tokenizer = open_clip.get_tokenizer(model_str)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=model.to(device)
    fns = os.listdir(img_dir)
    fns =random.sample(fns, 1000)

    crop_feats , crop_sizes = get_feats(os.listdir(crops_feats_fp), crops_feats_fp, model, preprocess_val)
    weed_feats , weed_sizes = get_feats(os.listdir(weeds_feats_fp), weeds_feats_fp, model, preprocess_val)

    crop_count = 0
    weed_count = 0


    for fn in tqdm(fns):
        image_pil = Image.open(os.path.join(img_dir, fn))
        image_patches = pre_preprocess_val(image_pil, preprocess_val)
        image_patches = image_patches.to(device)
        base_fn = "".join(fn.split(".")[:-1])

        
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image_patches)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            crop_sim = image_features @ crop_feats.T
            crop_score = crop_sim.sum()

            weed_sim = image_features @ weed_feats.T
            weed_score = weed_sim.sum()

            fn = f"{base_fn}_{crop_score:.2f}_{weed_score:.2f}.png"
            is_crop = crop_score > weed_score


            if is_crop:
                crop_count +=1
                if tag == "crops":  # meh
                    out_dir = corr_dir
                else:
                    out_dir = wrong_dir
            else:
                weed_count +=1
                if tag == "crops":
                    out_dir = wrong_dir
                else:
                    out_dir = corr_dir

            image_pil.save(os.path.join(out_dir,fn))

    print('crop: ', crop_count)
    print('weed: ', weed_count)
    crop_count = float(crop_count)
    weed_count = float(weed_count)
    print('tag', tag)
    if tag == "crops":
        print('crop_score', crop_count/(crop_count+weed_count) * 100)
    else:
        print('weed_score', weed_count/(crop_count+weed_count) * 100)


@click.command()
@click.option("--yaml_cfg")
@click.option("--output_dir")
def main(yaml_cfg, output_dir):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    with open(yaml_cfg, 'r') as file:
        cfgs_dict = yaml.safe_load(file)

    img_dir = cfgs_dict["img_dir"]
    num_of_candidates= int(cfgs_dict["num_of_candidates"])
    num_feats = int(cfgs_dict["num_feats"])
    min_patch_size = int(cfgs_dict["min_patch_size"])
    is_size = bool(cfgs_dict["use_size_matching"])

    model_str = 'hf-hub:imageomics/bioclip'
    # model_str = "hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K"  # ablations

    crop_dir = output_dir
    os.makedirs(crop_dir, exist_ok=True)

    for i in range(1):
        one_iteration(i, model_str, img_dir, crop_dir, num_of_candidates, num_feats=num_feats, min_patch_size=min_patch_size, is_size=is_size)


if __name__ == '__main__':
    main()
