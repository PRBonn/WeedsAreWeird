"""Script to extract cropped instances from phenobench labels
"""

import torch
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
import click
import gc


from ipb_loaders.rgb.augmentation import AugmentedIPBloader
from ipb_loaders.rgb.phenorob import PhenoRob
from get_anomaly_masks_optuna import exg
from sam_py import SAM_bbs


def make_square(bb, img_size=1024, min_bb_size=16, rect_sq=50):
    min_x, min_y, max_x, max_y = bb
    w = max_x - min_x
    h = max_y - min_y
    size = max(w, h)
    size = round(size / 2) * 2

    size = max(size, min_bb_size)
    if size > img_size:
        raise ValueError("bb size is bigger than image size!")

    cx = round((max_x + min_x) / 2)
    cy = round((max_y + min_y) / 2)

    dx = max(size / 2 - cx, 0)  # if the bb is too left
    dx = dx - max(cx + size / 2 - img_size, 0)  # if the bb is too right
    # cx = cx + dx

    dy = max(size / 2 - cy, 0)  # if the bb is too high
    dy = dy - max(cy + size / 2 - img_size, 0)  # if the bb is too low
    # cy = cy + dy

    if dx != 0 or dy != 0:
        if min(w, h) < min_bb_size:
            # TODO if bb is smaller than smallest size, still expand it
            new_w = max(w, min_bb_size)
            new_h = max(h, min_bb_size)
            new_bb = np.array(
                [cx - new_w / 2, cy - new_h / 2, cx + new_w / 2, cy + new_h / 2]
            )
        else:
            new_bb = np.array(bb)
    else:
        new_bb = np.array([cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2])

    new_bb[new_bb > img_size] = img_size
    new_bb[new_bb < 0] = 0

    return new_bb


def get_bb_from_cc(stats, min_area=100, min_size=16, max_size=512, is_inference=False):
    bbs = []
    for stat in stats:
        if stat[-1] > min_area:
            x, y, w, h, a = stat
            if w == 1024 and h == 1024:
                continue

            if w < min_size or h < min_size:
                cx = x + w / 2
                cy = y + h / 2
                w = max(w, min_size)
                h = max(h, min_size)

                # oob handling
                w = min(1024 - x, w)
                h = min(1024 - y, h)
                x = max(0, cx - w / 2)
                y = max(0, cy - h / 2)

            if (not is_inference) and (w > max_size or h > max_size):
                # TODO make this better
                # split in half
                w1 = min(w, max_size)
                w2 = max(0, w - max_size)

                h1 = min(h, max_size)
                h2 = max(0, h - max_size)

                def bb(x, y, w, h):
                    bb = [x, y, x + w, y + h]
                    if w > 0 and h > 0:
                        bbs.append(bb)

                bb(x, y, w1, h1)
                bb(x, y + h1, w1, h2)
                bb(x + w1, y, w2, h1)
                bb(x + w1, y + h1, w2, h2)

            else:
                min_x = x
                max_x = x + w
                min_y = y
                max_y = y + h

                bbs.append([min_x, min_y, max_x, max_y])
    return np.array(bbs)


def get_bb_connectedcomp_one(img, exg_thresh=0.1, is_inference=False):
    vm_mask = (exg(img) / 255) > exg_thresh
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        vm_mask.astype(np.uint8)
    )
    bb = get_bb_from_cc(stats, is_inference=is_inference)
    return bb


def get_bb_connectedcomp(batch):
    bbs = []
    bbs_class = []
    for img in batch["image"]:
        img = img / 2.0 + 0.5
        img_pil = transforms.functional.to_pil_image(img)
        img_np = np.array(img_pil)
        bb = get_bb_connectedcomp_one(img_np)
        bbs.append(bb)
        bbs_class.append(np.ones(bb.shape[0]) * 3)
    return bbs, bbs_class


@click.command()
@click.option("--output_dir")  # output parent dir
@click.option("--input_dir")  # dataset parent dir
@click.option("--vis_dir", default="")
@click.option("--vm_dir", default="")
@click.option("--aug_cfg", default="./cfgs/augs_clip.cfg")
@click.option("--split", default="train")
@click.option("--points_per_side", default=48)
@click.option("--is_remove_overlap", is_flag=True)
def main(output_dir, input_dir,vis_dir,vm_dir,aug_cfg,split,points_per_side, is_remove_overlap):
    min_size = 16  # px size in each dim for bb -- bioclip vit patch size
    is_save = True
    os.makedirs(output_dir, exist_ok=True)
    is_labels = False  # true if you want to use the labels to make the bb, otherwise we use connected comp in exg space
    # is_labels = True

    is_getclip_pred = True
    if vis_dir == "":
        is_vis=False
    else:
        is_vis = True
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(vm_dir, exist_ok=True)

    # clip_thresh = 0.233
    clip_text = ""

    if is_labels:
        keys = ["images", "semantics", "plant_instances"]
    else:
        keys = ["images"]
    train_dataset = AugmentedIPBloader(
        PhenoRob(data_source=input_dir, split=split, keys=keys),
        aug_cfg,
        is_semantics=is_labels,
        is_bb=is_labels,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=train_dataset.collate_fn,
    )

    device = "cuda"
    img_size = int(train_dataset.aug_dict["resize"])

    # is_filter = False # FIXME for ablation
    is_filter=True

    # use_grid=True  # FIXME for ablation
    use_grid=False

    sam_obj = SAM_bbs(
            device, 
            img_size=img_size,
            points_per_batch=100,
            exg_thresh=0.0,
            exg_percent=0.2,
            is_multimask=False,
            is_nms=True,
            nms_iou=0.7,
            max_seg_size=1000,
            seed_exg_thresh=0.1,
            points_per_side=points_per_side,
            min_mask_region_area=100,
            concomp_k_size=-1,
            is_remove_overlap=is_remove_overlap, 
            filter_vm_ksize=5,
            use_grid=use_grid
            ) # TODO pass img size from aug too

    for batch in tqdm(train_dataloader):
        if is_labels:
            bbs = batch["boxes"]
            bbs_class = batch["boxes_class"]
        else:
            bbs_l, bbs_class, seeds_l, masks_l, img_exg_l = sam_obj.get_bbs_frombatch(batch, return_seeds=True, is_filter=is_filter)
            if is_vis: 
                for img_torch, seeds, img_fn , bbs, masks, img_exg in zip(batch["image"], seeds_l, batch["filename"], bbs_l, masks_l, img_exg_l):
                    img_torch = (img_torch + 1) / 2.
                    img_pil = transforms.ToPILImage()(img_torch)

                    my_dpi = 80
                    plt.figure(figsize=(1024/my_dpi, 1024/my_dpi), dpi=my_dpi)
                    plt.gca().set_axis_off()

                    # draw points
                    plt.imshow(img_pil)
                    plt.savefig(os.path.join(vis_dir,img_fn+"_img.png"),bbox_inches = 'tight', pad_inches = 0, dpi=my_dpi)

                    sam_obj.show_points(seeds, plt.gca())
                    plt.savefig(os.path.join(vis_dir,img_fn+"_spots.png"),bbox_inches = 'tight', pad_inches = 0, dpi=my_dpi)

                    # draw segs
                    plt.clf()
                    plt.close()
                    plt.figure(figsize=(1024/my_dpi, 1024/my_dpi), dpi=my_dpi)
                    plt.gca().set_axis_off()

                    plt.imshow(img_pil)
                    img_np = np.array(img_pil)
                    sam_obj.show_anns(masks, img_np)
                    plt.savefig(os.path.join(vis_dir,img_fn+"_segs.png"),bbox_inches = 'tight', pad_inches = 0, dpi=my_dpi)
                    sam_obj.vis_bbs(bbs, plt.gca(), clr="pink")
                    plt.savefig(os.path.join(vis_dir,img_fn+"_bbs.png"),bbox_inches = 'tight', pad_inches = 0, dpi=my_dpi)
                    plt.clf()
                    plt.close()

                    # save the vm
                    vm = sam_obj.get_vm(masks)
                    vm = vm*255
                    Image.fromarray(vm.astype(np.uint8)).save(os.path.join(vm_dir, img_fn))

                    # vm used for seeding
                    img_exg[img_exg < 0]=0
                    img_exg = img_exg * 255
                    Image.fromarray(img_exg.astype(np.uint8)).save(os.path.join(vis_dir, img_fn+"_filtervm.png"))

        fns = batch["filename"]
        for i, fn in enumerate(fns):  # TODO maybe zip instead of enumerate
            img = train_dataset.detransform(batch["image"][i])
            fn_base = fn.split(".")[0]
            for bb, bb_cls in zip(bbs_l[i], bbs_class[i]):
                if bb_cls == 0:  # skip if its soil
                    continue
                bb = make_square(bb)
                cropped = img.crop(bb.tolist())

                if cropped.size[0] < min_size or cropped.size[1] < min_size:
                    continue
                if is_save:
                    cropped.save(
                            os.path.join(output_dir, f"{fn_base}_{bb}_{bb_cls}.png")
                            )
                    gc.collect()
            torch.cuda.empty_cache()  # Frees cached memory on GPU


if __name__ == "__main__":
    main()
