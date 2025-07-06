#!/usr/bin/env python3


import os
from PIL import Image 
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import click
import json
from torchvision import transforms
from util import resize_me

@click.command()
@click.option("--input_dir")
@click.option("--output_dir")
@click.option("--vm_dir")
@click.option("--ds_dir")  # for visualisation
@click.option("--thres", type=float, default=150)
@click.option("--img_size", type=int, default=1024)
@click.option("--center_crop", is_flag=True)
@click.option("--is_vis", is_flag=True)
def main(
        input_dir,
        output_dir,
        vm_dir, 
        ds_dir, 
        thres, 
        img_size, 
        center_crop,
        is_vis
        ):
    anom_dir = input_dir 
    out_dir =  os.path.join(output_dir, "cws")
    out_dir_viz =  os.path.join(output_dir, "cws_vis")
    img_dir = os.path.join(ds_dir, "images")  # for overlay in visualisation
    labels_dir = os.path.join(ds_dir, "semantics")
    gen_dir = os.path.join(output_dir, "gen")

    os.makedirs(out_dir, exist_ok=True)
    if is_vis:
        os.makedirs(out_dir_viz, exist_ok=True)

    # save configs
    cli_args_dict = {}
    cli_args_dict["input_dir"] =input_dir 
    cli_args_dict["output_dir"] =output_dir 
    cli_args_dict["vm_dir"] = vm_dir
    cli_args_dict["labels_dir"] = labels_dir
    cli_args_dict["thres"] = thres
    cli_args_dict["img_size"] = img_size
    with open(os.path.join(output_dir, 'cws_cli_args.json'), 'w') as fp:
        json.dump(cli_args_dict, fp)
    
    cmap_semseg = plt.cm.get_cmap('seismic', 4)

    for anom_fn in tqdm(os.listdir(anom_dir)):
        anom_fp = os.path.join(anom_dir, anom_fn)
        anom_pil = Image.open(anom_fp)
        anom_np = np.array(anom_pil)

        vm_fp = os.path.join(vm_dir, anom_fn)
        vm_pil = Image.open(vm_fp)
        # crop and resize accordingly
        """
        w,h = vm_pil.size
        new_size = min(w,h)
        vm_pil = vm_pil.crop((0,0,new_size,new_size))
        vm_pil = vm_pil.resize(anom_pil.size)
        """
        vm_np = resize_me(vm_pil, anom_pil.size[0], is_semseg=True)[:,:,0]


        # TODO im sure i can do something fancier in the future
        class_np = np.zeros(vm_np.shape, dtype=np.uint8)  # soil
        class_np[vm_np>0] = 1  # crop
        class_np[anom_np>thres] = 3  # anomalies 
        class_np[np.logical_and(vm_np>0,anom_np>thres)] = 2  # weeds

        # phenobench dont put anomalies in phenobench submission
        # class_np[class_np==3]=0

        out_img_fp = os.path.join(out_dir, anom_fn)
        cwsa_pil = Image.fromarray(class_np)
        # cwsa_pil = cwsa_pil.resize((1024, 1024), resample=Image.Resampling.NEAREST)  # phenobench resize to original image size for eval
        cwsa_pil.save(out_img_fp)

        if is_vis:
            # make overlays and put nice colors
            fig, axs = plt.subplots(2, 3)

            img_pil = Image.open(os.path.join(img_dir,anom_fn))
            axs[0,0].imshow(img_pil)
            img_np = np.array(img_pil).copy()
            if center_crop:
                img_torch = transforms.functional.pil_to_tensor(img_pil)
                img_torch = transforms.Resize(img_size)(img_torch)
                img_torch = transforms.CenterCrop(img_size)(img_torch)
                img_np = img_torch.cpu().numpy()
                img_np = np.moveaxis(img_np, 0, -1)
            else:
                w,h = img_pil.size
                new_size = min(w,h)
                img_pil = img_pil.crop((0,0,new_size, new_size))
                img_pil = img_pil.resize((img_size, img_size), resample=Image.NEAREST)
                img_np = np.array(img_pil)
                
            axs[0,2].imshow(np.array(Image.open(os.path.join(anom_dir,anom_fn))))
     
            try:
                axs[1,0].imshow(np.array(Image.open(os.path.join(gen_dir,anom_fn))))
            except:
                axs[1,0].imshow(img_np)

            sem_map2 = axs[1,1].imshow(
                    class_np, 
                    cmap=cmap_semseg,
                    vmin=0, 
                    vmax=3, 
                    interpolation="none"
                    )
            ax = axs[1,1]
            cax2 = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.01,ax.get_position().height])
            fig.colorbar(sem_map2, cax=cax2)
            axs[1,2].imshow(img_np)
            sem_map = axs[1,2].imshow(
                    class_np, 
                    alpha=0.5,
                    cmap=cmap_semseg,
                    vmin=0, 
                    vmax=3, 
                    interpolation="none"
                    )
            ax = axs[1,2]
            cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.01,ax.get_position().height])
            fig.colorbar(sem_map, cax=cax)

            label = Image.open(os.path.join(labels_dir,anom_fn))

            if not center_crop:  # top left
                w,h = label.size
                new_size = min(w,h)
                label = label.crop((0,0,new_size, new_size))
                label = label.resize((img_size, img_size), resample=Image.NEAREST)
            else:  # center crop
                img_torch = transforms.functional.pil_to_tensor(label)
                img_torch = transforms.Resize(img_size, transforms.InterpolationMode.NEAREST)(img_torch)
                img_torch = transforms.CenterCrop(img_size)(img_torch)
                label_np = img_torch.cpu().numpy()[0]

            label_np = np.array(label)
            label_np[label_np == 4] = 2
            label_np[label_np == 3] = 1
            fml = axs[0,1].imshow(
                                    label_np, 
                                    cmap=cmap_semseg, 
                                    vmin=0, 
                                    vmax=3, 
                                    interpolation="none"
                                    )
            ax = axs[0,1]
            cax3 = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
            fig.colorbar(fml, cax=cax3)
    
            plt.savefig(os.path.join(out_dir_viz, anom_fn.split(".")[0]+"_vis.png"), dpi = 800)
            plt.close()
    

if __name__ == '__main__':
    main()
