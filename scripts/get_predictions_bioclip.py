"""Script to use bioclip to get pseudolabels
"""

import os

from PIL import Image, ImageDraw
import open_clip
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import yaml
import click
import matplotlib.pyplot as plt
import cv2

from bioclip_popularity import pre_preprocess_val, get_feats, get_sizes
from get_bb_clips import make_square, get_bb_connectedcomp_one
from get_anomaly_masks_optuna import exg
from sam_py import SAM_bbs


import matplotlib.pyplot as plt

class ClipBBClassifier():
    def __init__(
            self, 
            model_str, 
            bb_type,  # "label" or "connectedcomp" or "sam"
            crop_feats_dir,
            bb_instance_labels_dir=None,  # only used if bb_type == "labels"
            min_bb_area=16, 
            img_size=1024,
            NN_k=3,  # number of neighbours for hypersphere
            voting_topk=10,
            use_size_votes=True,
            use_text=False,
            use_hypersphere=True,
            ):
        self.use_hypersphere=use_hypersphere
        self.model, _ , self.preprocess_clip = open_clip.create_model_and_transforms(model_str)
        tokenizer = open_clip.get_tokenizer(model_str)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # TODO scale at some point
        self.model=self.model.to(self.device)

        self.use_text = use_text
        if self.use_text:
            self.word_list =[]
            self.word_list.append("Plantae Tracheophyta Betsetopsida Caryophyllales Amaranthaceae Beta vulgaris")
            self.word_list.append("sugar beet")
            self.word_list.append("beet root")
            text = tokenizer(self.word_list)
            text=text.to(self.device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                self.text_features = self.model.encode_text(text)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        self.crop_feats, self.crop_sizes = get_feats(
                                os.listdir(crop_feats_dir), 
                                crop_feats_dir, 
                                self.model, 
                                self.preprocess_clip,  
                                pre_preprocesser=pre_preprocess_val,
                                batch_size = 20 # TODO scale at some point
                                )
        self.NN_k = NN_k
        self.use_size_votes=use_size_votes
        valid = self.crop_sizes @ (self.crop_sizes).T
        self.crop_hypersphere = self.hypersphere(self.crop_feats, self.NN_k, valid)
        self.bb_type = bb_type
        if self.bb_type == "sam":
            self.sam_obj = SAM_bbs(
                    self.device,
                    img_size=img_size,
                    points_per_batch=100,
                    exg_thresh=0.0,
                    exg_percent=0.2,
                    is_multimask=False,
                    is_nms=True,
                    nms_iou=0.7,
                    max_seg_size=1000,
                    seed_exg_thresh=0.1,
                    points_per_side=48,
                    min_mask_region_area=100,
                    concomp_k_size=-1,
                    is_remove_overlap=False,
                    filter_vm_ksize=5
                    )

        self.bb_instance_labels_dir = bb_instance_labels_dir
        self.min_bb_area = min_bb_area
        self.img_size = img_size
        self.voting_topk=voting_topk

    def l2_dist(self, a, b, valid, eps=1, scale=1):
        loss = torch.nn.MSELoss(reduction="none")
        dist = loss(a,b)
        """
        valid = valid.max() - valid
        valid = (valid * scale) + eps
        valid = valid.reshape(valid.shape[0] * valid.shape[1], 1).to(dist.device)
        dist = dist * valid
        """
        dist = dist.mean(1)
        return dist 

    def hypersphere(self, feats, NN_k, valid):
        times = feats.shape[0]
        full_repeat = feats.repeat(times,1)
        interleaved_repeat = feats.repeat_interleave(times,dim=0)
        dist = self.l2_dist(full_repeat, interleaved_repeat, valid)
        dist = dist.reshape(times,times,-1)
        lowest_k , _= torch.topk(
                dist,dim=1,
                k=(NN_k+1),
                largest=False, 
                sorted=True
                )
        l2_dist_1d = lowest_k[:,-1].squeeze(-1)
        max_dist = l2_dist_1d.mean() + l2_dist_1d.std()*1.

        return l2_dist_1d

    def get_semseg(self,img_fp, is_vm=False):
        img_fn = os.path.basename(img_fp)
        segments=None
        if self.bb_type == "label":
            bb = self.get_bb_labels(os.path.join(self.bb_instance_labels_dir, img_fn))
        elif self.bb_type == "connectedcomp":
            bb = get_bb_cc(img_fp)
        elif self.bb_type =="sam":
            bb, segments = self.get_bb_sam(img_fp)
        else:
            raise ValueError("Invalid bb_type: choose bb type: 'label' or 'connectedcomp'" )
        img_pil = Image.open(img_fp)

        # RESIZE
        gt_torch = transforms.functional.pil_to_tensor(img_pil)
        gt_torch = transforms.Resize(self.img_size, transforms.InterpolationMode.NEAREST)(gt_torch)
        gt_torch = transforms.CenterCrop(self.img_size)(gt_torch)
        img_np = gt_torch.cpu().numpy()
        img_np = np.moveaxis(img_np, 0, -1)
        img_pil = Image.fromarray(img_np.astype(np.uint8))

        semseg_pil = self.bb2semseg(bb, img_pil, segments)
        if is_vm:
            vm = exg(np.array(img_pil)) / 255 
            vm = vm > 0.1
            semseg_np = np.array(semseg_pil) * vm
            semseg_np[np.logical_and(vm, semseg_np==0)] = 1 
            semseg_pil = Image.fromarray(semseg_np)
        return semseg_pil, bb, segments

    def bb2semseg(self, bbs, img_pil, segments=None):
        semseg_np = np.ones((self.img_size, self.img_size), dtype=np.uint8) * 255
        counter = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        for i, bb in enumerate(bbs):
            sq_bb = make_square(bb.tolist(), min_bb_size=self.min_bb_area)
            patch = img_pil.crop(sq_bb.tolist())
            class_id = self.get_class(patch)

            # drop oob plants
            if (sq_bb ==0).any() or (sq_bb ==(self.img_size-1)).any():
                continue

            color = class_id
            if segments is None:
                drawer.rectangle(bb.tolist(), fill=color) 
            else:
                seg = segments[i]["segmentation"].cpu().numpy()
                update = semseg_np > class_id  # take the lower value 
                update = np.logical_and(seg, update)
                semseg_np[update]=class_id
                counter[seg] = counter[seg] + 1
        # average over overlapping segments
        for segment in segments: 
            seg = segment["segmentation"].cpu().numpy()
            (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(
                    seg.astype(np.uint8),8)
            for i in np.unique(labels):
                if i == 0:
                    continue  # rm bg
                segseg = labels == i
                semseg_np[segseg]=semseg_np[segseg].mean()
        semseg_np[counter==0]=0
        semseg_pil = Image.fromarray(semseg_np)
        return semseg_pil

    def in_hypersphere(self, image_features, feats, hypersphere, valids):
        times = feats.shape[0]
        img_feats_repeated = image_features.repeat_interleave(times,dim=0)
        assert image_features.shape[0] == 1  # TODO this would brek if we do batches

        dist = self.l2_dist(img_feats_repeated, feats, valids)
        is_in = (dist < hypersphere).any()

        return is_in

    def get_class(self, bb_pil): # TODO scale at some point
        image_patches = self.preprocess_clip(bb_pil).unsqueeze(0)
        image_patches = image_patches.to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image_patches)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            img_sizes = get_sizes([[bb_pil.size[0]], [bb_pil.size[1]]])
            img_sizes = img_sizes.to(self.device)

            if self.use_hypersphere:
                img_mat = img_sizes @ (self.crop_sizes.to(self.device)).T 
                is_crop = self.in_hypersphere(image_features, self.crop_feats, self.crop_hypersphere, img_mat)

            crop_sim = image_features @ (self.crop_feats).T
            if self.use_text:
                text_sim = image_features @ (self.text_features).T
        
        img_mat = img_sizes.cpu() @ (self.crop_sizes.cpu()).T 
        crop_sim = crop_sim.cpu().to(float)
        if self.use_size_votes:
            crop_sim = crop_sim * img_mat
       
        class_id = crop_sim.topk(self.voting_topk)[0].mean().item()
        if self.use_text:
            text_sim = text_sim.mean().cpu().to(float)
            class_id = class_id + text_sim
            # class_id = text_sim * 10

        class_id = 1 - class_id   # anomaly instead of similiarity
        class_id = max(0., class_id)
        class_id = min(1., class_id)
        class_id = int(class_id * 255)
 
        if self.use_hypersphere:
            if is_crop:
                class_id = 50
            else:
                class_id = 255

        return class_id  # 1:crop; 2:weed


    def get_bb_labels(self, instance_fp):
        masks=Image.open(instance_fp).convert("I")  # TODO
        masks = transforms.functional.pil_to_tensor(masks).to(torch.long)
        masks = F.one_hot(masks)
        masks = masks[0].permute(2,0,1)
    
        n = masks.shape[0]
        bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)
    
        # TODO remove bbs that are soil
    
        for index, mask in enumerate(masks):
            y, x = torch.where(mask != 0)
            if len(x) > 0:
                bounding_boxes[index, 0] = torch.min(x)
                bounding_boxes[index, 1] = torch.min(y)
                bounding_boxes[index, 2] = torch.max(x)
                bounding_boxes[index, 3] = torch.max(y)
    
    
        bounding_boxes_area = bounding_boxes.sum(dim=1)
        bounding_boxes = bounding_boxes[~(bounding_boxes_area == 0)]
    
        new_bbs = []
        for bb in bounding_boxes[1:]:  # first one is soil bg
            new_bbs.append(bb)
    
        return new_bbs

    def get_bb_sam(self,img_fn):
        img_pil= Image.open(img_fn)  
        # centercrop
        gt_torch = transforms.functional.pil_to_tensor(img_pil) 
        gt_torch = transforms.Resize(self.img_size)(gt_torch)
        gt_torch = transforms.CenterCrop(self.img_size)(gt_torch)
        img_np = gt_torch.cpu().numpy()
        img_np = np.moveaxis(img_np, 0, -1)

        bb,segments, _ , _= self.sam_obj.get_bbs(img_np)
        return bb, segments

def get_bb_cc(img_fn):
    img_pil= Image.open(img_fn)  
    img_np = np.array(img_pil)
    bb = get_bb_connectedcomp_one(img_np,is_inference=True)
    return bb


@click.command()
@click.option("--yaml_cfg")
@click.option("--crop_feats_dir")
@click.option("--vis_dir")
def main(yaml_cfg, crop_feats_dir, vis_dir):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    model_str = 'hf-hub:imageomics/bioclip'
    # model_str = "hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K"  # FIXME ablations 

    bb_instance_labels_dir=""

    with open(yaml_cfg, 'r') as file:
        cfgs_dict = yaml.safe_load(file)

    img_dir = cfgs_dict["inference_img_dir"]
    voting_topk = int(cfgs_dict["feature_voting_k"])
    use_size_votes =bool(cfgs_dict["use_size_matching"])
    use_text=bool(cfgs_dict["use_text"])
    nnk=int(cfgs_dict["hypersphere_nn_k"])

    os.makedirs(crop_feats_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    cbc = ClipBBClassifier(
                        model_str,
                        "sam", # "connectedcomp", # "label",
                        crop_feats_dir,
                        bb_instance_labels_dir=bb_instance_labels_dir,
                        min_bb_area= 16,  #224,
                        voting_topk=voting_topk,
                        use_size_votes=use_size_votes,
                        use_text=use_text,
                        NN_k=nnk,
                        )

    for img_fn in tqdm(os.listdir(img_dir)):
        mew, bbs, masks = cbc.get_semseg(os.path.join(img_dir, img_fn), is_vm=False)
        mew.save(os.path.join(vis_dir, img_fn))

if __name__ == '__main__':
    main()
