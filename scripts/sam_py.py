# based on readme https://github.com/facebookresearch/segment-anything?tab=readme-ov-file

from segment_anything import SamPredictor, sam_model_registry 
from sam.automatic_mask_generator import SamAutomaticMaskGeneratorLinn as SamAutomaticMaskGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from get_anomaly_masks_optuna import exg_correct, new_exgr, exgr, ikaw, mgrvi, gli, exg, exb
import torch
import random
from torchvision import transforms, ops
import cv2
import time
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

class SAM_bbs:
    def __init__(
            self, 
            device,
            img_size=1024,
            exg_thresh=0.1,
            exg_percent=0.1,   
            exg_nms_weight=0.5,  # 0 to use just sam score, 1 to just use nms
            nms_iou=0.3,
            points_per_batch=10,
            is_nms=True,  
            is_multimask=True,
            num_threads=6,  # num cpu threads for torch, incl nms
            max_seg_size=512,
            seed_exg_thresh=0.1,
            points_per_side=96,
            min_mask_region_area=100,
            use_sam2=False,
            use_sam2_vid=False,  # must be with use_sam2
            is_remove_overlap=False,
            use_clip=False,
            filter_vm_ksize=1,  # for postprocessing of exg before seeding if > 1
            concomp_k_size=-1,
            use_grid=False,  # for ablation, dont move seed from grid vertices
            checkpoint="./scripts/sam/sam_vit_l_0b3195.pth"
            ):
        torch.set_num_threads(num_threads)
        self.num_threads=num_threads
        self.max_seg_size=max_seg_size
        self.seed_exg_thresh=seed_exg_thresh
        self.use_grid=use_grid

        self.use_clip = use_clip
        if self.use_clip:
            import open_clip
            # regular clip
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
            self.clip_model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
            self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
            text = self.clip_tokenizer(["soil", "plant"])
            self.clip_text_features = self.clip_model.encode_text(text)
            self.clip_text_features /= self.clip_text_features.norm(dim=-1, keepdim=True)

        self.img_size=img_size
        self.is_remove_overlap = is_remove_overlap
        self.concomp_k_size = concomp_k_size
        if filter_vm_ksize >1:
            self.is_filtervm = True
            self.filtervm_k = filter_vm_ksize
        else:
            self.is_filtervm = False
            self.filtervm_k = 1
        
        self.exg_thresh = exg_thresh
        self.exg_percent = exg_percent

        self.points_per_batch=points_per_batch
        self.exg_nms_weight=exg_nms_weight
        self.nms_iou=nms_iou
        self.is_nms = is_nms
        self.use_sam2 = use_sam2
        self.use_sam2_vid = use_sam2_vid

        """
        # default
        self.points_per_side=32
        self.pred_iou_thresh=0.88
        self.stability_score_thresh=0.95
        self.crop_n_layers=0
        self.crop_n_points_downscale_factor=1
        self.min_mask_region_area=0
        """

        """
        # small
        self.points_per_side=32
        self.pred_iou_thresh=0.86
        self.stability_score_thresh=0.92
        self.crop_n_layers=1
        self.crop_n_points_downscale_factor=2
        self.min_mask_region_area=100
        """

        # linn
        self.points_per_side=points_per_side
        self.pred_iou_thresh=0.86
        self.stability_score_thresh=0.95
        self.crop_n_layers=0
        self.crop_n_points_downscale_factor=2
        self.min_mask_region_area=min_mask_region_area
        self.is_multimask = is_multimask
        self.device = device

        if self.use_sam2:

            from sam2.build_sam import build_sam2, build_sam2_video_predictor
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            checkpoint = "./sam/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"  # relative to sam2 repo
            if use_sam2_vid:
                self.predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=self.device)
            else:
                self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        else:
            self.sam = sam_model_registry["vit_l"](checkpoint=checkpoint) 
            self.sam = self.sam.to(device=device)
            self.predictor = SamPredictor(self.sam)

            self.mask_generator = SamAutomaticMaskGenerator(
                                model=self.sam,
                                points_per_side=self.points_per_side,
                                points_per_batch=self.points_per_batch,
                                pred_iou_thresh=self.pred_iou_thresh,
                                stability_score_thresh=self.stability_score_thresh,
                                crop_n_layers=self.crop_n_layers,
                                crop_n_points_downscale_factor=self.crop_n_points_downscale_factor,
                                min_mask_region_area=self.min_mask_region_area,
                                is_multimask=self.is_multimask,
                            )



    def get_class_clip(self, img_pil):
        image = preprocess(img_pil).unsqueeze(0)
        image_features = self.clip_model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ self.clip_text_features.T).softmax(dim=-1)
        return text_probs

    def show_anns(self, anns, img, is_second=False):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
    
        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            if is_second:
                m = ann['segmentation2'].cpu().numpy()
            else:
                m = ann['segmentation'].cpu().numpy()
            color_mask = np.concatenate([np.random.random(3), [1.0]])
            img[m.astype(bool)] = color_mask
        ax.imshow(img,  origin='upper')

    def get_seeds_grid(self, exg_img):  # for ablation study
        seeds = []
        w, h = exg_img.shape
        dw = w / self.points_per_side
        dh = h / self.points_per_side
        for i in range(self.points_per_side):
            for j in range(self.points_per_side):
                new_i = int(i * dh)
                new_j = int(j * dw)
                seeds.append([new_j/w, new_i/h])
        seeds_l = [np.array(seeds)]
        return seeds_l


    def get_seeds(self, exg_img):
        seeds = []
        w, h = exg_img.shape
        dw = w / self.points_per_side
        dh = h / self.points_per_side
        for i in range(self.points_per_side):
            for j in range(self.points_per_side):
                new_i = int(i * dh)
                new_j = int(j * dw)
                clip = exg_img[new_i:int(new_i+dh),new_j:int(new_j+dw)]
                if clip.size == 0:
                    continue
                argm = clip.argmax()
                clip_i, clip_j = np.unravel_index(argm, clip.shape)
                if clip[clip_i, clip_j] >  self.seed_exg_thresh:
                    nnew_i = max(clip_i + new_i,1)
                    nnew_j = max(clip_j + new_j ,1)
                    nnew_i = min(nnew_i, w-1)
                    nnew_j = min(nnew_j, h-1)
                    seeds.append([nnew_j/w, nnew_i/h])  # x,y
                else:
                    # import pdb; pdb.set_trace()
                    pass
        seeds_l = [np.array(seeds)]
        if self.crop_n_layers>0:
            seeds_l.append(np.array(seeds))  # FIXME
        return seeds_l

    def generate(self, img):
        masks = self.mask_generator.generate(img)
        return masks

    def get_bbs_onebyone(self, img, seeds=None):
        masks_l = []
        scores_l = []  
        masks_l2=[]
        scores_l2=[]
        input_points = []
        input_labels = []
        if self.use_sam2_vid:
            self.inference_state = self.predictor.init_state(video_path="")
        else:
            self.predictor.set_image(img)

        img_exg = exg(img) /255  # use normalised one to get seeds 

        if self.is_filtervm:
            # remove noise
            kernel_size = self.filtervm_k
            kernel = np.ones((kernel_size,kernel_size),np.float32)/(kernel_size * kernel_size)
            img_exg = cv2.erode(img_exg, kernel, iterations=1)

        if seeds is None:
            if self.use_grid:
                seeds = self.get_seeds_grid(img_exg)
            else:
                seeds = self.get_seeds(img_exg)

        def run_pred(input_points, input_labels, masks_l, scores_l):
            if self.use_sam2:
                if self.use_sam2_vid:
                    self.predictor.reset_state(self.inference_state)
                    ann_frame_idx = 0
                    for ann_obj_id, (point, label) in enumerate(zip(input_points, input_labels)):
                        points = np.expand_dims(point.astype(np.float32), 0)
                        labels = label.astype(np.int32)
                        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                            inference_state=self.inference_state,
                            frame_idx=ann_frame_idx,
                            obj_id=ann_obj_id,
                            points=points,
                            labels=labels,
                        )

                    for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
                        masks = out_mask_logits > 0.0
                        masks = masks.cpu()
                        masks = masks.flatten(0,1)
                        scores = torch.zeros((masks.shape[0]))

                        if out_frame_idx == 0:
                            for mask,score in zip(masks, scores):
                                masks_l.append(mask)
                                scores_l.append(score)
                        elif out_frame_idx == 1:
                            for mask,score in zip(masks, scores):
                                masks_l2.append(mask)
                                scores_l2.append(score)
                        else:
                            print("more than two frames?!")
                            import pdb; pdb.set_trace()  # TODO change to raise execption

                else:
                    ip = np.array(input_points)
                    ip = np.expand_dims(ip, 1)
                    pl = np.array(input_labels)
                    masks, scores, logits = self.predictor.predict(
                        point_coords=ip,
                        point_labels=pl,
                        multimask_output=self.is_multimask,
                            )
                    masks = torch.tensor(masks)
                    scores = torch.tensor(scores)
            else:
                point_coords = self.predictor.transform.apply_coords(np.array(input_points), self.predictor.original_size)
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)

                labels_torch = torch.as_tensor(np.array(input_labels), dtype=torch.int, device=self.device)
                input_points_torch, input_labels_torch = coords_torch[:, None, :], labels_torch

                masks, scores, logits = self.predictor.predict_torch(
                    point_coords=input_points_torch,
                    point_labels=input_labels_torch,
                    multimask_output=self.is_multimask,
                )
                del coords_torch, labels_torch
                del input_points_torch, input_labels_torch
                masks = masks.cpu()
            if not self.use_sam2_vid:
                if len(scores.shape) > 1:
                    scores = scores.flatten(0,1)
                    masks = masks.flatten(0,1)  # TODO maybe we should do something fancier here
                for mask,score in zip(masks,scores):  
                    if mask.sum() > self.min_mask_region_area:
                        masks_l.append(mask)
                        scores_l.append(score)

        for seed in seeds[0]:
            input_point = seed * self.img_size
            input_label = np.array([1])

            input_points.append(input_point)
            input_labels.append(input_label)

            if len(input_points) == self.points_per_batch:
                run_pred(input_points, input_labels, masks_l, scores_l)
                input_points = []
                input_labels = []
        if len(input_points) > 0:
            run_pred(input_points, input_labels, masks_l, scores_l)

        # FIXME there is no guarantee that there will be equal number of masks in both images
        # for example if the segment is lost, etc what happens? empty mask?

        # convert masks into bbs
        mask_dict_l = []
        def eins_iter(mask, score, mask2=None):
            start_time1=time.time()
            masks_out={}
            coords = np.argwhere(mask.cpu().numpy())
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            bb=[x_min, y_min, x_max-x_min, y_max-y_min]
            masks_out["segmentation"]=mask.to(bool)
            if not mask2 is None:
                masks_out["segmentation2"]=mask2.to(bool)
            masks_out["bbox"]= bb
            masks_out["score"]=score
            masks_out["area"] = mask.sum()
            masks_out["vm"] = img_exg
            end_time1 = time.time()
            return masks_out

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            if self.use_sam2_vid:
                future_to_input = {executor.submit(eins_iter, mask, score, mask2): (mask, score) for (mask, score, mask2) in zip(masks_l, scores_l, masks_l2) }
            else:
                future_to_input = {executor.submit(eins_iter, mask, score): (mask, score) for (mask, score) in zip(masks_l, scores_l) }
            for future in as_completed(future_to_input):
                masks_out = future.result()
                mask_dict_l.append(masks_out)

        end_time = time.time()
        print(f"time taken for batching {len(mask_dict_l)} preds", end_time - start_time)

        return mask_dict_l, seeds, img_exg


    def get_bbs(self, img, is_exggen=True, is_filter=True,seeds=None):
        start_time = time.time()
        bbs=[]
        masks_filtered=[]
        if is_exggen:
            masks, seeds, img_exg = self.get_bbs_onebyone(img, seeds)
        else:
            masks = self.generate(img)
        mask_time = time.time()
        print("time taken for mask", mask_time-start_time)

        for mask in masks:
            x, y, w, h = mask["bbox"]
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            if w == self.img_size-1 and h == self.img_size-1:
                continue  # the soil / the whole img
            cropped_img = img[y:y+h, x:x+w]
            segment = mask["segmentation"][y:y+h, x:x+w]
            cropped_img_exg = exgr(cropped_img)   # TODO could be made faster if just calc exg first
            vm = cropped_img_exg > self.exg_thresh
            vm = vm * segment.cpu().numpy()
            vm_percent = vm.sum() / mask["segmentation"].sum()
            if vm_percent > self.exg_percent:

                # check saturation
                cropped_pil = Image.fromarray(cropped_img)
                cropped_pil = cropped_pil.convert('HSV')
                hsv_np = np.array(cropped_pil)
                s_np = hsv_np[:,:,1] 
                s_score = s_np[segment.cpu().numpy()].mean()
                if s_score > 20: 
                    bbs.append([x, y, x+w, y+h])
                    masks_filtered.append(mask)
        vm_time = time.time()
        print("time taken for vm filtering", vm_time-mask_time)


        if is_filter:
            bbs, masks_filtered = self.filter_bbs(np.array(bbs), masks_filtered)
        
        filter_time = time.time()
        print("time taken for filtering", filter_time-vm_time)
        return np.array(bbs), masks_filtered, seeds, img_exg

    def filter_bbs(self, bbs, masks):
        start_time = time.time()
        bbs, masks = self.remove_small_bbs(bbs, masks)
        end_time = time.time()
        print("time taken for remove small masks", end_time - start_time)
        start_time = time.time()
        bbs, masks = self.remove_large_bbs(bbs, masks, self.max_seg_size)
        end_time = time.time()
        print("time taken for remove large bbs", end_time - start_time)
        start_time = time.time()
        if self.is_nms:
            bbs, masks = self.nms(bbs, masks)
            end_time = time.time()
            print("time taken for nms", end_time - start_time)
        if self.concomp_k_size > 0:
           bbs, masks = self.remove_nonconnected_comps(bbs, masks, self.concomp_k_size)
        if self.is_remove_overlap:
           bbs, masks = self.remove_leaves_if_plant_exists(bbs,masks)
        if self.use_clip:
            bbs, masks = self.remove_lowclip(bbs,masks)
        return bbs, masks

    def remove_lowclip(bbs,masks):
        for bb, mask in zip(bbs, masks):
            pass
        return  bbs, masks

    def remove_leaves_if_plant_exists(self, bbs, masks, min_overlap_percent=0.9):
        new_bbs = []
        new_masks = []
        masks2 = copy.deepcopy(masks)

        for bb, mask in zip(bbs, masks):
            is_represented_larger = False
            for t_mask in masks2:
                overlap =  (mask["segmentation"] * t_mask["segmentation"]).sum()
                source = mask["segmentation"].sum()
                if ((overlap/source) == 1): # self instance
                    continue
                if (overlap/source) > min_overlap_percent:
                    is_represented_larger = True
                    continue 
            if not is_represented_larger:
                new_bbs.append(bb)
                new_masks.append(mask)
        return new_bbs, new_masks

    def remove_nonconnected_comps(self, bbs, masks, k_size=5):
        new_bbs = []
        new_masks = []

        for bb, mask in zip(bbs, masks):
            mask_mew = mask["segmentation"].cpu().numpy()
            kernel_size = k_size
            kernel = np.ones((kernel_size,kernel_size),np.float32)/(kernel_size * kernel_size)
            mask_mew = cv2.dilate(mask_mew.astype(np.uint8), kernel, iterations=1)

            (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(
                        mask_mew.astype(np.uint8),8)
            if numLabels <= 2: # bg and fg
                new_bbs.append(bb)
                new_masks.append(mask)

        return new_bbs, new_masks

    def nms(self, bbs, masks):
        if len(bbs) ==0:
            return bbs, masks
        scores =[]
        bbs_torch = torch.tensor(np.array(bbs), dtype=torch.float32)
        for mask in masks:
            vm_score = (mask["vm"]*mask["segmentation"].cpu().numpy()).sum() / mask["segmentation"].sum()
            vm_score = vm_score * self.exg_nms_weight
            scores.append(mask["score"] * (1.0 - self.exg_nms_weight) + vm_score * self.exg_nms_weight )
        scores = torch.tensor(scores)
        print(bbs_torch.shape)
        selected_indices= ops.nms(bbs_torch.cpu(), scores.cpu(), self.nms_iou)
        new_bbs = [bbs[i] for i in selected_indices]
        new_masks = [masks[i] for i in selected_indices]

        del scores, bbs_torch

        return new_bbs, new_masks

    def remove_small_bbs(self, bbs, masks):
        new_bbs = []
        new_masks =[]
        for bb, mask in zip(bbs, masks):
            if mask["area"] >= self.min_mask_region_area: 
                new_bbs.append(bb)
                new_masks.append(mask)
        return new_bbs, new_masks

    def get_vm(self, masks):
        if len(masks) == 0:
            return np.zeros((self.img_size, self.img_size))
        vm = np.zeros((masks[0]["segmentation"].shape))
        for mask in masks:
            vm[mask["segmentation"].cpu().numpy()]=1

        # remove small components
        (numLabels, labels, stats, centroids) =cv2.connectedComponentsWithStats(
                    vm.astype(np.uint8),8)
        vm2 = np.zeros(vm.shape)
        for seg_i in range(numLabels):
            if seg_i == 0:  # bg
                continue
            if stats[seg_i,-1] > self.min_mask_region_area:
                vm2[labels==seg_i]=1
        return vm

    def remove_large_bbs(self, bbs, masks, max_bb):
        new_masks=[]
        new_bbs=[]
        for i, (bb, mask) in enumerate(zip(bbs, masks)):
            start_time=time.time()
            y_min, x_min, y_max, x_max = bb
            w = x_max - x_min
            h = y_max - y_min
            if w > max_bb or h > max_bb:  # remove larger bbs of instance if there is another mask representing it
                continue
            new_bbs.append(bb)
            new_masks.append(mask)
            end_time = time.time()
        return new_bbs, new_masks

    def has_overlap(self, source, list_masks, min_overlap=100):
        for mask in list_masks:
            overlap = torch.logical_and(mask["segmentation"], source["segmentation"])
            if overlap.sum() > min_overlap:
                return True
        return False


    def get_bbs_frombatch(self, batch, return_seeds=False, is_filter=True):
        bbs=[]
        bbs_class=[]
        seeds_l=[]
        masks_l = []
        img_exg_l = []
        for img_torch in batch["image"]:
            img = img_torch / 2. + 0.5
            img_pil = transforms.functional.to_pil_image(img)
            img = np.array(img_pil)

            bb, masks, seeds, img_exg =self.get_bbs(img, is_filter=is_filter)
            bbs.append(bb)
            bbs_class.append(np.ones(bb.shape[0]) * 3)  # TODO would be nice to send in the segment or exg instead
            seeds_l.append(seeds)
            masks_l.append(masks)  # TODO wow i really need a class to hold all this
            img_exg_l.append(img_exg)
        if return_seeds:
            return bbs, bbs_class, seeds_l, masks_l, img_exg_l
        else:
            return bbs, bbs_class

    def vis_bbs(self, bbs, ax, clr="green"):
        for bb in bbs:
            x0 = bb[0]
            y0 = bb[1]
            w = bb[2]-bb[0]
            h = bb[3]-bb[1]
            ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=clr, facecolor=(0,0,0,0), lw=2))

    def show_points(self,coords,ax, marker_size=60):
        pos_points = coords[0] * self.img_size
        if len(pos_points) > 0:
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='b', marker='o', s=marker_size, edgecolor='white', linewidth=1)


    def check_masks(self, masks, img):
        for i, mask in enumerate(masks):
            plt.imshow(img)
            plt.imshow(mask["segmentation"].cpu().numpy(), cmap='Blues', alpha=0.5)
            plt.savefig(f"mask{i}.png")
            plt.clf()

        return
    

if __name__ == '__main__':
    seed=0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    device="cuda" 

    sam_obj = SAM_bbs(device)

    img_pil = Image.open("img.png")
    img_np = np.array(img_pil)

    plt.figure(figsize=(20, 20)) 

    plt.imshow(img_pil)
    bbs, masks, seeds= sam_obj.get_bbs(img_np, is_filter=False)
    bbs2, masks2 = sam_obj.filter_bbs(bbs, masks)
    sam_obj.show_anns(masks, img_np)
    sam_obj.vis_bbs(bbs, plt.gca())
    sam_obj.vis_bbs(bbs2, plt.gca(), clr="blue")
    sam_obj.show_points(seeds, plt.gca())
    plt.xticks(range(0, 1024, 32)) 
    plt.yticks(range(0, 1024, 32)) 
    plt.grid()
    plt.savefig("exggrid2.png")
    plt.clf()
    sam_obj.check_masks(masks2,img_np)

    vm_mask = sam_obj.get_vm(masks2)
    Image.fromarray(vm_mask.astype(np.uint8) * 255).save("vm.png")
