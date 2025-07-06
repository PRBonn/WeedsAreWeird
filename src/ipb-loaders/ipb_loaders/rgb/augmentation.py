"""A class for augmenting the images and their corresponding labels
Only works out for image and semantics for now
"""

import configparser
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw

class AugmentedIPBloader():
    def __init__(
            self,
            ipbloader,
            augmentations_cfg,
            is_semantics = False,
            is_instance=False,
            is_bb=False
            ):

        self.ipbloader = ipbloader

        config = configparser.ConfigParser()
        config.read(augmentations_cfg)
        self.aug_dict = config._sections["AUGMENTATIONS"]
        self.is_semantics = is_semantics
        self.is_instance = is_instance
        self.is_bb = is_bb
        if is_bb:
            self.is_instance = True
        # TODO test for bb without semseg?

    def collate_fn(self,batch):
        sample = batch[0]
        if "boxes" in sample:
            batch_dict={}
            for key in sample:
                if key == "boxes" or key == "boxes_class" or key == "boxes_area":
                    batch_dict[key] = [sample[key] for sample in batch]
                else:
                    batch_dict[key] = torch.utils.data.default_collate([sample[key] for sample in batch])
            return batch_dict
        else:
            return torch.utils.data.default_collate(batch)
        


    def __len__(self):
        return len(self.ipbloader)

    def __getitem__(self, index):
        sample = self.ipbloader.__getitem__(index)

        augmented_sample = self.augment(sample)
        if self.is_bb:
            bb, bb_cls, bb_area  = self.ipbloader.masks_to_boxes(sample)
            augmented_sample['boxes'] = bb.cpu().numpy()
            augmented_sample['boxes_class'] = bb_cls.cpu().numpy()
            augmented_sample['bb_weedmask'] = self.get_weedmask_from_bb(augmented_sample, bb, bb_cls)

        return augmented_sample

    def augment(self, sample):
        resize = int(self.aug_dict["resize"]) 
        if resize > 0:
            sample = self.resize(sample, resize)
        if self.aug_dict["random_hori_flip"] == "True":
            sample = self.random_hori_flip(sample)
        if self.aug_dict["random_verti_flip"] == "True":
            sample = self.random_verti_flip(sample)
        if self.aug_dict["normalize_mean"]!="None":
            sample = self.normalize(
                                    float(self.aug_dict["normalize_mean"]), 
                                    float(self.aug_dict["normalize_std"]), 
                                    sample
                                   )
        return sample

    def resize(self, sample, res):
        tf = transforms.Resize(res, interpolation=transforms.InterpolationMode.BILINEAR)
        tf_crop = transforms.CenterCrop(res)

        sample['image'] =  tf(sample['image'])
        sample['image'] =  tf_crop(sample['image'])


        if self.is_semantics:
            tf = transforms.Resize(res, interpolation=transforms.InterpolationMode.NEAREST)
            sample['semantic'] =  tf(sample['semantic'].unsqueeze(0))
            sample['semantic'] =  tf_crop(sample['semantic'])


        if self.is_instance:
            tf = transforms.Resize(res, interpolation=transforms.InterpolationMode.NEAREST)
            sample['instance'] =  tf(sample['instance'].unsqueeze(0))
            sample['instance'] =  tf_crop(sample['instance'])
        return sample

    def random_hori_flip(self, sample):
        is_flip = torch.rand(1) > 0.5
        if is_flip:
            tf = transforms.functional.hflip
            sample['image'] =  tf(sample['image'])
            if self.is_semantics:
                sample['semantic'] =  tf(sample['semantic'])
            if self.is_instance:
                sample['instance'] =  tf(sample['instance'])
        return sample

    def random_verti_flip(self, sample):
        is_flip = torch.rand(1) > 0.5
        if is_flip:
            tf = transforms.functional.vflip
            sample['image'] =  tf(sample['image'])
            if self.is_semantics:
                sample['semantic'] =  tf(sample['semantic'])
            if self.is_instance:
                sample['instance'] =  tf(sample['instance'])
        return sample

    def normalize(self, mean, std, sample):
        tf = transforms.Normalize(mean=mean, std=std)
        sample['image'] = tf(sample['image'])
        return sample

    def detransform(self, tensor):
        if self.aug_dict["normalize_mean"]!="None":
            tensor= tensor * float(self.aug_dict["normalize_std"]) + float(self.aug_dict["normalize_mean"])

        image_pil = transforms.functional.to_pil_image(
                tensor
                )
        return image_pil

    def get_weedmask_from_bb(self, sample, boxes, bb_cls):
        img_pil = Image.fromarray(np.zeros(sample["image"].shape[1:], dtype=np.uint8))
        img_drawer = ImageDraw.Draw(img_pil) 
        # boxes = sample['boxes']
        # bb_cls = sample['boxes_class']
        for bb, cls_tensor in zip(boxes, bb_cls):
            cls = cls_tensor.item()
            if cls == 2:
                img_drawer.rectangle(bb.tolist(), fill="white")
        return transforms.functional.pil_to_tensor(img_pil)


def draw_bbs(img_pil, bbs_tensor, bb_classes):
    drawer = ImageDraw.Draw(img_pil)
    for bb, cls_tensor in zip(bbs_tensor, bb_classes):
        cls = cls_tensor.item()
        if cls == 0:
            continue
        elif cls ==1:
            clr = "blue"
        elif cls == 2:
            clr = "red"
        else:
            clr = "black"
        drawer.rectangle(bb.tolist(), outline =clr)
    return img_pil


if __name__ == "__main__":
    from phenorob import PhenoRob
    example_fp = "example_cfg.cfg"
    pr = PhenoRob(
            data_source = "/media/linn/export10tb/PhenoRob_Dataset_Challenge_PhenoBench/dataset_exposed/PhenoBench",
            )
    bs=10
    aug_ipbloader = AugmentedIPBloader(pr, example_fp, is_semantics=True, is_bb=True)
    torchloader = torch.utils.data.DataLoader(
                                              aug_ipbloader, 
                                              batch_size=bs, 
                                              shuffle=True, 
                                              num_workers=0
                                              )
    for batch in tqdm(torchloader):
        print(batch["filename"])
        for i in range(bs):
            sample_img = batch['image'][i]
            image_pil = aug_ipbloader.detransform(sample_img)
            image_pil.save(batch["filename"][i])
            # draw_bbs(image_pil, batch["boxes"][i], batch["boxes_class"][i])
            image_pil.save("meow.png")
            bb_mask_tensor=batch["bb_weedmask"][i]
            bb_mask_pil = transforms.functional.to_pil_image(bb_mask_tensor)
            bb_mask_pil.save("meowoew.png")

            semantics = (batch['semantic'][i][0] * 125).cpu().numpy().astype(np.uint8)
            sem_pil = Image.fromarray(semantics)
            sem_pil.save("sem_"+batch["filename"][i])


