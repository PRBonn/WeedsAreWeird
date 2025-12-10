# Weeds are Weird (WaW)

Zero-Shot Semantic Segmentation for Robots in Agriculture (IROS 2025)   
[Paper](https://www.ipb.uni-bonn.de/pdfs/chong2025iros.pdf)     
[Demo (vegetation mask with SAM only)](https://huggingface.co/spaces/linnchong/veg_mask_SAM)   
![Motivation](assets/motivation.png)

Our approach can segment crop plants and weeds without labels. 
We leverage foundation models [SAM](https://github.com/facebookresearch/segment-anything) and the [ViT from BioCLIP](https://imageomics.github.io/bioclip/) to build a bag of features representing crop plants. 
During inference, we extract plant features and compare them with the bag of features.
Plant features with low similarity with the bag of features are inferred as weeds.

![Results](assets/qual_results.png)
Qualitative results on different datasets. The top row shows input image, the second row shows ground truth, and the third row
shows our performance.


## Installation
```bash
pip install -r requirements.txt
cd src/ipb-loaders; pip install -U -e . 
wget -P scripts/sam/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```
TODO: update to the correct list after all other checks
/home/linn/venvs/waw-reprod

## Usage
### 0. Prepare data sets
The data set should follow the following directory structure:  
```
${DATASET_PARENT_DIR}  
├── train  
│   └── images  
│   └── semantics  
├── val  
│   ├── images  
│   └── semantics  
└── test  
    ├── images  
```
For further details, see the data set directory structure of [PhenoBench](https://www.phenobench.org/).

### 1. Curating the Bag of Features of Crop Plants
You can use the bag of features from the paper:
+ [PhenoBench](https://www.ipb.uni-bonn.de/html/projects/chong2025iros/phenobench-final_crops.zip)
+ [SB20](https://www.ipb.uni-bonn.de/html/projects/chong2025iros/sb20-final_crops.zip) 
+ [CropAndWeed-Sugar Beet](https://www.ipb.uni-bonn.de/html/projects/chong2025iros/cnw-sb-final_crops.zip)
+ [CropAndWeed-Maize](https://www.ipb.uni-bonn.de/html/projects/chong2025iros/cnw-maize-final2_crops.zip)

OR
 
Build your own (e.g., for a different dataset):
1. Obtain vegetation segments for train split images with SAM
    ```bash
    python3 scripts/get_bb_clips.py \
      --input_dir <path to dataset parent dir> \
      --output_dir <path of output dir> \
      --split train \
      --vm_dir <optional, path of output dir with vegetation masks> \
      --vis_dir <optional, path of output dir of visualisations> \
      --aug_cfg <optional, path to augmentation configuration file. defaults to ./cfgs/augs_clip.cfg> \
      --points_per_side <optional, number of point prompts for SAM> \
      --is_remove_overlap
    ```
    This will give the patches saved as .png files. 
    Additionally, if specified, the point prompts used to prompt SAM in vis_dir 
    and the resultant vegetation masks of the input images in vm_dir.
    
2. Separate out the popular features using BioCLIP's ViT   
    First, you need to create a .yaml file; see ./cfgs/vote_phenobench.yaml for an example.
    ```bash
    python scripts/bioclip_popularity.py \ 
      --yaml_cfg ./cfgs/vote_phenobench.yaml \
      --output_dir <output dir for patches of crop plants>
    ```
    (This might take some time, especially if the number of features is large)

### 2. Inference 

First you need to get the vegetation masks on the test set:
```bash
python scripts/get_bb_clips.py \
  --input_dir <path to dataset parent dir> \
  --output_dir <path of output dir> \
  --split test \
  --vm_dir <path of output dir with vegetation masks> \
  --vis_dir <optional, path of output dir of visualisations> \
  --points_per_side <optional, number of point prompts for SAM> \
  --aug_cfg ./cfgs/test_augs.cfg 
```

Then, we can get the semantic segmentation:
```bash
python scripts/get_predictions_bioclip.py \
  --yaml_cfg ./cfgs/vote_phenobench.yaml \
  --crop_feats_dir <output dir for patches of crop plants> \
  --vis_dir <output vis dir path>;
python scripts/get_cws.py \
  --input_dir <output vis dir path> \
  --output_dir <predictions dir path> \
  --vm_dir <vegetation m1ask directory> \
  --ds_dir <dataset directory> \
  --img_size <image size> \
  --center_crop \
  --is_vis;
```

### 3. Evaluation
```bash
python scripts/evaluate.py \
  --semantics_dir <predictions dir path> \
  --gt_dir <ground truth labels dir path> \
  --img_size <image size in px> \
  --center_crop;
```

Note: For PhenoBench test split evaluation, we used the CodaLab benchmark online.

## Machine specs
We developed/tested this code on Python 3.12 and utilising a NVIDIA RTX A6000 GPU.

## Cite us
```
@inproceedings{chong2025iros,
author = {Y.L. Chong and L. Nunes and F. Magistri and X. Zhong and J. Behley and C. Stachniss},
title = {{Zero-Shot Semantic Segmentation for Robots in Agriculture}},
booktitle = iros,
year = 2025,
codeurl = {https://github.com/PRBonn/WeedsAreWeird},
videurl = {https://youtu.be/1ORs07F0RsE}
}
```
