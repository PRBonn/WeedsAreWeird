# Reproducibility

## Artifacts
See main repo README for instructions to replicate these results.
1. Directories containing clips for each dataset's train set:   
    + PhenoBench: ipb14:/export/tmp/tmp_linn/waw_reproducibility/phenobench/bb/pb-train
    + SB20: ipb14:/export/tmp/tmp_linn/waw_reproducibility/sb20/bb/sb20-unlabelled_subsampled
    + CnW-SB: ipb14:/export/tmp/tmp_linn/waw_reproducibility/cnw-sb/bb/cnw-sb-train-concomp-1
    + CnW-Maize: ipb14:/export/tmp/tmp_linn/waw_reproducibility/cnw-maize/bb/cnw-maize-train
1. Bag of Features Directories:
    + PhenoBench: ipb14:/export/tmp/tmp_linn/waw_reproducibility/phenobench/bof/phenobench-final_crops
    + SB20: ipb14:/export/tmp/tmp_linn/waw_reproducibility/sb20/bof/sb20-final_crops
    + CnW-SB: ipb14:/export/tmp/tmp_linn/waw_reproducibility/cnw-sb/bof/cnw-sb-final_crops
    + CnW-Maize: ipb14:/export/tmp/tmp_linn/waw_reproducibility/cnw-maize/bof/cnw-maize-final2_crops
1. Vegetation masks for test set:
    + PhenoBench: ipb14:/export/tmp/tmp_linn/waw_reproducibility/phenobench/vm_test/vm
    + SB20: ipb14:/export/tmp/tmp_linn/waw_reproducibility/sb20/vm_test/vm
    + CnW-SB: ipb14:/export/tmp/tmp_linn/waw_reproducibility/cnw-sb/vm_test/vm
    + CnW-Maize: ipb14:/export/tmp/tmp_linn/waw_reproducibility/cnw-maize/vm_test/vm
1. Semantic segmentation for test set:
    + PhenoBench: ipb14:/export/tmp/tmp_linn/waw_reproducibility/phenobench/cws/phenobench-final/cws_t200/semantics
    + SB20: ipb14:/export/tmp/tmp_linn/waw_reproducibility/sb20/cws/sb20-final/cws_t150/anomalyv1_gt2
    + CnW-SB: ipb14:/export/tmp/tmp_linn/waw_reproducibility/cnw-sb/cws/cnw-sb-final/cws_t150/anomalyv1_gt2
    + CnW-Maize: ipb14:/export/tmp/tmp_linn/waw_reproducibility/cnw-maize/cws/cnw-maize-final2/cws_t200/anomalyv1_gt2
1. Datasets used:
    + PhenoBench: ipb14:/export/datasets/PhenoBench/processed
    + SB20: ipb14:/export/datasets/SB20/dataset
    + CnW-SB: ipb14:/export/datasets/CropAndWeed/chong2025iros/crop_and_weed_sugar_beets
    + CnW-Maize: ipb14:/export/datasets/CropAndWeed/chong2025iros/crop_and_weed_maize

### Vegetation Mask
Vegetation masks for validation sets:
+ PhenoBench: ipb14:/export/tmp/tmp_linn/waw_reproducibility/phenobench/vm_val
+ SB20: ipb14:/export/tmp/tmp_linn/waw_reproducibility/sb20/vm_val
+ CnW-SB: ipb14:/export/tmp/tmp_linn/waw_reproducibility/cnw-sb/vm_val
+ CnW-Maize: ipb14:/export/tmp/tmp_linn/waw_reproducibility/cnw-maize/vm_val

Steps to reproduce:
1. Get vegetation masks on val set
    ```bash
    cd ..
    python scripts/get_bb_clips.py \
    --input_dir <path to dataset parent dir> \
    --output_dir <path of output dir> \
    --split val \
    --vm_dir <path of output dir with vegetation masks> \
    --aug_cfg ./cfgs/test_augs.cfg;
    ```

2. Evaluate
    ```bash
    python ablations_N_baselines/evaluate_vm.py \
    --vm_dir <path to vegetation masks> \
    --gt_dir <path to ground truth semantic (CWS) labels> \
    --img_size <image size>
    ```
