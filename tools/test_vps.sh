#!/bin/bash

# test ps fusion 
CUDA_VISIBLE_DEVICES=4 python3 -W ignore VPS_Module/tools/train_net.py \
--config-file VPS_Module/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x_vkitti_511.yaml --num-gpu 1 \
--eval-only MODEL.WEIGHTS checkpoints/panFPN.pth \
OUTPUT_DIR shared_data/panoptic_segm_fusion/

# track match and prepare json and segm per scene
python VPS_Module/tools/1_tracking.py
python VPS_Module/tools/2_matching.py
python VPS_Module/tools/3_preparing.py

# calculate vpq per scene on clone 5:1:1
for sce in 's1' 's2' 's6' 's18' 's20'
do
    python VPS_Module/tools/4_eval_vpq.py \
    --gt-json datasets/Virtual_KITTI2/ALL_clone/vkitti_511_val_$sce.json \
    --gt-dir datasets/Virtual_KITTI2/ALL_clone/pan_gt_511_val_$sce \
    --pred-json shared_data/json/vo_fusion_vo_track_$sce.json \
    --pred-dir shared_data/final_vps_res/$sce \
    --output shared_data/vpq/$sce
done




