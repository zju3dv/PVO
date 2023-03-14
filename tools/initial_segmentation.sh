
# generate initial segmentation
CUDA_VISIBLE_DEVICES=0 python3 -W ignore VPS_Module/tools/train_net.py \
--config-file VPS_Module/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x_vkitti_init_clone.yaml --num-gpu 1 \
--eval-only MODEL.WEIGHTS checkpoints/panFPN.pth \
OUTPUT_DIR shared_data/panoptic_segm_init_clone/

CUDA_VISIBLE_DEVICES=0 python3 -W ignore VPS_Module/tools/train_net.py \
--config-file VPS_Module/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x_vkitti_init_test.yaml --num-gpu 1 \
--eval-only MODEL.WEIGHTS checkpoints/panFPN.pth \
OUTPUT_DIR shared_data/panoptic_segm_init_15-deg-left/

# split initial segmentation into datasets/
python tools/split_init_segm.py
