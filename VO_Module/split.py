import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp

scene = "/mnt/nas_8/group/lanxinyue/PanopticVisualOdom/Neural-iMAP/datasets/Virtual_KITTI2/Scene01"
mode = "clone"
images = np.array(sorted(
                glob.glob(osp.join(scene, mode, 'frames/rgb/Camera_0/*.jpg'))))
depths = sorted(
                glob.glob(osp.join(scene, mode, 'frames/depth/Camera_0/*.png')))
poses = np.loadtxt(
                osp.join(scene, mode, 'extrinsic.txt'), delimiter=' ', skiprows=1)[::2, 2:]
val_num = images.shape[0] // 7
train_num = images.shape[0] - val_num - val_num
images = images[train_num:train_num + val_num]

print(images)
# fo_flows = sorted(
#                     glob.glob(osp.join(scene, mode, 'frames/forwardFlow/Camera_0/*.png')))
# ba_flows = sorted(
#                     glob.glob(osp.join(scene, mode, 'frames/backwardFlow/Camera_0/*.png')))
# masks = sorted(
#                     glob.glob(osp.join(scene, mode, 'frames/dynamicMask/Camera_0/*.npy')))
# segments = sorted(
#                     glob.glob(osp.join(scene, mode, 'panoptic_gt_id/*.png')))
