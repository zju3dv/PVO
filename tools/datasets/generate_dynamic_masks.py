import torch
from torch.utils.data import DataLoader
import os.path as osp
import os
import numpy as np
import time
import sys
sys.path.append('VO_Module/droid_slam')
from data_readers.factory import dataset_factory
import data_readers.vkitti2

scenes = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']

for scene_id in scenes:
    res_path = osp.join('datasets/Virtual_KITTI2', scene_id, 'clone/frames/dynamicMask_00/Camera_0')
    if not osp.exists(res_path):
        os.makedirs(res_path)

    db = dataset_factory(['vkitti2'], datapath='datasets/Virtual_KITTI2',
                         n_frames=3, flow_label=True, aug_graph=False, split_mode='train', foo=False,
                         scene_id = scene_id,
                         need_inv=False, build_mask=True, mode='semisup',
                         rebuild=True,)
    train_loader = DataLoader(db, batch_size=1, num_workers=2)

    first_masks, last_masks = None, None
    first_vals, last_vals = None, None

    for i_batch, item in enumerate(train_loader):
        fo_masks, ba_masks, fo_vals, ba_vals = [x for x in item]

        if i_batch == 0:
            first_masks = fo_masks[0, 0]
            first_vals = fo_vals[0, 0]
        elif i_batch == len(train_loader)-1:
            last_masks = ba_masks[0, 1]
            last_vals = ba_masks[0, 1]

        gt_masks = torch.clamp(fo_masks[0, 1]+ba_masks[0, 0], min=0, max=1)
        gt_vals = (((fo_vals[0, 1]+ba_vals[0, 0])/2) > 0.5).float()
        print(float(gt_masks.mean()), float(gt_vals.mean()))

        index = i_batch + 1
        np.save(osp.join(res_path, 'dymask_{:0>5d}.npy'.format(
            index)), torch.cat([gt_masks, gt_vals], dim=-1).numpy())

    np.save(osp.join(res_path, 'dymask_{:0>5d}.npy'.format(
        0)), torch.cat([first_masks, first_vals], dim=-1).numpy())
    np.save(osp.join(res_path, 'dymask_{:0>5d}.npy'.format(
        len(train_loader)+1)), torch.cat([last_masks, last_vals], dim=-1).numpy())