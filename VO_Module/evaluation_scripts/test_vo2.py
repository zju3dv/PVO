import sys
sys.path.append('Neural-iMAP/droid_slam')

import torch
from torch.utils.data import DataLoader
import glob
import os
import os.path as osp
import cv2
import numpy as np
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from collections import OrderedDict
from lietorch import SE3
from flow_vis_utils import flow_to_image
from geom.projective_ops import projective_transform, coords_grid
from geom.graph_utils import graph_to_edge_list
from droid_net import DroidNet, upsample_inter
from data_readers.factory import dataset_factory
import data_readers.vkitti2
import argparse

def resize(mask, size, need_permute):
    if need_permute:
        mask = mask.permute(0, 1, 4, 2, 3).contiguous()
    batch, num, dim, ht, wd = mask.shape
    mask = mask.view(batch*num, dim, ht, wd)
    mask = F.interpolate(mask, size=size, mode='bilinear', align_corners=True)
    if need_permute:
        mask = mask.permute(0, 2, 3, 1).contiguous()
        return mask.view(batch, num, size[0], size[1], dim)
    else:
        return mask.view(batch, num, dim, size[0], size[1])

dic = { "Scene01":"0001",
        "Scene02":"0002",
        "Scene06":"0006",
        "Scene18":"0018",
        "Scene20":"0020",}

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_frames", type=int, default=2)
    parser.add_argument("--save_npy", type=bool, default=True, help="save flow and depth")
    parser.add_argument("--image_size", default=[376, 1248])

    parser.add_argument("--scene", default="Scene02")
    parser.add_argument("--full_flow_dir", default="shared_data/full_flow")
    parser.add_argument("--depth_dir", default="shared_data/depth")
    parser.add_argument("--weights_file", default="checkpoints/vkitti2_dy_train_semiv4_080000.pth")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = init()
    full_flow_dir = args.full_flow_dir
    depth_dir = args.depth_dir
    img_size = args.image_size
    n_frames = args.n_frames
    scene_id = args.scene
    save_npy = args.save_npy
    if not osp.exists(full_flow_dir):
        os.makedirs(full_flow_dir)
    if not osp.exists(depth_dir):
        os.makedirs(depth_dir)

    db = dataset_factory(['vkitti2'], datapath='datasets/Virtual_KITTI2', do_aug=False,
                     n_frames=n_frames, flow_label=False, aug_graph=False, split_mode='train', foo=True,
                     scene_id = scene_id,
                     need_inv=False, build_mask=False, crop_size=img_size, mode='semisup',
                     rebuild=True, 
                     )
    train_loader = DataLoader(db, batch_size=1, num_workers=2)

    device = 'cuda:0'
    torch.cuda.set_device(int(device.split(':')[-1]))

    graph = OrderedDict()
    for i in range(n_frames):
        graph[i] = [j for j in range(n_frames) if abs(i-j) == 1]
    ii, jj, _ = graph_to_edge_list(graph)

    model = DroidNet()
    state_dict = OrderedDict([(k.replace("module.", ""), v) for (k, v) in torch.load(args.weights_file, device).items()])
    model.load_state_dict(state_dict)
    model.to(device).eval()
    
    coords0 = coords_grid(img_size[0], img_size[1], device=device)

    Gs = None
    disps_est = None
    for i_batch, item in enumerate(train_loader):
        images, poses, disps, intrinsics, gt_masks, gt_vals, segments = [ x.to('cuda') for x in item if type(x) != list]
        file1, file2 = item[0]
        hr, wr = img_size[0]/images.shape[3], img_size[1]/images.shape[4]
        intrinsics[:, :, 0] *= wr
        intrinsics[:, :, 2] *= wr
        intrinsics[:, :, 1] *= hr
        intrinsics[:, :, 3] *= hr
        images = resize(images, img_size, False)
        disps = resize(disps.unsqueeze(2), img_size, False).squeeze(2)
        gt_masks = resize(gt_masks, img_size, True)
        gt_vals = resize(gt_vals, img_size, True)

        Gs = SE3(poses)
        disp0 = torch.ones_like(disps[:, :, 3::8, 3::8]) # 1x2x48x156

        for _ in range(1):
            poses_est, disps_est, residuals, full_flows, masks = \
                model(Gs, images, disp0, intrinsics/8,
                    graph, num_steps=15, fixedp=2,
                    ret_flow=True, downsample=True)
            Gs = poses_est[-1]
            disp0 = disps_est[-1][:, :, 3::8, 3::8]

        coords1, val0 = projective_transform(
            poses_est[-1], disps_est[-1], intrinsics, ii, jj)

        cam_flow = (coords1-coords0)[0, 0]
        full_flow = (upsample_inter(full_flows[-1]*8))[0, 0]
        resd = full_flow-cam_flow
        mask = (masks[-1][0, 0].mean(-1, keepdim=True) >= 0.5).float()

        img_id = dic[scene_id] + '_' + file1[0].rsplit('_')[-1][:-4]
        print(img_id)

        full_flow_np = full_flow.detach().cpu().numpy()
        full_flow_1 = full_flow_np*gt_vals[0,0].cpu().numpy()

        if save_npy:
            full_flow_1 = cv2.resize(full_flow_1, (375, 1242))
            np.save(osp.join(full_flow_dir, img_id +'.npy'), full_flow_1)
        
            depth = disps_est[-1][0, 0].detach().cpu().numpy()
            np.save(osp.join(depth_dir, img_id +'.npy'), depth)

    print('Finished building %d img with flows' % i_batch)

depth = disps_est[-1][0, 1].detach().cpu().numpy()
int_id = int(img_id[-2]) * 10 + int(img_id[-1]) + 1
img_id = img_id[:-2] + str(int_id)
np.save(osp.join(depth_dir, img_id +'.npy'), depth)
print(img_id)