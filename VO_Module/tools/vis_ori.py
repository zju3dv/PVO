import sys
sys.path.append('droid_slam')

import torch
from torch.utils.data import DataLoader
import glob
import os
import os.path as osp
import cv2
import numpy as np
import pandas as pd
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from collections import OrderedDict
from lietorch import SE3
from geom.projective_ops import projective_transform, coords_grid
from geom.graph_utils import graph_to_edge_list
from droid_net import DroidNet, upsample_inter
from data_readers.factory import dataset_factory
import data_readers.vkitti2


def load_weights(weights, device):
    """ load trained model weights """
    net = DroidNet()
    state_dict = OrderedDict([
        (k.replace("module.", ""), v) for (k, v) in torch.load(weights, device).items()])
    net.load_state_dict(state_dict)
    return net.to(device).eval()


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


def resize_flow(flow, size):
    flow = flow.permute(0, 1, 4, 2, 3).contiguous()
    batch, num, dim, ht, wd = flow.shape
    flow = flow.view(batch*num, dim, ht, wd)
    flow[:, 0, ...] /= wd
    flow[:, 1, ...] /= ht
    flow = F.interpolate(flow, size=size, mode='bilinear', align_corners=True)
    flow[:, 0, ...] *= size[1]
    flow[:, 1, ...] *= size[0]
    flow = flow.permute(0, 2, 3, 1).contiguous()
    return flow.view(batch, num, size[0], size[1], dim)


def occ_warp_img(u0, v0, flow1, dep_uv, ref_img, fix_img, dy_mask):

    # flow = flow.detach().cpu()
    flow1 = flow1.detach().cpu().numpy()
    ref_img = ref_img.permute(1, 2, 0).cpu().numpy()
    fix_img = fix_img.permute(1, 2, 0).cpu().numpy()
    dy_mask = dy_mask.detach().cpu().numpy()

    rows, cols = flow1.shape[:2]

    # grid_uv=torch.from_numpy(np.concatenate([u0/(cols-1),v0/(rows-1)],axis=-1)).view(1,1,-1,2)
    # grid_uv=grid_uv*2-1
    # # sm_flow=F.grid_sample(flow,grid_uv,padding_mode="border", align_corners=True).view(-1,2).numpy()
    # sm_flow1=F.grid_sample(flow1,grid_uv,padding_mode="border", align_corners=True).view(-1,2).numpy()
    # # u,v=u+sm_flow[:,0],v+sm_flow[:,1]
    # u1,v1=u0+sm_flow1[:,0],v0+sm_flow1[:,1]

    u1 = (u0 + flow1[:, :, 0])
    v1 = (v0 + flow1[:, :, 1])

    u0 = u0.flatten()
    v0 = v0.flatten()
    u1 = u1.flatten()
    v1 = v1.flatten()

    # tu = (u + flow[v, u, 0]).astype(np.int32)
    # tv = (v + flow[v, u, 1]).astype(np.int32)
    # u, v = tu, tv

    mm = (0 <= u1).__and__(u1 < cols).__and__(0 <= v1).__and__(v1 < rows)
    # mm2 = (0 <= u).__and__(u < cols).__and__(0 <= v).__and__(v < rows)
    # mm = mm1.__and__(mm2)
    # u, v = u[mm], v[mm]
    u1, v1 = u1[mm], v1[mm]
    u0, v0 = u0[mm], v0[mm]
    dep_uv = dep_uv[mm]

    # encode_uvu1v1 = u0 * 1e14 + v0 * 1e10 + u1 * 1e6 + v1 * 1e2
    encode_uvu1v1 = np.stack([u0, v0, u1, v1], axis=-1)
    df = pd.DataFrame(encode_uvu1v1, index=dep_uv).sort_index(ascending=False)
    new_encode_uvu1v1 = df.to_numpy()

    # dic = dict(zip(encode_uvu1v1, dep_uv))
    # ndic = np.array(
    #     sorted(dic.items(), key=lambda item: item[1], reverse=True))
    # new_encode_uvu1v1 = ndic[:, 0]

    # u0 = (new_encode_uvu1v1 // 1e14).astype(np.int32)
    # v0 = (new_encode_uvu1v1 % 1e14 // 1e10).astype(np.int32)
    # u1 = (new_encode_uvu1v1 % 1e10 // 1e6).astype(np.int32)
    # v1 = (new_encode_uvu1v1 % 1e6 // 1e2).astype(np.int32)

    u0, v0 = new_encode_uvu1v1[:, 0], new_encode_uvu1v1[:, 1]
    u1, v1 = new_encode_uvu1v1[:, 2], new_encode_uvu1v1[:, 3]

    # grid_uv0 = torch.from_numpy(np.concatenate(
    #     [u0/(cols-1), v0/(rows-1)], axis=-1)).view(1, 1, -1, 2)
    # grid_uv0 = grid_uv0*2-1
    # ref_pixs = F.grid_sample(ref_img.unsqueeze(
    #     0).double(), grid_uv0, padding_mode="border", align_corners=True).view(-1, 3).numpy()

    u1 = np.clip(np.around(u1), 0, cols-1).astype(np.int32)
    v1 = np.clip(np.around(v1), 0, rows-1).astype(np.int32)
    warp_img = np.ones([rows, cols, 3])*255
    # warp_img[v1, u1] = ref_pixs
    warp_img[v1, u1] = ref_img[v0.astype(np.int32), u0.astype(np.int32)]

    fix_index = (warp_img.mean(axis=-1, keepdims=True)
                 >= 255).__and__(dy_mask < 1)
    fix_index[:rows//3, :, :] = True
    fix_index = np.concatenate([fix_index, fix_index, fix_index], axis=-1)
    warp_img[fix_index] = fix_img[fix_index]

    return warp_img


scene_id = 'Scene02'
img_size = [368, 1240]

max_fac = 5
speed_fac = []
for f in np.linspace(1, max_fac, 10):
    speed_fac.append(f)
for f in np.linspace(1, max_fac, 10)[::-1]:
    speed_fac.append(f)
speed_fac.append(0)
for f in np.linspace(1, max_fac, 10):
    speed_fac.append(-f)
for f in np.linspace(1, max_fac, 10)[::-1]:
    speed_fac.append(-f)
speed_fac.append(0)

i_start = 158
i_end = i_start+1
test_file = 'droid_slam/data_readers/vkitti2_test.txt'
flow_label = True

f = open(test_file)
data_readers.vkitti2.test_split = f.read().split()
f.close()

ori_img_dir = osp.join('tools/edit-demo/results', scene_id, 'ori_img')
warp_img_dir = osp.join('tools/edit-demo/results', scene_id, 'warp_img')
if not osp.exists(ori_img_dir):
    os.makedirs(ori_img_dir)
if not osp.exists(warp_img_dir):
    os.makedirs(warp_img_dir)

db = dataset_factory(['vkitti2'], datapath='datasets/Virtual_KITTI2', do_aug=False,
                     n_frames=2, flow_label=flow_label, aug_graph=False, split_mode='val',
                     need_inv=False, build_mask=False, crop_size=img_size, mode='semisup', rebuild=True)
train_loader = DataLoader(db, batch_size=1, num_workers=2)

weights_file = "checkpoints/vkitti2_dy_train_semiv4_080000.pth"
device = 'cuda:0'
torch.cuda.set_device(int(device.split(':')[-1]))

graph = OrderedDict()
graph[0] = [1]
graph[1] = [0]
ii, jj, _ = graph_to_edge_list(graph)

model = load_weights(weights_file, device)
coords0 = coords_grid(img_size[0], img_size[1], device=device)

v = np.arange(img_size[0])
v = v.reshape(img_size[0], 1)
v = np.repeat(v, img_size[1], axis=1)
u = np.arange(img_size[1])
u = np.tile(u, (img_size[0], 1))

v1, u1 = v.copy(), u.copy()
v0, u0 = v.copy(), u.copy()

for i_batch, item in enumerate(train_loader):
    if i_batch < i_start and i_batch >= i_start-20:
        images, _, _, _, _, _ = [x.to('cuda') for x in item]
        images = resize(images, img_size, False)
        ori_img = images[0, 0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        cv2.imwrite(osp.join(ori_img_dir, str(i_batch)+'.png'), ori_img)
        continue
    elif i_batch < i_start-20:
        continue
    elif i_batch == i_end:
        break

    if flow_label:
        images, poses, disps, intrinsics, fo_flows, _ = [ x.to('cuda') for x in item]
        fo_flows = resize_flow(fo_flows, img_size)
    else:
        images, poses, disps, intrinsics, _, _ = [x.to('cuda') for x in item]

    hr, wr = img_size[0]/images.shape[3], img_size[1]/images.shape[4]
    intrinsics[:, :, 0] *= wr
    intrinsics[:, :, 2] *= wr
    intrinsics[:, :, 1] *= hr
    intrinsics[:, :, 3] *= hr
    images = resize(images, img_size, False)
    disps = resize(disps.unsqueeze(2), img_size, False).squeeze(2)

    Gs = SE3(poses)
    disp0 = disps[:, :, 3::8, 3::8]

    for _ in range(1):
        poses_est, disps_est, residuals, full_flows, masks = \
                model(Gs, images, disp0, intrinsics/8,
                         graph, num_steps=15, fixedp=2,
                        ret_flow=True, downsample=True)
        # Gs = poses_est[-1]
        # disp0 = disps_est[-1][:, :, 3::8, 3::8]

    # coords1, val0 = projective_transform(
    #     poses_est[-1], disps_est[-1], intrinsics, ii, jj)
    coords1, val0 = projective_transform(Gs, disps, intrinsics, ii, jj)

    cam_flow = (coords1-coords0)[0, 0]
    if flow_label:
        full_flow = fo_flows[0, 0]
    else:
        full_flow = (upsample_inter(full_flows[-1]*8))[0, 0]

    dy_flow = full_flow-cam_flow
    dy_mask = (masks[-1][0, 0].mean(-1, keepdim=True) < 0.5).float()

    ref_depth = disps[0, 0]
    ref_depth = torch.clamp(1/ref_depth, 1e-3, 80)
    ref_depth = ref_depth.detach().cpu().numpy()
    dep_uv = ref_depth.flatten()
    ref_img = images[0, 0]

    for i, fac in enumerate(speed_fac):
        warp_img = occ_warp_img(u0, v0, dy_flow*fac, dep_uv, ref_img, images[0, 0], dy_mask)
        warp_img = warp_img.astype(np.uint8)
        cv2.imwrite(osp.join(warp_img_dir, str(i_batch)+'_%03d.png' % i), warp_img)
        print('Finished warping %d img in speed factor %f' % (i_batch, fac))

    ori_img = images[0, 0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    cv2.imwrite(osp.join(ori_img_dir, str(i_batch)+'.png'), ori_img)
