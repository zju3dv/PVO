
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import sys
import csv
import os
import cv2
import math
import random
import json
import pickle
import os.path as osp
from collections import OrderedDict
from lietorch import SE3

from .augmentation import RGBDAugmentor
from .rgbd_utils import *

from geom.projective_ops import projective_transform, coords_grid
from geom.graph_utils import graph_to_edge_list


class RGBDDataset(data.Dataset):
    def __init__(self, name, datapath, n_frames=4, crop_size=[384, 512],
                 fmin=8.0, fmax=75.0, do_aug=True, flow_label=False, aug_graph=False,
                 need_inv=True, build_mask=False, mode='sup', rebuild=False):
        """ Base class for RGBD dataset """
        self.aug = None
        self.root = datapath
        self.name = name
        self.mode = mode

        self.n_frames = n_frames
        self.fmin = fmin  # exclude very easy examples
        self.fmax = fmax  # exclude very hard examples
        self.flow_label = flow_label
        self.aug_graph = aug_graph
        self.need_inv = need_inv
        self.build_mask = build_mask
        self.only_pose = False
        if do_aug:
            self.aug = RGBDAugmentor(crop_size=crop_size)

        # building dataset is expensive, cache so only needs to be performed once
        cur_path = osp.dirname(osp.abspath(__file__))
        if not os.path.isdir(osp.join(cur_path, 'cache')):
            os.mkdir(osp.join(cur_path, 'cache'))

        cache_path = osp.join(cur_path, 'cache', '{}.pickle'.format(self.name))

        print("cache_path: ",cache_path)
        if osp.isfile(cache_path) and rebuild==False:
            scene_info = pickle.load(open(cache_path, 'rb'))[0]
        else:
            scene_info = self._build_dataset()
            with open(cache_path, 'wb') as cachefile:
                pickle.dump((scene_info,), cachefile)

        self.scene_info = scene_info
        self._build_dataset_index()

        self.has_segm = False

    def _build_dataset_index(self):
        self.dataset_index = []
        for scene in self.scene_info:
            # if not self.__class__.is_test_scene(scene):
            if not self.is_test_scene(scene):
                if self.aug_graph:
                    graph = self.scene_info[scene]['graph']
                    for i in graph:
                        if len(graph[i][0]) > self.n_frames:
                            self.dataset_index.append((scene, i))
                else:
                    for i in range(len(self.scene_info[scene]['images'])-self.n_frames+1):
                        self.dataset_index.append((scene, i))
            else:
                print("Reserving {} for validation".format(scene))

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        return np.load(depth_file)

    def build_frame_graph(self, poses, depths, intrinsics, f=16, max_flow=256):
        """ compute optical flow distance between all pairs of frames """
        def read_disp(fn):
            depth = self.__class__.depth_read(fn)[f//2::f, f//2::f]
            depth[depth < 0.01] = np.mean(depth)
            return 1.0 / depth

        poses = np.array(poses)
        intrinsics = np.array(intrinsics) / f

        disps = np.stack(list(map(read_disp, depths)), 0)
        # need_inv = False for vkitti2
        d = f * \
            compute_distance_matrix_flow(
                poses, disps, intrinsics, need_inv=self.need_inv)

        graph = {}
        for i in range(d.shape[0]):
            j, = np.where(d[i] < max_flow)
            graph[i] = (j, d[i, j])

        return graph

    def __getitem__(self, index):
        """ return training video """

        index = index % len(self.dataset_index)
        scene_id, ix = self.dataset_index[index]

        if self.aug_graph:
            frame_graph = self.scene_info[scene_id]['graph']
        if self.flow_label:
            fo_flow_list = self.scene_info[scene_id]['fo_flows']
            ba_flow_list = self.scene_info[scene_id]['ba_flows']
            if 'segments' in self.scene_info[scene_id]:
                self.has_segm = True
                segments_list = self.scene_info[scene_id]['segments']
        else:
            if not self.only_pose:
                dymask_list = self.scene_info[scene_id]['dymasks']
                segments_list = self.scene_info[scene_id]['segments']
                self.has_segm = True

        images_list = self.scene_info[scene_id]['images']
        if not self.only_pose:
            depths_list = self.scene_info[scene_id]['depths']
        poses_list = self.scene_info[scene_id]['poses']
        intrinsics_list = self.scene_info[scene_id]['intrinsics']

        if self.aug_graph:
            inds = [ix]
            while len(inds) < self.n_frames:
                # get other frames within flow threshold
                k = (frame_graph[ix][1] > self.fmin) & (
                    frame_graph[ix][1] < self.fmax)
                frames = frame_graph[ix][0][k]

                # prefer frames forward in time
                if np.count_nonzero(frames[frames > ix]):
                    ix = np.random.choice(frames[frames > ix])

                elif np.count_nonzero(frames):
                    ix = np.random.choice(frames)

                inds += [ix]
        else:
            inds = []
            for i in range(self.n_frames):
                inds.append(ix + i)

        images, depths, poses, intrinsics = [], [], [], []
        fo_flows, ba_flows = [], []
        fo_vals, ba_vals = [], []
        gt_masks, gt_vals = [], []
        segments = []
        filename_list = []

        for num, i in enumerate(inds):
            images.append(self.__class__.image_read(images_list[i]))
            if not self.only_pose:
                depths.append(self.__class__.depth_read(depths_list[i]))
            poses.append(poses_list[i])
            intrinsics.append(intrinsics_list[i])
            filename_list.append(images_list[i].rsplit('/')[-1])

            if self.flow_label and num < len(inds)-1:
                # 每组flow比图片少一项
                f, v = self.__class__.flow_read(fo_flow_list[i])
                fo_flows.append(f)
                fo_vals.append(v)
                f, v = self.__class__.flow_read(ba_flow_list[i])
                ba_flows.append(f)
                ba_vals.append(v)
                if self.has_segm:
                    seg = self.__class__.segment_read(segments_list[i])
                    segments.append(seg)
            if not self.flow_label :
                if not self.only_pose:
                    mask, v = self.__class__.dymask_read(dymask_list[i])
                    gt_masks.append(mask)
                    gt_vals.append(v)
                    seg = self.__class__.segment_read(segments_list[i])
                    segments.append(seg)

        images = np.stack(images).astype(np.float32)
        if not self.only_pose:
            depths = np.stack(depths).astype(np.float32)
        poses = np.stack(poses).astype(np.float32)
        intrinsics = np.stack(intrinsics).astype(np.float32)

        if self.flow_label:
            fo_flows = np.stack(fo_flows).astype(np.float32)
            ba_flows = np.stack(ba_flows).astype(np.float32)
            fo_flows = torch.from_numpy(fo_flows)
            ba_flows = torch.from_numpy(ba_flows)
            fo_vals = np.stack(fo_vals).astype(np.float32)
            ba_vals = np.stack(ba_vals).astype(np.float32)
            fo_vals = torch.from_numpy(fo_vals)
            ba_vals = torch.from_numpy(ba_vals)
            if self.has_segm:
                segments = np.stack(segments).astype(np.float32)
                segments = torch.from_numpy(segments)
        else:
            if not self.only_pose:
                gt_masks = np.stack(gt_masks).astype(np.float32)
                gt_masks = torch.from_numpy(gt_masks)
                gt_vals = np.stack(gt_vals).astype(np.float32)
                gt_vals = torch.from_numpy(gt_vals)
                segments = np.stack(segments).astype(np.float32)
                segments = torch.from_numpy(segments)

        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2)

        if not self.only_pose:
            disps = torch.from_numpy(1.0 / depths)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        if self.only_pose:
            return filename_list, images, poses, intrinsics

        if self.aug is not None:
            if self.flow_label:
                fo_masks, ba_masks = self.build_motion_masks(poses, disps, intrinsics,
                                                             fo_flows, ba_flows)
                # Only use when build dymasks' labels
                if self.build_mask:
                    return fo_masks.unsqueeze(-1), ba_masks.unsqueeze(-1), \
                        fo_vals.unsqueeze(-1), ba_vals.unsqueeze(-1)

                images, poses, disps, intrinsics, \
                    fo_flows, ba_flows, fo_masks, ba_masks = self.aug(images, poses, disps, intrinsics,
                                                                      fo_flows, fo_vals, ba_flows, ba_vals,
                                                                      fo_masks, ba_masks)
            else:
                images, poses, disps, intrinsics, \
                    gt_masks, gt_vals, segments = self.aug(images, poses, disps, intrinsics,
                                                 gt_masks=gt_masks, gt_vals=gt_vals, 
                                                 segments=segments)
        if len(disps[disps > 0.01]) > 0:
            s = disps[disps > 0.01].mean()
            disps = disps / s
            poses[..., :3] *= s
        
        if self.flow_label:
            if self.has_segm:
                return images, poses, disps, intrinsics, fo_flows, ba_flows, segments
            else:
                return images, poses, disps, intrinsics, fo_flows, ba_flows
        else:
            gt_masks, gt_vals = gt_masks.unsqueeze(-1), gt_vals.unsqueeze(-1)
            if self.mode == 'sup':
                return images, poses, disps, intrinsics, gt_masks, gt_vals
            elif self.mode == 'semisup':
                return filename_list, images, poses, disps, intrinsics, gt_masks, gt_vals, segments
            elif self.mode == 'unsup':
                return images, poses, disps, intrinsics
            else:
                raise Exception('ERROR: Unknown mode!')

    def __len__(self):
        return len(self.dataset_index)

    def __imul__(self, x):
        self.dataset_index *= x
        return self

    def build_motion_masks(self, poses, disps, intrinsics, fo_flows, ba_flows, thresh=0.5):
        N, ht, wd = poses.shape[0], fo_flows.shape[1], fo_flows.shape[2]
        graph = OrderedDict()
        for i in range(N):
            graph[i] = [j for j in range(N) if abs(i-j) == 1]
        ii, jj, _ = graph_to_edge_list(graph)

        poses, disps = poses.unsqueeze(0), disps.unsqueeze(0)
        intrinsics = intrinsics.unsqueeze(0)
        fo_flows, ba_flows = fo_flows.unsqueeze(0), ba_flows.unsqueeze(0)

        # use w2c for vkitti2
        poses = SE3(poses)

        coords_cam, _ = projective_transform(poses, disps, intrinsics, ii, jj)
        cam_flows = coords_cam - coords_grid(ht, wd, device=poses.device)

        fo_masks = ((cam_flows[:, 0::2, ...] -
                    fo_flows[..., 0:2]).norm(dim=-1) <= thresh).float()
        ba_masks = ((cam_flows[:, 1::2, ...] -
                    ba_flows[..., 0:2]).norm(dim=-1) <= thresh).float()

        return fo_masks.squeeze(0), ba_masks.squeeze(0)
