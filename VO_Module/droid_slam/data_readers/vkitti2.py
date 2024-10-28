
import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp

from scipy.spatial.transform import Rotation as R
from lietorch import SE3
from torch.functional import split
from .base import RGBDDataset
from .stream import RGBDStream

from panopticapi.utils import  rgb2id
import PIL.Image as Image

def rmat_to_quad(mat):
    r = R.from_matrix(mat)
    quat = r.as_quat()
    return quat

class VKitti2(RGBDDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0
    split = {
        'train': 'clone',
        'val': '15-deg-left',
        'test': '30-deg-right'
    }
    scenes = ['Scene18', 'Scene20' , 'Scene06', 'Scene02', 'Scene01']
    print(scenes)
    def __init__(self, split_mode='train', foo=False, scene_id='Scene01',  **kwargs):
        self.split_mode = split_mode
        self.n_frames = 2
        self.foo = foo
        self.test_split = [x for x in VKitti2.scenes if x != scene_id]
        super(VKitti2, self).__init__(name='VKitti2', **kwargs)

    # @staticmethod
    def is_test_scene(self, scene):
        return any(x in scene for x in self.test_split)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building VKitti2 dataset")
        scenes = VKitti2.scenes
        scene_info = {}
        for scene in tqdm(sorted(scenes)):
            scene = osp.join(self.root, scene)
            images = sorted(
                glob.glob(osp.join(scene, VKitti2.split[self.split_mode], 'frames/rgb/Camera_0/*.jpg')))
            depths = sorted(
                glob.glob(osp.join( scene, VKitti2.split[self.split_mode], 'frames/depth/Camera_0/*.png')))

            poses = np.loadtxt(
                osp.join(scene, VKitti2.split[self.split_mode], 'extrinsic.txt'), delimiter=' ', skiprows=1)[::2, 2:]
            
            if self.foo:
                images = np.array(images)
                depths = np.array(depths)
                val_num = images.shape[0] // 7
                train_num = images.shape[0] - val_num - val_num
                print(val_num,'\n',images[train_num], "\n", images[train_num+val_num-1])
                images = images[train_num:train_num + val_num]
                depths = depths[train_num:train_num + val_num]
                poses = poses[train_num:train_num + val_num]
                print(poses.shape)
                images.tolist()
                depths.tolist()

            poses = poses.reshape(-1, 4, 4)  
            r = rmat_to_quad(poses[:, 0:3, 0:3])
            t = poses[:, :3, 3] / VKitti2.DEPTH_SCALE
            poses = np.concatenate((t, r), axis=1)

            intrinsics = [VKitti2.calib_read()] * len(images)
            scene = '/'.join(scene.split('/'))

            # graph of co-visible frames based on flow
            graph = None
            if self.aug_graph:
                graph = self.build_frame_graph(poses, depths, intrinsics)

            if self.flow_label:
                print(scene, self.split_mode, VKitti2.split[self.split_mode])
                fo_flows = sorted(
                    glob.glob(osp.join(scene, VKitti2.split[self.split_mode], 'frames/forwardFlow/Camera_0/*.png')))
                ba_flows = sorted(
                    glob.glob(osp.join(scene, VKitti2.split[self.split_mode], 'frames/backwardFlow/Camera_0/*.png')))
                # segments = sorted(
                #     glob.glob(osp.join(scene,  VKitti2.split[self.split_mode], 'panoptic_gt_id/*.png')))
                segments = sorted(
                    glob.glob(osp.join("datasets/Virtual_KITTI2/ALL_15-deg-left", 'panoptic_gt_id/*.png')))
                scene_info[scene] = {'images': images, 'depths': depths, 'fo_flows': fo_flows,
                                     'ba_flows': ba_flows, 'poses': poses, 'intrinsics': intrinsics, 'segments' : segments}
            else: 
                print(self.split_mode)
                masks = sorted(
                    glob.glob(osp.join(scene, VKitti2.split[self.split_mode], 'frames/dynamicMask_00/Camera_0/*.npy')))
                segments = sorted(
                    glob.glob(osp.join(scene,  VKitti2.split[self.split_mode], 'panFPN_segm/*.png')))

                scene_info[scene] = {'images': images, 'depths': depths, 'dymasks': masks,
                                     'poses': poses, 'intrinsics': intrinsics, 'graph': graph, 'segments': segments, }
            print('segments' in scene_info[scene])
        return scene_info

    @staticmethod
    def calib_read():
        return np.array([725.0087, 725.0087, 620.5, 187])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR |
                           cv2.IMREAD_ANYDEPTH) / (VKitti2.DEPTH_SCALE*100)
        depth[depth == np.nan] = 1.0
        depth[depth == np.inf] = 1.0
        depth[depth == 0] = 1.0
        return depth

    @staticmethod
    def flow_read(flow_file):
        bgr = cv2.imread(flow_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        h, w, _c = bgr.shape
        out_flow = 2.0 / (2**16 - 1.0) * bgr[..., 2:0:-1].astype('f4') - 1
        out_flow[..., 0] *= w - 1
        out_flow[..., 1] *= h - 1
        val = (bgr[..., 0] > 0).astype(np.float32)
        return out_flow, val

    @staticmethod
    def dymask_read(mask_file):
        content = np.load(mask_file)
        return content[..., 0], content[..., 1]

    @staticmethod
    def segment_read(segment_file):
        segment = rgb2id(np.array(Image.open(segment_file)))
        return segment 


class VKitti2Stream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(VKitti2Stream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        self.root = 'datasets/VKitti2'

        scene = osp.join(self.root, self.datapath)
        image_glob = osp.join(scene, 'image_left/*.png')
        images = sorted(glob.glob(image_glob))

        poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')
        poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]

        poses = SE3(torch.as_tensor(poses))
        poses = poses[[0]].inv() * poses
        poses = poses.data.cpu().numpy()

        intrinsic = self.calib_read(self.datapath)
        intrinsics = np.tile(intrinsic[None], (len(images), 1))

        self.images = images[::int(self.frame_rate)]
        self.poses = poses[::int(self.frame_rate)]
        self.intrinsics = intrinsics[::int(self.frame_rate)]

    @staticmethod
    def calib_read(datapath):
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)


class VKitti2TestStream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(VKitti2TestStream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        self.root = 'datasets/mono'
        image_glob = osp.join(self.root, self.datapath, '*.png')
        images = sorted(glob.glob(image_glob))

        poses = np.loadtxt(osp.join(self.root, 'mono_gt',
                           self.datapath + '.txt'), delimiter=' ')
        poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]

        poses = SE3(torch.as_tensor(poses))
        poses = poses[[0]].inv() * poses
        poses = poses.data.cpu().numpy()

        intrinsic = self.calib_read(self.datapath)
        intrinsics = np.tile(intrinsic[None], (len(images), 1))

        self.images = images[::int(self.frame_rate)]
        self.poses = poses[::int(self.frame_rate)]
        self.intrinsics = intrinsics[::int(self.frame_rate)]

    @staticmethod
    def calib_read(datapath):
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)
