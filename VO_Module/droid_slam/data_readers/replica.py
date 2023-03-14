
import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp

from lietorch import SE3
from .base import RGBDDataset
from .stream import RGBDStream

cur_path = osp.dirname(osp.abspath(__file__))
test_split = osp.join(cur_path, 'replica_test.txt')
test_split = open(test_split).read().split()


class Replica(RGBDDataset):

    # scale depths to balance rot & trans
    # Replica: 1.0 or even smaller?
    DEPTH_SCALE = 1.0

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2
        super(Replica, self).__init__(name='Replica', **kwargs)

    @staticmethod 
    def is_test_scene(scene):
        # print(scene, any(x in scene for x in test_split))
        return any(x in scene for x in test_split)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building Replica dataset")

        scene_info = {}
        scenes = glob.glob(osp.join(self.root, '*'))
        for scene in tqdm(sorted(scenes)):
            images = sorted(glob.glob(osp.join(scene, 'image_left/*.jpg')))
            depths = sorted(glob.glob(osp.join(scene, 'depth_left/*.npy')))
            
            poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')
            # tx,ty,tz,qx,qy,qz,qw
            # poses = poses[:, [1, 2, 0, 4, 5, 3, 6]] # 交换是因为NED frame, 修改坐标轴
            poses[:,:3] /= Replica.DEPTH_SCALE
            intrinsics = [Replica.calib_read()] * len(images)

            # graph of co-visible frames based on flow
            graph = self.build_frame_graph(poses, depths, intrinsics)

            scene = '/'.join(scene.split('/'))
            scene_info[scene] = {'images': images, 'depths': depths, 
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

        return scene_info

    @staticmethod
    def calib_read():
        return np.array([600.0, 600.0, 599.5, 339.5])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = np.load(depth_file) / Replica.DEPTH_SCALE
        depth[depth==np.nan] = 1.0
        depth[depth==np.inf] = 1.0
        return depth


class ReplicaStream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(ReplicaStream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        self.root = 'datasets/Replica'

        scene = osp.join(self.root, self.datapath)
        image_glob = osp.join(scene, 'image_left/*.jpg')
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
        return np.array([600.0, 600.0, 599.5, 339.5])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)


class ReplicaTestStream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(ReplicaStream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        self.root = 'datasets/mono'
        image_glob = osp.join(self.root, self.datapath, '*.jpg')
        images = sorted(glob.glob(image_glob))

        poses = np.loadtxt(osp.join(self.root, 'mono_gt', self.datapath + '.txt'), delimiter=' ')
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
        return np.array([600.0, 600.0, 599.5, 339.5])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)