import sys
sys.path.append('VO_Module/droid_slam')
from droid import Droid

import glob
import torch.nn.functional as F
import argparse
import time
import os
import cv2
import lietorch
import torch
import numpy as np
from tqdm import tqdm

from panopticapi.utils import  rgb2id, id2rgb
import PIL.Image as Image

def image_stream(datapath, image_size=[240, 808], mode='train', args=None):  # -------------------------------------------
    """ image generator """
    fx, fy, cx, cy = 725.0087, 725.0087, 620.5, 187

    split = {
        'train': 'clone',
        'val': '15-deg-left',
        'test': '30-deg-right'
    }
    images = np.array(sorted(glob.glob(os.path.join(datapath, split[mode], 'frames/rgb/Camera_0/*.jpg'))))
    segments_list = sorted(glob.glob(os.path.join(datapath, split[mode], 'panFPN_segm/*.png'))) 
    
    images_list = images.tolist() 
    print("-----test images :", len(images_list))
    segm = None
    for t, imfile in enumerate(images_list):
        image = cv2.imread(imfile)

        h0, w0, _ = image.shape
        h1, w1 = image_size[0], image_size[1]

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1 % 8, :w1-w1 % 8]
        image = torch.as_tensor(image).int().permute(2, 0, 1)

        if args.segm_filter:
            segm = rgb2id(np.array(Image.open(segments_list[t])))
            segm = torch.as_tensor(segm).unsqueeze(0).unsqueeze(0).float()
            segm = F.interpolate(segm, size=(h1,w1))
            segm = segm[:h1-h1 % 8, :w1-w1 % 8]
            segm = F.interpolate(segm, scale_factor=1/8,
                            recompute_scale_factor=True).int()

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0:2] *= (w1 / w0)
        intrinsics[2:4] *= (h1 / h0)
        
        yield t, image, intrinsics , segm

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath')
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", default="checkpoints/vkitti2_dy_train_semiv4_080000.pth")
    parser.add_argument("--buffer", type=int, default=1024)
    parser.add_argument("--image_size", default=[240, 808])
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--use_aff_bri", type=bool, default=False)

    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--filter_thresh", type=float, default=1.75)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=2.25)
    parser.add_argument("--frontend_thresh", type=float, default=12.0)
    parser.add_argument("--frontend_window", type=int, default=25)
    # 2
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=15.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)

    parser.add_argument("--segm_filter", type=bool, default=False, help="if filter weight with segmentaion in factory graph")
    parser.add_argument("--thresh", type=float, default=0.8)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = init()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(':')[-1]
    torch.multiprocessing.set_start_method('spawn')

    print("Running evaluation on {}".format(args.datapath))
    print(args)

    if args.datapath[-2:] == "20":
        args.thresh = 0.9

    droid = Droid(args)
    time.sleep(5)

    print("segm_filter: ",args.segm_filter)

    for (t, image, intrinsics, segm) in tqdm(image_stream(args.datapath, mode='val', args=args)):
        droid.track(t, image, intrinsics=intrinsics, segments=segm)

    print("video frames: ", droid.video.counter.value)
    traj_est = droid.terminate(image_stream(args.datapath, mode='val', args=args), need_inv=True)

    ### run evaluation ###
    print("#"*20 + " Results...")

    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core.trajectory import PosePath3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation

    def read_vkitti2_poses_file(file_path, args) -> PosePath3D:
        """
        parses pose file in Virtual KITTI 2 format (first 3 rows of SE(3) matrix per line)
        :param file_path: the trajectory file path (or file handle)
        :return: trajectory.PosePath3D
        """
        raw_mat = np.loadtxt(file_path, delimiter=' ', skiprows=1)[::2, 2:]
        error_msg = ("Virtual KITTI 2 pose files must have 16 entries per row "
                     "and no trailing delimiter at the end of the rows (space)")
        if raw_mat is None or (len(raw_mat) > 0 and len(raw_mat[0]) != 16):
            raise file_interface.FileInterfaceException(error_msg)
        try:
            mat = np.array(raw_mat).astype(float)
        except ValueError:
            raise file_interface.FileInterfaceException(error_msg)
        poses = [np.linalg.inv(np.array([[r[0], r[1], r[2], r[3]],
                                         [r[4], r[5], r[6], r[7]],
                                         [r[8], r[9], r[10], r[11]],
                                         [r[12], r[13], r[14], r[15]]])) for r in mat]

        # yapf: enable
        if not hasattr(file_path, 'read'):  # if not file handle
            print("Loaded {} poses from: {}".format(len(poses), file_path))
        return PosePath3D(poses_se3=poses)

    gt_file = os.path.join(args.datapath, '15-deg-left/extrinsic.txt')
    traj_ref = read_vkitti2_poses_file(gt_file, args)

    traj_est = PosePath3D(
        positions_xyz=traj_est[:, :3],
        orientations_quat_wxyz=traj_est[:, 3:])
    
    root = 'shared_data/traj/'
    root = os.path.join(root, args.datapath.rsplit('/')[-1])
    root = os.path.join(root, '15-deg-left')
    if not os.path.isdir(root):
        os.makedirs(root)
    est_file = os.path.join(root,  'pvo_traj.txt')
    print(est_file)

    file_interface.write_kitti_poses_file(est_file, traj_est)
    result = main_ape.ape(traj_ref, traj_est, est_name='traj',
                          pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
    print(result)
    