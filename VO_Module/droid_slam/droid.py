import torch
import torch.nn.functional as F
import lietorch
import numpy as np

from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process

import lietorch

from lietorch import SE3

class Droid:
    def __init__(self, args):
        super(Droid, self).__init__()
        self.args = args
        self.load_weights(args.weights, args.use_aff_bri)
        self.disable_vis = args.disable_vis

        # store images, depth, poses, intrinsics (shared between processes)
        print( "use segment filter:",args.segm_filter)
        self.video = DepthVideo(args.image_size, args.buffer, args.device, args.segm_filter, args.thresh)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(
            self.net, self.video, thresh=args.filter_thresh, device=args.device)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)

        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        # visualizer
        if not self.disable_vis:
            from visualization import droid_visualization
            self.visualizer = Process(
                target=droid_visualization, args=(self.video, args.device))
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(
            self.net, self.video, args.device)

    def load_weights(self, weights, use_aff_bri=False):
        """ load trained model weights """
        self.net = DroidNet(use_aff_bri)
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights, self.args.device).items()])

        self.net.load_state_dict(state_dict)
        self.net.to(self.args.device).eval()

    def track(self, tstamp, image, depth=None, intrinsics=None, segments=None):
        """ main thread - update map """

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(tstamp, image, depth, intrinsics, segments)

            # local bundle adjustment
            self.frontend()

            # global bundle adjustment
            # self.backend()
    
    def terminate(self, stream=None, need_inv=True):
        """ terminate the visualization process, return poses [t, q] """

        del self.frontend

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(7)

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(12)

        camera_trajectory = self.traj_filler(stream)
        if need_inv:
            return camera_trajectory.inv().data.cpu().numpy()
        else:
            # for vkitti2(already w2c)
            return camera_trajectory.data.cpu().numpy()

    def get_traj(self):
        Gs = SE3(self.video.poses[:self.video.counter.value])
        return lietorch.cat([Gs], 0).data.cpu().numpy()
    
    def get_depth(self):
        depth = upsample_inter(self.video.disps[:self.video.counter.value].unsqueeze(0).unsqueeze(4)).squeeze(4).squeeze(0)
        return depth

    def get_flow(self):
        flow =  upsample_inter(self.video.full_flow[:self.video.counter.value].unsqueeze(0)*8)
        return flow

def upsample_inter(mask):
    batch, num, ht, wd, dim = mask.shape
    mask = mask.permute(0, 1, 4, 2, 3).contiguous()
    mask = mask.view(batch*num, dim, ht, wd)
    mask = F.interpolate(mask, scale_factor=8, mode='bilinear',
                         align_corners=True, recompute_scale_factor=True)
    mask = mask.permute(0, 2, 3, 1).contiguous()
    return mask.view(batch, num, 8*ht, 8*wd, dim)
