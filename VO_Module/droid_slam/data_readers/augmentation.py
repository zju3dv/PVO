import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F


class RGBDAugmentor:
    """ perform augmentation on RGB-D video """

    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.augcolor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.25, hue=0.4/3.14),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor()])

        self.max_scale = 0.25

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        """from RAFT"""
        N, ht, wd = flow.shape[:3]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)
        coords = torch.from_numpy(coords).expand_as(flow)

        coords = coords.reshape(N, -1, 2).float()
        flow = flow.reshape(N, -1, 2).float()
        valid = valid.reshape(N, -1).float()

        coords0 = coords[valid >= 1]
        flow0 = flow[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * torch.tensor([fx, fy])
        flow1 = flow0 * torch.tensor([fx, fy])

        xx = torch.round(coords1[..., 0]).long()
        yy = torch.round(coords1[..., 1]).long()

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = torch.zeros([N, ht1, wd1, 2], dtype=torch.float)
        valid_img = torch.zeros([N, ht1, wd1], dtype=torch.int)

        flow_img[:, yy, xx] = flow1
        valid_img[:, yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, images, depths, poses, intrinsics,
                          fo_flows=None, fo_vals=None, ba_flows=None, ba_vals=None,
                          fo_masks=None, ba_masks=None, gt_masks=None, gt_vals=None,
                          segments=None):
        """ cropping and resizing """
        ht, wd = images.shape[2:]

        max_scale = self.max_scale
        min_scale = np.log2(np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd)))

        scale = 2 ** np.random.uniform(min_scale, max_scale)
        intrinsics = scale * intrinsics
        depths = depths.unsqueeze(dim=1)

        images = F.interpolate(images, scale_factor=scale, mode='bilinear',
                               align_corners=False, recompute_scale_factor=True)

        depths = F.interpolate(depths, scale_factor=scale,
                               recompute_scale_factor=True)

        if fo_flows != None:
            fo_flows, fo_vals = self.resize_sparse_flow_map(
                fo_flows, fo_vals, fx=scale, fy=scale)
            fo_flows = torch.cat([fo_flows, fo_vals.unsqueeze(-1)], dim=-1)
            ba_flows, ba_vals = self.resize_sparse_flow_map(
                ba_flows, ba_vals, fx=scale, fy=scale)
            ba_flows = torch.cat([ba_flows, ba_vals.unsqueeze(-1)], dim=-1)

            fo_masks = fo_masks.unsqueeze(1)
            ba_masks = ba_masks.unsqueeze(1)
            fo_masks = F.interpolate(fo_masks, scale_factor=scale,
                                     recompute_scale_factor=True)
            ba_masks = F.interpolate(ba_masks, scale_factor=scale,
                                     recompute_scale_factor=True)
        else:
            gt_masks = gt_masks.unsqueeze(1)
            gt_vals = gt_vals.unsqueeze(1)
            segments = segments.unsqueeze(1)
            gt_masks = F.interpolate(gt_masks, scale_factor=scale,
                                     recompute_scale_factor=True)
            gt_vals = F.interpolate(gt_vals, scale_factor=scale,
                                    recompute_scale_factor=True)
            segments = F.interpolate(segments, scale_factor=scale,
                                    recompute_scale_factor=True)

        # always perform center crop (TODO: try non-center crops)
        y0 = (images.shape[2] - self.crop_size[0]) // 2
        x0 = (images.shape[3] - self.crop_size[1]) // 2

        intrinsics = intrinsics - torch.tensor([0.0, 0.0, x0, y0])
        images = images[:, :, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        depths = depths[:, :, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        if fo_flows != None:
            fo_flows = fo_flows[:, y0:y0 +
                                self.crop_size[0], x0:x0+self.crop_size[1], :]
            ba_flows = ba_flows[:, y0:y0 +
                                self.crop_size[0], x0:x0+self.crop_size[1], :]
            fo_masks = fo_masks[:, :, y0:y0 +
                                self.crop_size[0], x0:x0+self.crop_size[1]]
            ba_masks = ba_masks[:, :, y0:y0 +
                                self.crop_size[0], x0:x0+self.crop_size[1]]
        else:
            gt_masks = gt_masks[:, :, y0:y0 +
                                self.crop_size[0], x0:x0+self.crop_size[1]]
            gt_vals = gt_vals[:, :, y0:y0 +
                              self.crop_size[0], x0:x0+self.crop_size[1]]
            segments = segments[:, :, y0:y0 +
                              self.crop_size[0], x0:x0+self.crop_size[1]]

        depths = depths.squeeze(dim=1)
        if fo_flows != None:
            fo_masks = fo_masks.squeeze(1)
            ba_masks = ba_masks.squeeze(1)
            return images, poses, depths, intrinsics, fo_flows, ba_flows, fo_masks, ba_masks
        else:
            gt_masks = gt_masks.squeeze(1)
            gt_vals = gt_vals.squeeze(1)
            segments = F.interpolate(segments, scale_factor=1/8,
                                    recompute_scale_factor=True)
            segments = segments.squeeze(1).int()
            return images, poses, depths, intrinsics, gt_masks, gt_vals, segments

    def spatial_transform_only_pose(self, images, poses, intrinsics):
        """ cropping and resizing """
        ht, wd = images.shape[2:]

        max_scale = self.max_scale
        min_scale = np.log2(np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd)))

        scale = 2 ** np.random.uniform(min_scale, max_scale)
        intrinsics = scale * intrinsics

        images = F.interpolate(images, scale_factor=scale, mode='bilinear',
                               align_corners=False, recompute_scale_factor=True)

        # always perform center crop (TODO: try non-center crops)
        y0 = (images.shape[2] - self.crop_size[0]) // 2
        x0 = (images.shape[3] - self.crop_size[1]) // 2

        intrinsics = intrinsics - torch.tensor([0.0, 0.0, x0, y0])
        images = images[:, :, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return images, poses, intrinsics

    def color_transform(self, images):
        """ color jittering """
        num, ch, ht, wd = images.shape
        images = images.permute(1, 2, 3, 0).reshape(ch, ht, wd*num)
        images = 255 * self.augcolor(images[[2, 1, 0]] / 255.0)
        return images[[2, 1, 0]].reshape(ch, ht, wd, num).permute(3, 0, 1, 2).contiguous()

    def __call__(self, images, poses, depths, intrinsics,
                 fo_flows=None, fo_vals=None, ba_flows=None, ba_vals=None,
                 fo_masks=None, ba_masks=None, gt_masks=None, gt_vals=None,
                 segments=None):
        if depths == None:
            images = self.color_transform(images)
            return self.spatial_transform_only_pose(images, poses, intrinsics)
        else:
            images = self.color_transform(images)
            return self.spatial_transform(images, depths, poses, intrinsics,
                                      fo_flows, fo_vals, ba_flows, ba_vals,
                                      fo_masks, ba_masks, gt_masks, gt_vals,
                                      segments)
