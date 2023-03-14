# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import Dict, List
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from detectron2.config import configurable
from detectron2.structures import ImageList
from detectron2.data.datasets.builtin_meta import CITYSCAPES_CATEGORIES, KITTI_STEP_CATEGORIES, VKITTI_CATEGORIES

from ..postprocessing import detector_postprocess, sem_seg_postprocess
from .build import META_ARCH_REGISTRY
from .rcnn import GeneralizedRCNN
from .semantic_seg import build_sem_seg_head

import importlib
from detectron2.aot_modules.layers.loss import CrossEntropyLoss, SoftJaccordLoss
import PIL.Image as Image
import copy
import cv2  
from collections import OrderedDict

__all__ = ["PanopticFPN"]

thing_ids = [k["trainId"] for k in VKITTI_CATEGORIES if k["isthing"] == 1]
thing_id2cat = {i: k for i,k in enumerate(thing_ids)}
stuff_ids = [k["trainId"] for k in VKITTI_CATEGORIES if k["isthing"] == 0]
stuff_id2cat = {i+1: k for  i,k in enumerate(stuff_ids)}


@META_ARCH_REGISTRY.register()
class PanopticFPN(GeneralizedRCNN):
    """
    Implement the paper :paper:`PanopticFPN`.
    """

    @configurable
    def __init__(
        self,
        *,
        sem_seg_head: nn.Module,
        combine_overlap_thresh: float = 0.5,
        combine_stuff_area_thresh: float = 4096,
        combine_instances_score_thresh: float = 0.5,
        fusion,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            sem_seg_head: a module for the semantic segmentation head.
            combine_overlap_thresh: combine masks into one instances if
                they have enough overlap
            combine_stuff_area_thresh: ignore stuff areas smaller than this threshold
            combine_instances_score_thresh: ignore instances whose score is
                smaller than this threshold

        Other arguments are the same as :class:`GeneralizedRCNN`.
        """
        super().__init__(**kwargs)
        self.sem_seg_head = sem_seg_head
        # options when combining instance & semantic outputs
        self.combine_overlap_thresh = combine_overlap_thresh
        self.combine_stuff_area_thresh = combine_stuff_area_thresh
        self.combine_instances_score_thresh = combine_instances_score_thresh
        
        self.flow_is_npy = True
        self.fusion = fusion 
        self.alpha = 1.0
        if self.fusion:
            for param in self.parameters(): 
                param.requires_grad = False

            self.flow_transport = True
            self.flow_depth_transport = False  
            self.depth_proj_op = False

            self.fusion_conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1) 

            self.fx = 725.0087
            self.fy = 725.0087
            self.cx = 620.5
            self.cy = 187
            
            self.vid = None
            self.ref_flow = None
            self.ref_image = None
            self.ref_id = None

            n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print('ACTIVE number of params:', n_parameters)     
        else:
            self.fusion_conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1) 
            for p in self.fusion_conv1.parameters():
                p.requires_grad = False

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update(
            {
                "combine_overlap_thresh": cfg.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH,
                "combine_stuff_area_thresh": cfg.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT,
                "combine_instances_score_thresh": cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH,  # noqa
                "fusion" : cfg.MODEL.PANOPTIC_FPN.FUSION,
            }
        )
        ret["sem_seg_head"] = build_sem_seg_head(cfg, ret["backbone"].output_shape())
        logger = logging.getLogger(__name__)
        if not cfg.MODEL.PANOPTIC_FPN.COMBINE.ENABLED:
            logger.warning(
                "PANOPTIC_FPN.COMBINED.ENABLED is no longer used. "
                " model.inference(do_postprocess=) should be used to toggle postprocessing."
            )
        if cfg.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT != 1.0:
            w = cfg.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT
            logger.warning(
                "PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT should be replaced by weights on each ROI head."
            )

            def update_weight(x):
                if isinstance(x, dict):
                    return {k: v * w for k, v in x.items()}
                else:
                    return x * w

            roi_heads = ret["roi_heads"]
            roi_heads.box_predictor.loss_weight = update_weight(roi_heads.box_predictor.loss_weight)
            roi_heads.mask_head.loss_weight = update_weight(roi_heads.mask_head.loss_weight)
        return ret

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                * "image": Tensor, image in (C, H, W) format.
                * "instances": Instances
                * "sem_seg": semantic segmentation ground truth.
                * Other information that's included in the original dicts, such as:
                  "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "instances": see :meth:`GeneralizedRCNN.forward` for its format.
                * "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
                * "panoptic_seg": See the return value of
                  :func:`combine_semantic_and_instance_outputs` for its format.
        """
        self.batch_size = len(batched_inputs)
        if self.fusion :
            if self.training:  
                return self.train_fusion(batched_inputs)
            else:
                return self.inference_fusion(batched_inputs)

        if not self.training:
            return self.inference(batched_inputs)
        
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        assert "sem_seg" in batched_inputs[0]
        gt_sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]
        gt_sem_seg = ImageList.from_tensors(
            gt_sem_seg, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
        ).tensor
        sem_seg_results, sem_seg_losses = self.sem_seg_head(features, gt_sem_seg)

        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        detector_results, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances
        )

        losses = sem_seg_losses
        losses.update(proposal_losses)
        losses.update(detector_losses)
        return losses
    
    # feature 
    def wrap_in_stage2(self, seq):
        images = [v["image"].to(self.device) for k,v in seq.items()] 
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if self.flow_transport:
            flow = seq['ref']['flow']
            features = self.flow_transport_feature(features, flow)
        elif self.flow_depth_transport:
            flow = seq['ref']['flow']
            depth = seq['ref']['depth']
            if self.depth_proj_op:
                pose = [v["pose_extrinsic"] for k,v in seq.items()]
                depth = self.pose_transport_depth(depth, pose)
            features = self.flow_transport_feature_with_depth(features, flow, depth)

        features = self.fusion_module(features)

        if self.training:
            # gt_sem_seg = [v["sem_seg"].to(self.device) for k,v in seq.items()] 
            gt_sem_seg = [seq['cur']['sem_seg'].to(self.device)]
            gt_sem_seg = ImageList.from_tensors(gt_sem_seg, self.backbone.size_divisibility, self.sem_seg_head.ignore_value).tensor
            sem_seg_results, sem_seg_losses = self.sem_seg_head(features, gt_sem_seg)
        else:
            sem_seg_results, _ = self.sem_seg_head(features, None)
            

        if self.training:
            # gt_instances = [v["instances"].to(self.device) for k,v in seq.items()]
            gt_instances = [seq['cur']['instances'].to(self.device)]
            images = [seq['cur']["image"].to(self.device)] 
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            detector_results, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        else:
            images = [seq['cur']["image"].to(self.device)] 
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
            proposals, _ = self.proposal_generator(images, features, None)
            detector_results, _ = self.roi_heads(images, features, proposals, None)
        

        if self.training:
            return features, sem_seg_results, detector_results[0], sem_seg_losses, detector_losses, proposal_losses,proposals[0]
        else:
            return sem_seg_results, detector_results, images

    def train_fusion(self, batched_inputs):
        for seq in batched_inputs: # ref, cur
            wfeat2, sem2, inst2, sem_loss2, inst_loss2, pro_loss2, pro2 = self.wrap_in_stage2(seq)
            losses = {
                        'loss_sem_seg': sem_loss2['loss_sem_seg']
                    }
            for k in inst_loss2.keys():
                losses.update({k: inst_loss2[k]})  
            for k in pro_loss2.keys():
                losses.update({k: pro_loss2[k]}) 
        return losses
    
    def inference_fusion(self, batched_inputs, do_postprocess=True):
        cur_vid = batched_inputs[0]['image_id'][:4]
        if cur_vid != self.vid: 
            self.vid = cur_vid
            images = [batched_inputs[0]["image"].to(self.device)] 
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
            features = self.backbone(images.tensor)
            sem_seg_results, sem_seg_losses = self.sem_seg_head(features, None)
            proposals, _ = self.proposal_generator(images, features, None)
            detector_results, _ = self.roi_heads(images, features, proposals, None)
        else:
            seq = {'ref': {}, 'cur':{}}
            seq['ref']['flow'] = self.ref_flow
            seq['ref']['image'] = self.ref_image
            seq['ref']['image_id'] = self.ref_id
            seq['cur'] = batched_inputs[0]
            if self.flow_depth_transport:
                seq['ref']['depth'] = self.ref_depth
            if self.depth_proj_op :
                seq['ref']['pose_extrinsic'] = self.ref_pose
            sem_seg_results, detector_results, images = self.wrap_in_stage2(seq)

        if 'flow' in batched_inputs[0]:
            self.ref_flow = batched_inputs[0]['flow']
            self.ref_image = batched_inputs[0]['image']
            self.ref_id = batched_inputs[0]['image_id']
        if 'depth' in batched_inputs[0]:
            self.ref_depth = batched_inputs[0]['depth']
        if self.depth_proj_op:
            self.ref_pose = batched_inputs[0]['pose_extrinsic']
                
        if do_postprocess:
            processed_results = []
            for sem_seg_result, detector_result, input_per_image, image_size in zip(
                sem_seg_results, detector_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                sem_seg_r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
                detector_r = detector_postprocess(detector_result, height, width)

                processed_results.append({"sem_seg": sem_seg_r, "instances": detector_r})

                panoptic_r = combine_semantic_and_instance_outputs(
                    detector_r,
                    sem_seg_r.argmax(dim=0),
                    self.combine_overlap_thresh,
                    self.combine_stuff_area_thresh,
                    self.combine_instances_score_thresh,
                )
                processed_results[-1]["panoptic_seg"] = panoptic_r
            return processed_results
        else:
            return detector_results, sem_seg_results

    def flow_transport_feature(self, features, ori_flow): 
        for k,feat in features.items():   
            ref_flow = self.resize(ori_flow.unsqueeze(0), feat.shape[-2:]).squeeze(0).permute(1,2,0)
            if self.flow_is_npy:
                ref_flow = ref_flow.detach().cpu().numpy().astype(np.uint16)
            else:
                ref_flow = self.read_vkitti_png_flow(ref_flow.detach().cpu().numpy().astype(np.uint16), ori_flow.shape)          
            [rows, cols] = feat.shape[-2:]
    
            mask = torch.zeros(feat.shape[-3:]).to(self.device) 
            v = np.arange(rows)
            v = v.reshape(rows, 1)
            v = np.repeat(v, cols, axis=1)
            u = np.arange(cols)
            u = np.tile(u, (rows, 1))

            u1 = (u + ref_flow[:,:,0]).astype(np.int32) 
            v1 = (v + ref_flow[:,:,1]).astype(np.int32) 

            u = u.flatten()
            v = v.flatten()
            u1 = u1.flatten()
            v1 = v1.flatten()

            mm = (0 <= u1).__and__(u1 < cols).__and__(0 <= v1).__and__(v1 < rows)
            u1 = u1[mm]
            v1 = v1[mm]
            u = u[mm]
            v = v[mm]

            mask[:,v1,u1] = feat[0][:,v,u]
            query_mask = torch.as_tensor(mask) 
            features[k] = torch.cat((feat[1],  self.alpha * query_mask), dim=0).unsqueeze(0)

        return features

    def flow_transport_feature_with_depth(self, features, ori_flow, ori_depth):
        for k,feat in features.items():
            ref_flow = self.resize(ori_flow.unsqueeze(0), feat.shape[-2:]).squeeze(0).permute(1,2,0)
            if self.flow_is_npy:
                ref_flow = ref_flow.detach().cpu().numpy().astype(np.uint16)
            else:
                ref_flow = self.read_vkitti_png_flow(ref_flow.detach().cpu().numpy().astype(np.uint16), ori_flow.shape)            
            ref_depth = self.resize(ori_depth.unsqueeze(0).unsqueeze(0), feat.shape[-2:]).squeeze(0).squeeze(0).cpu().numpy()

            [rows, cols] = feat.shape[-2:]
    
            mask = torch.zeros(feat.shape[-3:]).to(self.device) 
            v = np.arange(rows)
            v = v.reshape(rows, 1)
            v = np.repeat(v, cols, axis=1)
            u = np.arange(cols)
            u = np.tile(u, (rows, 1))

            u1 = (u + ref_flow[:,:,0]).astype(np.int32) 
            v1 = (v + ref_flow[:,:,1]).astype(np.int32) 

            u = u.flatten()
            v = v.flatten()
            u1 = u1.flatten()
            v1 = v1.flatten()
            dep_uv = ref_depth.flatten()

            mm = (0 <= u1).__and__(u1 < cols).__and__(0 <= v1).__and__(v1 < rows)
            u1 = u1[mm]
            v1 = v1[mm]
            u = u[mm]
            v = v[mm]
            dep_uv = dep_uv[mm]

            u1,v1,u,v = self.depth_filter(u1, v1, u, v, dep_uv)

            mask[:,v1,u1] = feat[0][:,v,u]
            query_mask = torch.as_tensor(mask)
            features[k] = torch.cat((feat[1],  self.alpha * query_mask), dim=0).unsqueeze(0)

        return features

    def pose_transport_depth(self, ori_depth, pose):
        [rows, cols] = ori_depth.shape[-2:]
        v = np.repeat(np.arange(rows).reshape(rows, 1), cols, axis=1)
        u = np.tile(np.arange(cols), (rows, 1))

        Zc0 = ori_depth.detach().cpu().numpy()
        Xc0 = (u[:,:] - self.cx) / self.fx * Zc0[:,:]
        Yc0 = (v[:,:] - self.cy) / self.fy * Zc0[:,:]
        Ones = np.ones((rows, cols))
        point_camera_0 = np.array([Xc0, Yc0, Zc0 ,Ones]).reshape(4, -1)  

        extrinsics_inv_0 = np.array(pose[0]).reshape((4, 4))
        extrinsics_0 = np.linalg.inv(extrinsics_inv_0) # cam2wold
        extrinsics_inv_1 = np.array(pose[1]).reshape((4, 4)) # world2cam  # extrinsics_inv Twc: camera to world
           
        relative_pose = np.matmul(extrinsics_inv_1, extrinsics_0)
        point_camera_1 = np.matmul(relative_pose, point_camera_0)

        Xc1, Yc1, Zc1 = point_camera_1[:3, :] 
        depth = torch.as_tensor(Zc1.reshape(rows, cols)).to(self.device)
        return depth

    def fusion_module(self, features):
        for k,v in features.items():
            x = self.fusion_conv1(v)
            features[k] = x
        return features

    def depth_filter(self, u1, v1, u, v, dep_uv):
        encode_uvu1v1 = u * 1e14 + v * 1e10 + u1 * 1e6 + v1 * 1e2
        dic = dict(zip(encode_uvu1v1, dep_uv))
        ndic = np.array(sorted(dic.items(), key=lambda item:item[1], reverse=True))

        if ndic.shape[0] != 0:
            new_encode_uvu1v1 = ndic[:,0]
        else:
            new_encode_uvu1v1 = encode_uvu1v1

        u = (new_encode_uvu1v1 // 1e14).astype(np.int32)
        v = (new_encode_uvu1v1 % 1e14 // 1e10).astype(np.int32)
        u1 = (new_encode_uvu1v1 % 1e10 // 1e6).astype(np.int32)
        v1 = (new_encode_uvu1v1 % 1e6 // 1e2).astype(np.int32)

        return u1, v1, u, v

    def resize(self, mask, output_size):
        mask = F.interpolate(mask,
                            size=output_size,
                            mode="bilinear",
                            align_corners=True)
        return mask

    def read_vkitti_png_flow(self, bgr, ori_shape):
        # bgr = cv2.imread(flow_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        _c, h, w = ori_shape
        assert bgr.dtype == np.uint16 and _c == 3
        # b == invalid flow flag == 0 for sky or other invalid flow
        invalid = bgr[:,:,0] == 0
        # g,r == flow_y,x normalized by height,width and scaled to [0;2**16 – 1]
        out_flow = 2.0 / (2**16 - 1.0) * bgr[:,:,2:0:-1].astype('f4') -1
        out_flow[:,:, 0] *= w - 1
        out_flow[:,:, 1] *= h - 1
        out_flow[invalid] = 0  # or another value (e.g., np.nan)
        return out_flow

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], do_postprocess: bool = True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, see docs in :meth:`forward`.
            Otherwise, returns a (list[Instances], list[Tensor]) that contains
            the raw detector outputs, and raw semantic segmentation outputs.
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        sem_seg_results, sem_seg_losses = self.sem_seg_head(features, None)
        proposals, _ = self.proposal_generator(images, features, None)
        detector_results, _ = self.roi_heads(images, features, proposals, None)

        if do_postprocess:
            processed_results = []
            for sem_seg_result, detector_result, input_per_image, image_size in zip(
                sem_seg_results, detector_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                sem_seg_r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
                detector_r = detector_postprocess(detector_result, height, width)

                processed_results.append({"sem_seg": sem_seg_r, "instances": detector_r})

                panoptic_r = combine_semantic_and_instance_outputs(
                    detector_r,
                    sem_seg_r.argmax(dim=0),
                    self.combine_overlap_thresh,
                    self.combine_stuff_area_thresh,
                    self.combine_instances_score_thresh,
                )
                processed_results[-1]["panoptic_seg"] = panoptic_r
            return processed_results
        else:
            return detector_results, sem_seg_results


def combine_semantic_and_instance_outputs(
    instance_results,
    semantic_results,
    overlap_threshold,
    stuff_area_thresh,
    instances_score_thresh,
):
    """
    Implement a simple combining logic following
    "combine_semantic_and_instance_predictions.py" in panopticapi
    to produce panoptic segmentation outputs.

    Args:
        instance_results: output of :func:`detector_postprocess`.
        semantic_results: an (H, W) tensor, each element is the contiguous semantic
            category id

    Returns:
        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
            Each dict contains keys "id", "category_id", "isthing".
    """
    panoptic_seg = torch.zeros_like(semantic_results, dtype=torch.int32)

    # sort instance outputs by scores
    sorted_inds = torch.argsort(-instance_results.scores)

    current_segment_id = 0
    segments_info = []

    instance_masks = instance_results.pred_masks.to(dtype=torch.bool, device=panoptic_seg.device)

    # Add instances one-by-one, check for overlaps with existing ones
    for inst_id in sorted_inds:
        score = instance_results.scores[inst_id].item()
        if score < instances_score_thresh:
            break
        mask = instance_masks[inst_id]  # H,W
        mask_area = mask.sum().item()

        if mask_area == 0:
            continue

        intersect = (mask > 0) & (panoptic_seg > 0)
        intersect_area = intersect.sum().item()

        if intersect_area * 1.0 / mask_area > overlap_threshold:
            continue

        if intersect_area > 0:
            mask = mask & (panoptic_seg == 0)

        current_segment_id += 1
        cat = instance_results.pred_classes[inst_id].item()
        cat = thing_id2cat[cat]
        instance_id = cat * 10000 + current_segment_id
        panoptic_seg[mask] = instance_id
        segments_info.append(
            {
                "id": instance_id,
                "isthing": True,
                "score": score,
                "category_id": cat,
                "instance_id": inst_id.item(),
            }
        )

    # Add semantic results to remaining empty areas
    semantic_labels = torch.unique(semantic_results).cpu().tolist()
    for semantic_label in semantic_labels:
        if semantic_label == 0:  # 0 is a special "thing" class
            continue
        mask = (semantic_results == semantic_label) & (panoptic_seg == 0)
        mask_area = mask.sum().item()
        if mask_area < stuff_area_thresh:
            continue

        current_segment_id += 1
        # cat = semantic_label
        cat = stuff_id2cat[semantic_label]
        stuff_id = cat * 10000
        panoptic_seg[mask] = stuff_id
        segments_info.append(
            {
                "id": stuff_id,
                "isthing": False,
                "category_id": cat,
                "area": mask_area,
            }
        )

    return panoptic_seg, segments_info
