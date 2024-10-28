import json
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager
from detectron2.data.datasets.builtin_meta import VKITTI_CATEGORIES
from detectron2.structures import BoxMode
import copy

import numpy as np

def register_others(root):
    register_vkitti_511_val()
    register_vkitti_clone()
    register_vkitti_test()

def _get_vkitti_meta():
    # thing
    thing_ids = [k["trainId"] for k in VKITTI_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in VKITTI_CATEGORIES if k["isthing"] == 1]
    thing_dataset_id_to_contiguous_id = {k: i for i,k in enumerate(thing_ids)} 
    thing_classes = [k["name"] for k in VKITTI_CATEGORIES if k["isthing"] == 1]

    # stuff
    stuff_ids = [k["trainId"] for k in VKITTI_CATEGORIES if k["isthing"] == 0]
    stuff_colors = [k["color"] for k in VKITTI_CATEGORIES if k["isthing"] == 0]
    stuff_dataset_id_to_contiguous_id = {k: i+1 for  i,k in enumerate(stuff_ids)} 
    stuff_classes = [k["name"] for k in VKITTI_CATEGORIES if k["isthing"] == 0]

    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def load_vkitti_annotation(ann_file, image_dir, sem_seg_dir, gt_json):
    meta = _get_vkitti_meta()
    with PathManager.open(ann_file, "r") as f:
        jsonobj = json.load(f)
    
    id2file = {}
    for item in jsonobj['images']:
        id2file[item['id']] = item
    img_annotations = {}
    for ann_info in jsonobj['annotations']:
        image_id = ann_info['image_id']
        if image_id not in img_annotations:
            img_annotations[image_id] = []
        for ann in ann_info['segments_info']:
            ann['segmentation']['counts'] = ann['segmentation']['counts'].encode()
            img_annotations[image_id].append(
                {
                    'iscrowd': 0,
                    'bbox': ann['bbox'],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    'category_id': meta['thing_dataset_id_to_contiguous_id'][ann['category_id']], # 0~9
                    'segmentation' : ann['segmentation']
                }
            )
    ret = []
    for k,v in id2file.items():
        filename = v['file_name']
        sem_seg_filename = v['id'] + '.png'
        if k not in img_annotations:
            img_annotations[k] = []
        image_id = v['id']
        ret.append(
            {
                "file_name": os.path.join(image_dir, filename),
                'height': v['height'],
                'width': v['width'],
                "image_id": image_id,
                'annotations':img_annotations[k],
                "sem_seg_file_name": os.path.join(sem_seg_dir, sem_seg_filename),
            }
        )
    return ret
def load_vkitti_flow_eval_annotation(        ann_file, image_dir, sem_seg_dir, gt_json, flow_dir):
    meta = _get_vkitti_meta()
    with PathManager.open(ann_file, "r") as f:
        jsonobj = json.load(f)
    
    id2file = {}
    for item in jsonobj['images']:
        id2file[item['id']] = item
    img_annotations = {}
    for ann_info in jsonobj['annotations']:
        image_id = ann_info['image_id']
        if image_id not in img_annotations:
            img_annotations[image_id] = []
        for ann in ann_info['segments_info']:
            ann['segmentation']['counts'] = ann['segmentation']['counts'].encode()
            img_annotations[image_id].append(
                {
                    'iscrowd': 0,
                    'bbox': ann['bbox'],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    'category_id': meta['thing_dataset_id_to_contiguous_id'][ann['category_id']], # 0~9
                    'segmentation' : ann['segmentation']
                }
            )
    filelist = os.listdir(flow_dir)
    ret = []
    for k,v in id2file.items():
        filename = v['file_name']
        sem_seg_filename = v['id'] + '.png'
        flow_filename = v['id'] + '.npy'
        if k not in img_annotations:
            img_annotations[k] = []
        image_id = v['id']
        ret.append(
            {
                "file_name": os.path.join(image_dir, filename),
                'height': v['height'],
                'width': v['width'],
                "image_id": image_id,
                'annotations':img_annotations[k],
                "sem_seg_file_name": os.path.join(sem_seg_dir, sem_seg_filename),
                # "flow_file_name": os.path.join(flow_dir, sem_seg_filename) if sem_seg_filename in filelist else None,
                "flow_file_name": os.path.join(flow_dir, flow_filename)  if flow_filename in filelist else None ,
            }
        )
    return ret
def load_vkitti_flow_depth_eval_annotation(  ann_file, image_dir, sem_seg_dir, gt_json, flow_dir, depth_dir):
    meta = _get_vkitti_meta()
    with PathManager.open(ann_file, "r") as f:
        jsonobj = json.load(f)
    
    id2file = {}
    for item in jsonobj['images']:
        id2file[item['id']] = item
    img_annotations = {}
    for ann_info in jsonobj['annotations']:
        image_id = ann_info['image_id']
        if image_id not in img_annotations:
            img_annotations[image_id] = []
        for ann in ann_info['segments_info']:
            ann['segmentation']['counts'] = ann['segmentation']['counts'].encode()
            img_annotations[image_id].append(
                {
                    'iscrowd': 0,
                    'bbox': ann['bbox'],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    'category_id': meta['thing_dataset_id_to_contiguous_id'][ann['category_id']], # 0~9
                    'segmentation' : ann['segmentation']
                }
            )
    filelist = os.listdir(flow_dir)
    ret = []
    for k,v in id2file.items():
        filename = v['file_name']
        sem_seg_filename = v['id'] + '.png'
        if k not in img_annotations:
            img_annotations[k] = []
        image_id = v['id']
        ret.append(
            {
                "file_name": os.path.join(image_dir, filename),
                'height': v['height'],
                'width': v['width'],
                "image_id": image_id,
                'annotations':img_annotations[k],
                "sem_seg_file_name": os.path.join(sem_seg_dir, sem_seg_filename),
                "flow_file_name": os.path.join(flow_dir, sem_seg_filename) if sem_seg_filename in filelist else None,
                "depth_file_name": os.path.join(depth_dir, sem_seg_filename),
            }
        )
    return ret

_VKITTI_511_VAL = {
    "vkitti_511_val": (  # on clone 5:1:1  train 5, val 1
        "datasets/Virtual_KITTI2/ALL_clone/img", # img_dir
        "datasets/Virtual_KITTI2/ALL_clone/vkitti_instances_511_val.json", # ann json
        "datasets/Virtual_KITTI2/ALL_clone/stuff_labelTrainIds", # sem_seg_file
        "datasets/Virtual_KITTI2/ALL_clone/panoptic_gt_id",   # gt_dir
        "datasets/Virtual_KITTI2/ALL_clone/vkitti_panoptic_gt_511_val.json", # gt json
    ),
}
def register_vkitti_511_val():
    meta = _get_vkitti_meta()
    for key, (image_dir, ann_json, sem_seg_dir, pan_gt_dir, gt_json) in _VKITTI_511_VAL.items():
        # flow_dir = "shared_data/full_flow"                       # full_flow_dir
        # depth_dir = "shared_data/depth"                          # pred_depth_dir

        flow_dir = "shared_data/full_flow"
        DatasetCatalog.register(
            key, 
            lambda:load_vkitti_flow_eval_annotation(ann_json, image_dir, sem_seg_dir, gt_json, flow_dir),
            # lambda:load_vkitti_flow_depth_eval_annotation(ann_json, image_dir, sem_seg_dir, gt_json, flow_dir, depth_dir),
        )
        MetadataCatalog.get(key).set(
            panoptic_root=pan_gt_dir,
            image_root=image_dir,
            panoptic_json=gt_json,
            sem_seg_root=sem_seg_dir,
            ann_json=ann_json,
            evaluator_type="coco_panoptic_seg",
            ignore_label=255,
            **meta,
        )

_VKITTI_clone = {
    "vkitti_clone": ( # vo 用 15-deg-left 生成该视角下的分割集
        "datasets/Virtual_KITTI2/ALL_clone/img", # img_dir
        "datasets/Virtual_KITTI2/vkitti_instances_clone.json", # ann json
        "datasets/Virtual_KITTI2/ALL_clone/stuff_labelTrainIds", # sem_seg_file
        "datasets/Virtual_KITTI2/ALL_clone/panoptic_gt_id",   # gt_dir
        "datasets/Virtual_KITTI2/vkitti_panoptic_gt_clone.json", # gt json
    ),
}
def register_vkitti_clone():
    meta = _get_vkitti_meta()
    for key, (image_dir, ann_json, sem_seg_dir, pan_gt_dir, gt_json) in _VKITTI_clone.items():
        DatasetCatalog.register(
            key, 
            lambda:load_vkitti_annotation(ann_json, image_dir, sem_seg_dir, gt_json),
        )
        MetadataCatalog.get(key).set(
            panoptic_root=pan_gt_dir,
            image_root=image_dir,
            panoptic_json=gt_json,
            sem_seg_root=sem_seg_dir,
            ann_json=ann_json,
            evaluator_type="coco_panoptic_seg",
            ignore_label=255,
            **meta,
        )

_VKITTI_test = {
    "vkitti_test": ( # vo 用 15-deg-left 生成该视角下的分割集
        "datasets/Virtual_KITTI2/ALL_15-deg-left/img", # img_dir
        "datasets/Virtual_KITTI2/vkitti_instances_test.json", # ann json
        "datasets/Virtual_KITTI2/ALL_15-deg-left/stuff_labelTrainIds", # sem_seg_file
        "datasets/Virtual_KITTI2/ALL_15-deg-left/panoptic_gt_id",   # gt_dir
        "datasets/Virtual_KITTI2/vkitti_panoptic_gt_test.json", # gt json
    ),
}
def register_vkitti_test():
    meta = _get_vkitti_meta()
    for key, (image_dir, ann_json, sem_seg_dir, pan_gt_dir, gt_json) in _VKITTI_test.items():
        DatasetCatalog.register(
            key, 
            lambda:load_vkitti_annotation(ann_json, image_dir, sem_seg_dir, gt_json),
        )
        MetadataCatalog.get(key).set(
            panoptic_root=pan_gt_dir,
            image_root=image_dir,
            panoptic_json=gt_json,
            sem_seg_root=sem_seg_dir,
            ann_json=ann_json,
            evaluator_type="coco_panoptic_seg",
            ignore_label=255,
            **meta,
        )