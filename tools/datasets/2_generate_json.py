import shutil
import os
import json
from PIL import Image
import numpy as np
from panopticapi.utils import rgb2id, id2rgb, save_json, IdGenerator
import pycocotools.mask as mask_util
from CATEGORY import categories

root = "datasets/Virtual_KITTI2/"
pan_dir = ["ALL_clone/panoptic_gt_id",
           "ALL_15-deg-left/panoptic_gt_id", ]

# ================================================
# panoptic_gt.json
# ================================================
print("===========\n generate panoptic json and instance annotation \n===========")
pan_json_dir = ['vkitti_panoptic_gt_clone.json',
                'vkitti_panoptic_gt_test.json',]
for pan_path, json_name in zip(pan_dir, pan_json_dir):
    path = os.path.join(root, pan_path)
    filelist = os.listdir(path)
    filelist.sort()

    images = []
    anns = []
    for f in filelist:
        img_id = f[:-4]
        # print(img_id)
        img = np.array(Image.open(os.path.join(path, f)))
        width = img.shape[1]
        height = img.shape[0]
        images.append(
            {
                "id": img_id, 
                "width":width, 
                "height": height,
                "file_name": f,
            }
        )

        pan_map = rgb2id(img)
        item = np.unique(pan_map)
        segments_info = []
        for i in item:
            if i > 1000:
                category_id = i // 1000
            else:
                category_id = i
            if i == 255:
                continue
            mask = pan_map == i
            segm = np.asfortranarray(mask*1, dtype=np.uint8)  
            segmentation = mask_util.encode(segm)
            bbox = mask_util.toBbox(segmentation)
            area = mask_util.area(segmentation)
            segments_info.append(
                {
                    "id": int(i),
                    "category_id": int(category_id),
                    "area": int(area),
                    "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    "iscrowd": 0,
                }
            )
        anns.append(
            {
                "image_id": img_id,
                "file_name": f,
                "segments_info": segments_info,
            }
        ) 
    annotations = {
        "images": images,
        "annotations": anns,
        "categories": categories,
    }
    json_file = os.path.join(root, json_name)
    save_json(annotations, json_file)
    print("done====" + json_file)

# ================================================
# annotation_json
# ================================================
inst_json_dir = ['vkitti_instances_clone.json',
                'vkitti_instances_test.json',]
for pan_path, json_name in zip(pan_dir, inst_json_dir):
    path = os.path.join(root, pan_path)
    filelist = os.listdir(path)
    filelist.sort()

    images = []
    anns = []
    for f in filelist:
        img_id = f[:-4]
        # print(img_id)
        img = np.array(Image.open(os.path.join(path, f)))
        width = img.shape[1]
        height = img.shape[0]
        images.append(
            {
                "id": img_id, 
                "width":width, 
                "height": height,
                "file_name": img_id + '.jpg',
            }
        )
        pan_map = rgb2id(img)
        item = np.unique(pan_map)
        segments_info = []
        for i in item:
            if i < 1000:
                continue
            mask = pan_map == i
            segm = np.asfortranarray(mask*1, dtype=np.uint8)  
            segmentation = mask_util.encode(segm)
            bbox = mask_util.toBbox(segmentation)
            area = mask_util.area(segmentation)
            segmentation['counts'] = segmentation['counts'].decode()
            segments_info.append(
                {
                "inst_id": int(i),
                "category_id": int(i // 1000),
                "segmentation": segmentation,
                "area": int(area),
                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                "iscrowd": 0,
                }
            )
        anns.append(
            {
                "image_id": img_id,
                "segments_info": segments_info,
            }
        ) 
    annotations = {
        "images": images,
        "annotations": anns,
        "categories": categories,
    }
    json_file = os.path.join(root, json_name)
    save_json(annotations, json_file)
    print("done====" + json_file)

# ================================================
# ALL_clone split into 5:1:1
# ================================================
scene = {
            "0001":"datasets/Virtual_KITTI2/Scene01/clone/frames/rgb/Camera_0",
            "0002":"datasets/Virtual_KITTI2/Scene02/clone/frames/rgb/Camera_0",
            "0006":"datasets/Virtual_KITTI2/Scene06/clone/frames/rgb/Camera_0",
            "0018":"datasets/Virtual_KITTI2/Scene18/clone/frames/rgb/Camera_0",
            "0020":"datasets/Virtual_KITTI2/Scene20/clone/frames/rgb/Camera_0"
        }
train_img_id = []
val_img_id = []
test_img_id = []
for k, sc in scene.items():
    filelist = os.listdir(sc)
    img_num = len(filelist)
    a = img_num // 7
    b = int(img_num - a - a)
    print(sc, a, b)
    for i in range(0, b):
        img_id = k + "_" + str(i).zfill(5)
        train_img_id.append(img_id)
        # print(img_id)
    for i in range(b, b+a):
        img_id = k + "_" + str(i).zfill(5)
        val_img_id.append(img_id)
        # print(img_id)
    for i in range(b+a, b+a+a):
        img_id = k + "_" + str(i).zfill(5)
        test_img_id.append(img_id)
        # print(img_id)

ann_dir = { 'panoptic_gt_id' : "datasets/Virtual_KITTI2/vkitti_panoptic_gt_clone.json",
            'inatance' : "datasets/Virtual_KITTI2/vkitti_instances_clone.json", }
train_json_file = { 'panoptic_gt_id' : "ALL_clone/vkitti_panoptic_gt_511_train.json",
                    'inatance' : "ALL_clone/vkitti_instances_511_train.json", }
val_json_file = {   'panoptic_gt_id' : "ALL_clone/vkitti_panoptic_gt_511_val.json",
                    'inatance' : "ALL_clone/vkitti_instances_511_val.json", }
test_json_file = {  'panoptic_gt_id' : "ALL_clone/vkitti_panoptic_gt_511_test.json",
                    'inatance' : "ALL_clone/vkitti_instances_511_test.json", }
for k, ann_file in ann_dir.items():
    with open(ann_file, "r") as f:
        jsonobj = json.load(f)

    images = jsonobj['images']
    annotations = jsonobj['annotations']
    categories = jsonobj['categories']

    images_dic = {}
    annotaitons_dic = {}
    for item in images:
        images_dic[item['id']] = item
    for item in annotations:
        annotaitons_dic[item['image_id']] = item

    train_images = []
    train_annotations = []
    val_images = []
    val_annotaions = []
    test_images = []
    test_annotations = []

    for img_id in train_img_id:
        train_images.append(images_dic[img_id])
        train_annotations.append(annotaitons_dic[img_id])
    for img_id in val_img_id:
        val_images.append(images_dic[img_id])
        val_annotaions.append(annotaitons_dic[img_id])
    for img_id in test_img_id:
        test_images.append(images_dic[img_id])
        test_annotations.append(annotaitons_dic[img_id])

    annotations = {
                "images": train_images,
                "annotations": train_annotations,
                "categories": categories,
              }
    json_file = os.path.join(root, train_json_file[k])
    save_json(annotations, json_file)
    print("done====" + json_file)

    annotations = {
                "images": val_images,
                "annotations": val_annotaions,
                "categories": categories,
              }
    json_file = os.path.join(root, val_json_file[k])
    save_json(annotations, json_file)
    print("done====" + json_file)

    annotations = {
                "images": test_images,
                "annotations": test_annotations,
                "categories": categories,
              }
    json_file = os.path.join(root, test_json_file[k])
    save_json(annotations, json_file)
    print("done====" + json_file)

    print(len(train_img_id),len(test_img_id))

# ================================================
# val datasets under per scene view
# ================================================
pan_gt_root = os.path.join(root, "ALL_clone/panoptic_gt_id")
target_dir = {
    "0001": "ALL_clone/pan_gt_511_val_s1",
    "0002": "ALL_clone/pan_gt_511_val_s2",
    "0006": "ALL_clone/pan_gt_511_val_s6",
    "0018": "ALL_clone/pan_gt_511_val_s18",
    "0020": "ALL_clone/pan_gt_511_val_s20",
}
for k, target_path in target_dir.items():
    target_path = os.path.join(root, target_path)
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
    for file_id in val_img_id:
        seg_id = file_id[:4]
        if seg_id == k:
            filename = file_id + '.png'
            dst = os.path.join(target_path, filename)
            src = os.path.join(pan_gt_root, filename)
            # print(filename, target_dir)
            shutil.copyfile(src, dst)

json_dir = {
    "0001": "ALL_clone/vkitti_511_val_s1.json",
    "0002": "ALL_clone/vkitti_511_val_s2.json",
    "0006": "ALL_clone/vkitti_511_val_s6.json",
    "0018": "ALL_clone/vkitti_511_val_s18.json",
    "0020": "ALL_clone/vkitti_511_val_s20.json",
}
ann_file = os.path.join(root, "ALL_clone/vkitti_panoptic_gt_511_val.json") 
with open(ann_file, "r") as f:
    jsonobj = json.load(f)

images = jsonobj['images']
annotations = jsonobj['annotations']
categories = jsonobj['categories']

val_images = {}
val_annotaions = {}
for k in json_dir.keys():
    val_images[k] = []
    val_annotaions[k] = []
for item in images:
    key = item['id'][:4]
    val_images[key].append(item)
for item in annotations:
    key = item['image_id'][:4]
    val_annotaions[key].append(item)

for k, json_file in json_dir.items():
    annotations = {
        "images": val_images[k],
        "annotations": val_annotaions[k],
        "categories": categories,
    }
    json_file = os.path.join(root, json_file)
    save_json(annotations, json_file)
    print("done====" + json_file)
    