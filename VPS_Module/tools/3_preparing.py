import os
import json
from PIL import Image
import numpy as np
from panopticapi.utils import rgb2id, id2rgb, save_json, IdGenerator
import argparse
from default import VKITTI_CATEGORIES
import pycocotools.mask as mask_util
import shutil

max_ins = 10000

def generate_json(filelist, scene_id, path, target_dir, json_file):   
    images = []
    anns = []
    for f in filelist:
        cur_seq_id = f[:4]
        if cur_seq_id != scene_id:
            continue
        img_id = f[:-4]
        print(img_id)
        img = np.array(Image.open(os.path.join(path, f)))
        images.append(
            {
                "id": img_id, 
                "width":img.shape[1], 
                "height": img.shape[0],
                "file_name": f,
            }
        )
        pan_res = rgb2id(img)
        item = np.unique(pan_res)

        segments_info = []
        for i in item:
            if i == 0:
                continue
            mask = pan_res == i     
            segments_info.append(
                {
                    "id": int(i),
                    "category_id": int(i // max_ins),
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
        dst = os.path.join(target_dir, f)
        src = os.path.join(path, f)
        shutil.copyfile(src, dst)

    annotations = {
                "images": images,
                "annotations": anns,
                "categories": VKITTI_CATEGORIES,
              }
    save_json(annotations, json_file)
    print("done====" + json_file)

categories_dict = {el['trainId']: el for el in VKITTI_CATEGORIES}
id_generator = IdGenerator(categories_dict)
stuff_cat_map = {}
for i in range(1,12):
    new_id, color = id_generator.get_id_and_color(i)
    stuff_cat_map[new_id] = i

path = "shared_data/tmp/vo_fusion_vo_match" 
filelist = os.listdir(path)
filelist.sort()

json_root = "shared_data/json" 
if not os.path.isdir(json_root):
    os.makedirs(json_root)
target_root = "shared_data/final_vps_res"
dic = {
    '0001': 'vo_fusion_vo_track_s1.json',
    '0002': 'vo_fusion_vo_track_s2.json',
    '0006': 'vo_fusion_vo_track_s6.json',
    '0018': 'vo_fusion_vo_track_s18.json',
    '0020': 'vo_fusion_vo_track_s20.json',
}
sce_dic = {
    '0001': 's1',
    '0002': 's2',
    '0006': 's6',
    '0018': 's18',
    '0020': 's20',
}
for scene_id, json_name in dic.items():
    json_file = os.path.join(json_root, json_name)
    target_dir = os.path.join(target_root, sce_dic[scene_id])
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    generate_json(filelist, scene_id, path, target_dir, json_file)
