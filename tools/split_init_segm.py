import os
import shutil

print("copy segmentation into datasets")
target_root = "datasets/Virtual_KITTI2/"
dic = { "0001": "Scene01",
        "0002": "Scene02",
        "0006": "Scene06",
        "0018": "Scene18",
        "0020": "Scene20"}

clone_path = "shared_data/panoptic_segm_init_clone/inference/pan_seg"
filelist = os.listdir(clone_path)
filelist.sort()
for filename in filelist:
    seg_id = filename[:4]
    sce = dic[seg_id]
    target_dir = os.path.join(os.path.join(target_root, sce), "clone/panFPN_segm")
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    dst = os.path.join(target_dir, filename)
    src = os.path.join(clone_path, filename)
    shutil.copyfile(src, dst)

left_path = "shared_data/panoptic_segm_init_15-deg-left/inference/pan_seg"
filelist = os.listdir(left_path)
filelist.sort()
for filename in filelist:
    seg_id = filename[:4]
    sce = dic[seg_id]
    target_dir = os.path.join(os.path.join(target_root, sce), "15-deg-left/panFPN_segm")
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    dst = os.path.join(target_dir, filename)
    src = os.path.join(clone_path, filename)
    shutil.copyfile(src, dst)
