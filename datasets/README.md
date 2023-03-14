# vkitti 15-deg-left dataset
- download [Virtual_KITTI2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) into datasets/Virtual_KITTI2
## Expected dataset structure for Virtual_KITTI2
```
Virtual_KITTI2/
  Scene01/
    15-deg-left/
    15-deg-right/
    ...
    clone/
  Scene02/
  Scene06/
  Scene18/
  Scene20/
```
- generate annotation for training and evaluating
```Bash
conda activate droidenv
sh tools/datasets/generate_vkitti_datasets.sh
python tools/datasets/generate_dynamic_masks.py
```
