
# checkpoint
-  Download the panFPN model from google drive: [panFPN_checkpoint](https://pan.baidu.com/s/1ncSi_EihY479SkCEFLHYzw?pwd=zsjq) and save to checkpoints/panFPN.pth
- Download the vo model from google drive:[vo_checkpoint](https://pan.baidu.com/s/10_tIdaDPf5DjgmU9O6iVYg?pwd=nei5) and save to checkpoints/vkitti2_dy_train_semiv4_080000.pth 

# droidenv env
1. Creating a new anaconda environment using the provided .yaml file. Use `VO_Module/environment_novis.yaml` to if you do not want to use the visualization
```Bash
conda env create -f VO_Module/environment.yml
pip install evo --upgrade --no-binary evo
pip install gdown
```
2. Compile the extensions (takes about 10 minutes)
```Bash
python setup.py install
```
3. video panoptic segmentation requirements. The Video panoptic segmentation module is based on [Detectron2](https://github.com/facebookresearch/detectron2), you can install Detectron2 following [the instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
```Bash
conda activate droidenv
conda install pytorch==1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia

python -m pip install -e VPS_Module
pip install git+https://github.com/cocodataset/panopticapi.git
```

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

