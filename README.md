# PVO: Panoptic Visual Odometry
### [Project Page](https://zju3dv.github.io/pvo/) | [Paper](https://arxiv.org/abs/2207.01610)
<br/>

> PVO: Panoptic Visual Odometry  

> [[Weicai Ye](https://ywcmaike.github.io/), [Xinyue Lan](https://github.com/siyisan)]<sup>Co-Authors</sup>, [Shuo Chen](https://github.com/Eric3778), [Yuhang Ming](https://github.com/YuhangMing), [Xinyuan Yu](https://github.com/RickyYXY), [Hujun Bao](http://www.cad.zju.edu.cn/home/bao/), [Zhaopeng Cui](https://zhpcui.github.io/), [Guofeng Zhang](http://www.cad.zju.edu.cn/home/gfzhang)

> CVPR 2023

![demo_vid](assets/pvo_teaser.gif)

## Test on vkitti 15-deg-left datasets.
0) prepare.
follow [prepare.md](prepare.md)
```
conda activate droidenv
```

1) generate inital panoptic segmentation.
```
sh tools/initial_segmentation.sh  
```

2) vps->vo，vo Module generate pose, flow and depth.
```
sh tools/test_vo_scene.sh  
```

3) vo->vps, vps Module use flow and depth from vo Module and generate final video panoptic segmentation results and vpq.
```
sh tools/test_vps.sh  
```

## Metrics on Virtual_KITTI2
|Scene|RMSE|vpq_all/vpq_thing/vpq_stuff|
|-----|----|---------------------------|
|Scene01|0.371|40.39/26.43/44.57|
|Scene02|0.058|68.84/88.83/62.18|
|Scene06|0.113|66.38/79.99/62.97|
|Scene18|0.951|68.35/83.86/63.92|
|Scene20|3.503|35.11/16.83/40.59|

You can get the results in the paper by iterating multiple times.

## Train on vkitti 15-deg-left datasets.
1)  To train VPS_Module, you can refer to [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html) for more training details.
Here for example, you can train  vkitti 15-deg-left on 4 GPUs, and training results are saved on `output/vps_training/`. You can modify the config-file according to the hardware conditions.
```
python3 -W ignore VPS_Module/tools/train_net.py \
--config-file VPS_Module/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x_vkitti_511.yaml --num-gpu 4 \
MODEL.WEIGHTS checkpoints/panFPN.pth \
OUTPUT_DIR output/vps_training/
```

2) To train VO_Module, you can refer to [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) for more training details.
Here for example, you can train vkitti on 4 GPUs.
```
python VO_Module/train.py --gpus=4 --lr=0.00025
```

## Visualization
You can refer to [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) for visualization.
All demos can be run on a GPU with 11G of memory. While running, press the "s" key to increase the filtering threshold (= more points) and "a" to decrease the filtering threshold (= fewer points).
```
python VO_Module/evaluation_scripts/test_vo.py --datapath=datasets/Virtual_KITTI2/Scene01 --segm_filter True 
```

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex

@inproceedings{Ye2023PVO,
  title={{PVO: Panoptic visual odometry}},
  author={Ye, Weicai and Lan, Xinyue and Chen, Shuo and Ming, Yuhang and Yu, Xingyuan and Bao, Hujun and Cui, Zhaopeng and Zhang, Guofeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9579--9589},
  year={2023}
}

```

## Acknowledgement

Some code snippets are borrowed from [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) and [Detectron2](https://github.com/facebookresearch/detectron2). Thanks for these great projects.
