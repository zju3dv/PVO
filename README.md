# PVO: Panoptic Visual Odometry
### [Project Page](https://zju3dv.github.io/pvo/) | [Paper](https://arxiv.org/abs/2207.01610)
<br/>

> PVO: Panoptic Visual Odometry  

> [Weicai Ye](https://ywcmaike.github.io/), [Xinyue Lan](https://github.com/siyisan), [Shuo Chen](https://github.com/Eric3778), [Yuhang Ming](https://github.com/YuhangMing), [Xinyuan Yu](https://github.com/RickyYXY), [Hujun Bao](http://www.cad.zju.edu.cn/home/bao/), [Zhaopeng Cui](https://zhpcui.github.io/), [Guofeng Zhang](http://www.cad.zju.edu.cn/home/gfzhang)

> Arxiv 2022

![demo_vid](assets/pvo_teaser.gif)

## test on vkitti 15-deg-left datasets.
0) prepare
follow [prepare.md](prepare.md)
```
conda activate droidenv
```

1) generate inital panoptic segmentation
```
sh tools/initial_segmentation.sh  
```

2) vps->vo，vo Module generate pose, flow and depth
```
sh tools/test_vo_scene.sh  
```

3) vo->vps, vps Module use flow and depth from vo Module and generate final video panoptic segmentation results and vpq.
```
sh tools/test_vps.sh  
```

## metrics on Virtual_KITTI2
|Scene|RMSE|vpq_all/vpq_thing/vpq_stuff|
|-----|----|---------------------------|
|Scene01|0.371|40.39/26.43/44.57|
|Scene02|0.058|68.84/88.83/62.18|
|Scene06|0.113|66.38/79.99/62.97|
|Scene18|0.951|68.35/83.86/63.92|
|Scene20|3.503|35.11/16.83/40.59|

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@article{Ye2022PVO,
    title={PVO: Panoptic Visual Odometry},
    author={Ye, Weicai and Lan, Xinyue and Chen, Shuo and Ming, Yuhang and Yu, Xinyuan and Bao, Hujun and Cui, Zhaopeng and Zhang, Guofeng},
    booktitle={arxiv}, 
    year={2022}
  }
```