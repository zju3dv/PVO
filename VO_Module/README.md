
## Requirements

To run the code you will need ...
* **Inference:** Running the demos will require a GPU with at least 11G of memory. 

* **Training:** Training requires a GPU with at least 24G of memory. We train on 4 x RTX-3090 GPUs.

## Getting Started
1. Creating a new anaconda environment using the provided .yaml file. Use `environment_novis.yaml` to if you do not want to use the visualization
```Bash
conda env create -f environment.yml
pip install evo --upgrade --no-binary evo
pip install gdown
```

2. Compile the extensions (takes about 10 minutes)
```Bash
python setup.py install
```

## Acknowledgements
We additionally use evaluation tools from [evo](https://github.com/MichaelGrupp/evo) and [tartanair_tools](https://github.com/castacks/tartanair_tools). Our code is based on the code provided by [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM).

<center><img src="misc/DROID.png" width="640" style="center"></center>

[DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras](https://arxiv.org/abs/2108.10869)  
Zachary Teed and Jia Deng

```
@article{teed2021droid,
  title={{DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras}},
  author={Teed, Zachary and Deng, Jia},
  journal={arXiv preprint arXiv:2108.10869},
  year={2021}
}
```
