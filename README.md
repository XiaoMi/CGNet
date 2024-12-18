# CGNet
The official implementation of the ECCV 2024 paper: 
[Continuity Preserving Online CenterLine Graph Learning](https://arxiv.org/abs/2407.11337)

## Abstract
Lane topology, which is usually modeled by a centerline
graph, is essential for high-level autonomous driving. For a high-quality
graph, both topology connectivity and spatial continuity of centerline
segments are critical. However, most of existing approaches pay more
attention to connectivity while neglect the continuity. Such kind of cen-
terline graph usually cause problem to planning of autonomous driving.
To overcome this problem, we present an end-to-end network, CGNet,
with three key modules: 1) Junction Aware Query Enhancement module,
which provides positional prior to accurately predict junction points; 2)
Bézier Space Connection module, which enforces continuity constraints
on any two topologically connected segments in a Bézier space; 3) It-
erative Topology Refinement module, which is a graph-based network
with memory to iteratively refine the predicted topological connectivity.
CGNet achieves state-of-the-art performance on both nuScenes and Ar-
goverse2 datasets

## Motivation
<div align="center">
  <img src="https://github.com/XiaoMi/CGNet/blob/main/figs/teaser.png" width="70%">
</div>

Top: A toy example which illustrates the centerline graph and the impact of overlooking the continuity. Bottom: Comparison with MapTR and TopoNet. They predicts inaccurate position of junction points and wrong topology, all leading to the discontinuous path. Our CGNet obtain the continuous path.


## Qualitative results
<div align="center">
  <img src="https://github.com/XiaoMi/CGNet/blob/main/figs/result.png" width="70%">
</div>
Qualitative comparisons under different weather and lighting conditions on nuScenes. CGNet predicts more accurate position of junction points and correct topology, leading to a more continuous and smooth path.

## Usage

#### Download
Download the pretrained models using these link: [pretrained_models](https://drive.google.com/drive/folders/1iHs5kZ2IEl0BxrkBFqgMFaur2NUj8flE?usp=drive_link).


#### Installation
```
conda create -n cgnet-env python=3.8 -y
pip install -r requirement.txt

cd mmdetection3d
python setup.py develop

## Install GeometricKernelAttention. Please refer to [MapTR](https://github.com/hustvl/MapTR).
cd projects/mmdet3d_plugin/cgnet/modules/ops/geometric_kernel_attn
python setup.py build install
```

#### Prepare nuScenes data

```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes/ann --extra-tag nuscenes --version v1.0 --canbus ./data/nuscenes
```


#### Train, Test and Visualize
```
#train
python tools/train.py projects/configs/cgnet/cgnet_ep110.py

#test
python tools/test.py projects/configs/cgnet/cgnet_ep110.py ckpts/cgnet_ep110.pth --eval chamfer openlane topology

#vis
python tools/CGNet_visualize.py projects/configs/test/cgnet_ep110.py ckpts/cgnet_ep110.pth --show-dir ./show
```


## Citation
If you find this work useful for your research, please cite:
```
@inproceedings{han2025continuity,
  title={Continuity preserving online centerline graph learning},
  author={Han, Yunhui and Yu, Kun and Li, Zhiwei},
  booktitle={European Conference on Computer Vision},
  pages={342--359},
  year={2025},
  organization={Springer}
}
```

## Acknowledgements
We would like to thank [MapTR](https://github.com/hustvl/MapTR), [STSU](https://github.com/ybarancan/STSU), [LaneGNN](https://github.com/jzuern/lanegnn/tree/main), [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2), [TopoNet](https://github.com/OpenDriveLab/TopoNet), [VectorMapNet](https://github.com/Mrmoore98/VectorMapNet_code) for their great codes!
