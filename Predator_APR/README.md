# APR: Online Distant Point Cloud Registration Through Aggregated Point Cloud Reconstruction, Implemented with PREDATOR: Registration of 3D Point Clouds with Low Overlap ([Predator](https://github.com/overlappredator/OverlapPredator))

For many driving safety applications, it is of great importance to accurately register LiDAR point clouds generated on distant moving vehicles. However, such point clouds have extremely different point density and sensor perspective on the same object, making registration on such point clouds very hard. In this paper, we propose a novel feature extraction framework, called APR, for online distant point cloud registration. Specifically, APR leverages an autoencoder design, where the autoencoder reconstructs a denser aggregated point cloud with several frames instead of the original single input point cloud. Our design forces the encoder to extract features with rich local geometry information based on one single input point cloud. Such features are then used for online distant point cloud registration. We conduct extensive experiments against state-of-the-art (SOTA) feature extractors on KITTI and nuScenes datasets. Results show that APR outperforms all other extractors by a large margin, increasing average registration recall of SOTA extractors by 7.1% on LoKITTI and 4.6% on LoNuScenes.

This repository is an implementation of APR using Predator as the feature extractor.

## Requirements

- Ubuntu 14.04 or higher
- CUDA 11.1 or higher
- Python v3.7 or higher
- Pytorch v1.6 or higher
- [MinkowskiEngine](https://github.com/stanfordvl/MinkowskiEngine) v0.5 or higher

## Training & Testing

First, use Predator_APR as the working directory:

```
cd ./Predator_APR
conda activate apr
```

If you haven't downloaded any of the datasets (KITTI and nuScenes) according to our specification, please refer to the README.md in the [parent directory](../README.md).

### Setting the distance between two LiDARs (registration difficulty)

As the major focus of this paper, we divide the registration datasets into different slices according to the registration difficulty, i.e., the distance $d$ between two LiDARs. Greater $d$ leads to a smaller overlap and more divergent point density, resulting in a higher registration difficulty. We denote range of $d$ with the parameter `pair_min_dist` and `pair_max_dist`, which can be found in `./configs/{$phase}/{$dataset}.yaml`. For example, setting

```
pair_min_dist: 5
pair_max_dist: 20
```

will set $d\in [5m,20m]$. In other words, for every pair of point clouds, the ground-truth euclidean distance betwen two corresponding LiDAR positions (i.e., the origins of the two specified point clouds) obeys a uniform distribution between 5m and 20m.

### Training suggestions

For cases where you want `pair_max_dist` to be larger than 20, we recommend following the two-stage training paradigm as pointed out in Section 5 of our paper:

1. Pretrain a model with the following distance parameters: `pair_min_dist: 5` and `pair_max_dist: 20`. Find out the converged model checkpoint file path according to your training config. It shoud be some path like this: `./snapshot/{$exp_dir}/checkpoints/model_best_recall.pth`
2. Finetune a new model on `pair_min_dist 5` and `pair_max_dist {$YOUR_SPECIFIC_DISTANCE}`, while setting the pretrained checkpoint file path in `pretrain: "{$PRETRAINED_CKPT_PATH}"` in the config file `./configs/{$phase}/{$dataset}.yaml`. Do not forget to set `pretrain_restart: True`.

Emperically, the pretraining strategy helps a lot in model convergence especially when the distance is large; Otherwise the model just diverges.

### Launch the training

Specify the GPU usage in `main.py`, then train Predator-APR with either of the following command inside conda environment `apr`:

```
python main.py ./configs/train/kitti.yaml
python main.py ./configs/train/nuscenes.yaml
```

Note: The symmetric APR setup is not supported with Predator backbone due to GPU memory issue.

### Testing

To test a specific model, please constitute the model directory into the 'pretrain' section in the `./config/test/{$dataset}.yaml`, then run either of the following:

```
python main.py configs/test/kitti.yaml
python main.py configs/test/nuscenes.yaml
```

## Pre-trained models

We provide our [model](https://drive.google.com/file/d/1mLqiahQMgYMRyB4XKhp-HJdy5yavL2fj/view?usp=sharing) trained on Predator+APR with different point cloud distance. To reproduce the results using our model, please extract the model checkpoints to the 'snapshot' directory.
