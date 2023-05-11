# APR: Online Distant Point Cloud Registration Through Aggregated Point Cloud Reconstruction, Implemented with Fully Convolutional Geometric Features ([FCGF](https://github.com/chrischoy/FCGF))

For many driving safety applications, it is of great importance to accurately register LiDAR point clouds generated on distant moving vehicles. However, such point clouds have extremely different point density and sensor perspective on the same object, making registration on such point clouds very hard. In this paper, we propose a novel feature extraction framework, called APR, for online distant point cloud registration. Specifically, APR leverages an autoencoder design, where the autoencoder reconstructs a denser aggregated point cloud with several frames instead of the original single input point cloud. Our design forces the encoder to extract features with rich local geometry information based on one single input point cloud. Such features are then used for online distant point cloud registration. We conduct extensive experiments against state-of-the-art (SOTA) feature extractors on KITTI and nuScenes datasets. Results show that APR outperforms all other extractors by a large margin, increasing average registration recall of SOTA extractors by 7.1% on LoKITTI and 4.6% on LoNuScenes.

This repository is an implementation of APR using FCGF as the feature extractor.

## Requirements

- Ubuntu 14.04 or higher
- CUDA 11.1 or higher
- Python v3.7 or higher
- Pytorch v1.6 or higher
- [MinkowskiEngine](https://github.com/stanfordvl/MinkowskiEngine) v0.5 or higher

## Training & Testing

First, use FCGF_APR as the working directory:

```
cd ./FCGF_APR```
conda activate apr
```

If you haven't downloaded any of the datasets (KITTI and nuScenes) according to our specification, please refer to the README.md in the [parent directory](../README.md).

### Setting the distance between two LiDARs (registration difficulty)

As the major focus of this paper, we divide the registration datasets into different slices according to the distance $d$ between two LiDARs. Greater $d$ leads to a smaller overlap and more divergent point density, resulting in a higher registration difficulty. We denote range of $d$ with the parameter `--pair_min_dist` and `--pair_max_dist`, which can be found in `./scripts/train_{$method}_{$dataset}.sh`. For example, setting

```
--pair_min_dist 5 \
--pair_max_dist 20 \
```

will set $d\in [5m,20m]$. In other words, for every pair of point clouds, the ground-truth euclidean distance betwen two corresponding LiDAR positions (i.e., the origins of the two specified point clouds) obeys a uniform distribution between 5m and 20m.

### Training suggestions

For cases where you want `--pair_max_dist` to be larger than 20, we recommend following the two-stage training paradigm as pointed out in Section 5 of our paper:

1. Pretrain a model with the following distance parameters: `--pair_min_dist 5  --pair_max_dist 20`. Record the pretrained model path that is printed at the beginning of the training. It shoud be some path like this: `./outputs/Experiments/PairComplementKittiDataset-v0.3/GenerativePairTrainer//SGD-lr1e-1-e200-b4i1-modelnout32/YYYY-MM-DD_HH-MM-SS`
2. Finetune a new model on `--pair_min_dist 5  --pair_max_dist {$YOUR_SPECIFIC_DISTANCE}`, and paste the pretrained model path to  `--resume "{$PRETRAINED_PATH}/chechpoint.pth"` and `--resume_dir "{$PRETRAINED_PATH}"`. Do not forget to set `--finetune_restart true`.

Emperically, the pretraining strategy helps a lot in model convergence especially when the distance is large; Otherwise the model just diverges.

### Launch the training

Notes:

1. Remember to set `--use_old_pose` to true when using the nuScenes dataset.
2. The symmetric APR setup can be enabled by setting `--symmetric` to True.

To train FCGF-APR on either dataset, run either of the following command inside conda environment `apr`:

```
./scripts/train_apr_kitti.sh
./scripts/train_apr_nuscenes.sh
```

The baseline method FCGF can be trained similarly with our dataset:

```
./scripts/train_fcgf_kitti.sh
./scripts/train_fcgf_nuscenes.sh
```

### Testing

To test FCGF-APR on either dataset, set  `OUT_DIR` to the specific model path before running the corresponding script inside conda environment `apr`:

```
./scripts/test_apr_kitti.sh
./scripts/test_apr_nuscenes.sh
```

The baseline method FCGF can be tested similarly:

```
./scripts/test_fcgf_kitti.sh
./scripts/test_fcgf_nuscenes.sh
```

## Pre-trained models

We provide our [model](https://drive.google.com/file/d/1mLqiahQMgYMRyB4XKhp-HJdy5yavL2fj/view?usp=sharing) pretrained on FCGF+APR, with different point cloud distance. Extract the file into the 'outputs' directory and constitute './outputs/some_model' into the 'OUT_DIR' section in the test scripts to reproduce the results showed in our paper.
