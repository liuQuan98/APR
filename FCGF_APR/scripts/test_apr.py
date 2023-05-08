import open3d as o3d  # prevent loading error

import sys
import logging
import json
import argparse
import numpy as np
from easydict import EasyDict as edict

import torch
from model import load_model

from lib.complement_data_loader import make_data_loader

from util.pointcloud import make_open3d_point_cloud, make_open3d_feature
from lib.eval import find_nn_gpu
from lib.timer import AverageMeter, Timer

from chamferdist import ChamferDistance

import MinkowskiEngine as ME
import matplotlib.pyplot as plt  
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])


def chamfer_distance(array1, array2, device):
    cd_dist = ChamferDistance()
    n1 = len(array1)
    n2 = len(array2)
    array1 = torch.unsqueeze(array1, dim=0).to(device)
    array2 = torch.unsqueeze(array2, dim=0).to(device)
    forward_cd_dist = cd_dist(array1, array2)
    backward_cd_dist = cd_dist(array2, array1)
    return forward_cd_dist / n1 + backward_cd_dist / n2


def find_corr(xyz0, xyz1, F0, F1, subsample_size=-1):
    subsample = len(F0) > subsample_size
    if subsample_size > 0 and subsample:
        N0 = min(len(F0), subsample_size)
        N1 = min(len(F1), subsample_size)
        inds0 = np.random.choice(len(F0), N0, replace=False)
        inds1 = np.random.choice(len(F1), N1, replace=False)
        F0, F1 = F0[inds0], F1[inds1]

    # Compute the nn
    nn_inds = find_nn_gpu(F0, F1, nn_max_n=500)
    if subsample_size > 0 and subsample:
        return xyz0[inds0], xyz1[inds1[nn_inds]]
    else:
        return xyz0, xyz1[nn_inds]

def apply_transform(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    return pts @ R.t() + T

def evaluate_nn_dist(xyz0, xyz1, T_gth):
    xyz0 = apply_transform(xyz0, T_gth)
    dist = np.sqrt(((xyz0 - xyz1)**2).sum(1) + 1e-6)
    return dist.tolist()

def main(config):
  test_loader = make_data_loader(
      config, config.test_phase, 1, num_threads=config.test_num_thread, shuffle=False)

  num_feats = 1

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  Model = load_model(config.encoder_model)
  model = Model(
      num_feats,
      config.model_n_out,
      bn_momentum=config.bn_momentum,
      conv1_kernel_size=config.conv1_kernel_size,
      normalize_feature=config.normalize_feature)
  checkpoint = torch.load(config.save_dir + '/checkpoint.pth')
  model.load_state_dict(checkpoint['encoder_state_dict'])
  model = model.to(device)
  model.eval()

  success_meter, rte_meter, rre_meter = AverageMeter(), AverageMeter(), AverageMeter()
  data_timer, feat_timer, reg_timer = Timer(), Timer(), Timer()

  test_iter = test_loader.__iter__()
  N = len(test_iter)
  n_gpu_failures = 0

  # downsample_voxel_size = 2 * config.voxel_size
  dists_success = []
  dists_fail = []
  dists_nn = []
  list_rte = []
  list_rre = []
  trans_gt = []
  T_gt = []
  T_est = []

  rte_thresh = 2
  rre_thresh = 5

  print(f"rre thresh: {rre_thresh}; rte_thresh: {rte_thresh}")

  for i in range(len(test_iter)):
    data_timer.tic()
    try:
      data_dict = test_iter.next() 
    except ValueError:
      n_gpu_failures += 1
      logging.info(f"# Erroneous GPU Pair {n_gpu_failures}")
      continue
    data_timer.toc()
    xyz0, xyz1 = data_dict['pcd0'][0], data_dict['pcd1'][0]
    T_gth = data_dict['T_gt']
    T_gt.append(T_gth)
    dist_gth = np.sqrt(np.sum((T_gth[:3, 3].cpu().numpy())**2))
    trans_gt.append(dist_gth)
    xyz0np, xyz1np = xyz0.numpy(), xyz1.numpy()

    pcd0 = make_open3d_point_cloud(xyz0np)
    pcd1 = make_open3d_point_cloud(xyz1np)

    with torch.no_grad():
      feat_timer.tic()
      sinput0 = ME.SparseTensor(
          data_dict['sinput0_F'].to(device), coordinates=data_dict['sinput0_C'].to(device))
      enc0 = model(sinput0)
      F0 = enc0.F.detach()
      sinput1 = ME.SparseTensor(
          data_dict['sinput1_F'].to(device), coordinates=data_dict['sinput1_C'].to(device))
      enc1 = model(sinput1)
      F1 = enc1.F.detach()
      feat_timer.toc()

    xyz0_corr, xyz1_corr = find_corr(xyz0, xyz1, F0, F1, subsample_size=5000)
    dists_nn.append(evaluate_nn_dist(xyz0_corr, xyz1_corr, T_gth))

    feat0 = make_open3d_feature(F0, 32, F0.shape[0])
    feat1 = make_open3d_feature(F1, 32, F1.shape[0])

    reg_timer.tic()
    distance_threshold = config.voxel_size * 1.0
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd0, pcd1, feat0, feat1, False, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 10000))
    T_ransac = torch.from_numpy(ransac_result.transformation.astype(np.float32))
    reg_timer.toc()

    T_est.append(T_ransac)

    # Translation error
    rte = np.linalg.norm(T_ransac[:3, 3] - T_gth[:3, 3])
    rre = np.arccos((np.trace(T_ransac[:3, :3].t() @ T_gth[:3, :3]) - 1) / 2)

    # Check if the ransac was successful. successful if rte < 2m and rre < 5◦
    # http://openaccess.thecvf.com/content_ECCV_2018/papers/Zi_Jian_Yew_3DFeat-Net_Weakly_Supervised_ECCV_2018_paper.pdf

    # rte_thresh = 2
    # rre_thresh = 5

    if rte < rte_thresh:
      rte_meter.update(rte)

    if not np.isnan(rre) and rre < np.pi / 180 * rre_thresh:
      rre_meter.update(rre * 180 / np.pi)

    if rte < rte_thresh and not np.isnan(rre) and rre < np.pi / 180 * rre_thresh:
      success_meter.update(1)
      dists_success.append(dist_gth)
    else:
      success_meter.update(0)
      dists_fail.append(dist_gth)
      logging.info(f"Failed with RTE: {rte}, RRE: {rre * 180 / np.pi}")
    
    list_rte.append(rte)
    list_rre.append(rre)

    if i % 10 == 0:
      logging.info(
          f"{i} / {N}: Data time: {data_timer.avg}, Feat time: {feat_timer.avg}," +
          f" Reg time: {reg_timer.avg}, RTE: {rte_meter.avg}," +
          f" RRE: {rre_meter.avg}, Success: {success_meter.sum} / {success_meter.count}"
          + f" ({success_meter.avg * 100} %)")
      data_timer.reset()
      feat_timer.reset()
      reg_timer.reset()

  print(f"rre thresh: {rre_thresh}; rte_thresh: {rte_thresh}")

  # dists_nn = np.array(dists_nn) # dists_nn: N_pairs * N_num_point_one_frame
  # hit_ratios = []
  # for seed_tao_1 in range(1, 11):
  #   tao_1 = seed_tao_1 * 0.1
  #   hit_ratios.append(np.mean((dists_nn < tao_1).astype(float), axis=1).tolist())

  # fmrs = []
  # # hit_ratios: 10 * N_pairs
  # for seed_tao_2 in range(1, 11):
  #   tao_2 = seed_tao_2 * 0.01
  #   fmrs.append(np.mean((np.array(hit_ratios) > tao_2).astype(float), axis=1).tolist())
  
  # fmrs: 10*10, dim0: seed_tao_2; dim2: seed_tao_1
  # np.save('FMRs_apr.npy', fmrs)

  logging.info(
      f"RTE: {rte_meter.avg}, var: {rte_meter.var}," +
      f" RRE: {rre_meter.avg}, var: {rre_meter.var}, Success: {success_meter.sum} " +
      f"/ {success_meter.count} ({success_meter.avg * 100} %)")


def str2bool(v):
  return v.lower() in ('true', '1')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_dir', default=None, type=str)
  parser.add_argument('--test_phase', default='test', type=str)
  parser.add_argument('--LoKITTI', default=False, type=str2bool)
  parser.add_argument('--LoNUSCENES', default=False, type=str2bool)
  parser.add_argument('--test_num_thread', default=5, type=int)
  parser.add_argument('--pair_min_dist', default=None, type=int)
  parser.add_argument('--pair_max_dist', default=None, type=int)
  parser.add_argument('--downsample_single', default=1.0, type=float)
  parser.add_argument('--kitti_root', type=str, default="/data/kitti/")
  parser.add_argument('--dataset', type=str, default="PairComplementKittiDataset")
  args = parser.parse_args()

  config = json.load(open(args.save_dir + '/config.json', 'r'))
  config = edict(config)
  config.save_dir = args.save_dir
  config.test_phase = 'test'
  config.kitti_root = args.kitti_root
  config.kitti_odometry_root = args.kitti_root + '/dataset'
  config.test_num_thread = args.test_num_thread
  config.LoKITTI = args.LoKITTI
  config.LoNUSCENES = args.LoNUSCENES
  config.debug_use_old_complement = True
  config.debug_need_complement = False
  config.phase = 'test'
  config.dataset = args.dataset
  if config.dataset == "PairComplementNuscenesDataset":
    config.use_old_pose = True
  else:
    config.use_old_pose = False

  if args.pair_min_dist is not None and args.pair_max_dist is not None:
    config.pair_min_dist = args.pair_min_dist
    config.pair_max_dist = args.pair_max_dist
  config.downsample_single = args.downsample_single

  main(config)
