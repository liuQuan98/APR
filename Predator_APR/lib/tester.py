from datasets.nuscenes import NUSCENESDataset
from lib.trainer import Trainer
import os, torch
from tqdm import tqdm
import numpy as np
from lib.benchmark_utils import ransac_pose_estimation, random_sample, get_angle_deviation, to_o3d_pcd, to_array
from lib.timer import Timer
import open3d as o3d

# Modelnet part
from common.math_torch import se3
from common.math.so3 import dcm2euler
from common.misc import prepare_logger
from collections import defaultdict
import coloredlogs


class KITTITester(Trainer):
    """
    KITTI tester
    """
    def __init__(self,args):
        Trainer.__init__(self,args)
        if "rot_threshold" in [k for (k, v) in args.items()] and "trans_threshold" in [k for (k, v) in args.items()]:
            self.rot_threshold = args.rot_threshold
            self.trans_threshold = args.trans_threshold
        else:
            print('No rot & trans upper bound designated. Using default (5 degrees and 2 meters).')
            self.rot_threshold = 5
            self.trans_threshold = 2
        print(f"rot_threshold: {self.rot_threshold}, trans_threshold: {self.trans_threshold}")
    
    def test(self):
        print('Start to evaluate on test datasets...')
        tsfm_est = []
        num_iter = int(len(self.loader['test'].dataset) // self.loader['test'].batch_size)
        c_loader_iter = self.loader['test'].__iter__()
        
        self.model.eval()
        rot_gt, trans_gt =[],[]
        with torch.no_grad():
            for i in tqdm(range(num_iter)): # loop through this epoch
                prepare_timer, feat_timer = Timer(), Timer()
                prepare_timer.tic()
                inputs = c_loader_iter.next()
                prepare_timer.toc()
                ###############################################
                # forward pass
                for k, v in inputs.items():  
                    if type(v) == list:
                        inputs[k] = [item.to(self.device) for item in v]
                    else:
                        inputs[k] = v.to(self.device)

                feat_timer.tic()
                try:
                    feats, scores_overlap, scores_saliency = self.model(inputs)  #[N1, C1], [N2, C2]
                except:
                    feat_timer.toc()
                    continue
                feat_timer.toc()
                scores_overlap = scores_overlap.detach().cpu()
                scores_saliency = scores_saliency.detach().cpu()

                len_src = inputs['stack_lengths'][0][0]
                c_rot, c_trans = inputs['rot'], inputs['trans']
                rot_gt.append(c_rot.cpu().numpy())
                trans_gt.append(c_trans.cpu().numpy())
                src_feats, tgt_feats = feats[:len_src], feats[len_src:]
                src_pcd , tgt_pcd = inputs['src_pcd_raw'], inputs['tgt_pcd_raw']
                src_overlap, tgt_overlap = scores_overlap[:len_src], scores_overlap[len_src:]
                src_saliency, tgt_saliency = scores_saliency[:len_src], scores_saliency[len_src:]

                n_points = 5000
                ########################################
                # run random sampling or probabilistic sampling
                # src_pcd, src_feats = random_sample(src_pcd, src_feats, n_points)
                # tgt_pcd, tgt_feats = random_sample(tgt_pcd, tgt_feats, n_points)

                src_scores = src_overlap * src_saliency
                tgt_scores = tgt_overlap * tgt_saliency

                if(src_pcd.size(0) > n_points):
                    idx = np.arange(src_pcd.size(0))
                    probs = (src_scores / src_scores.sum()).numpy().flatten()
                    idx_src = np.random.choice(idx, size= n_points, replace=False, p=probs)
                    src_pcd, src_feats = src_pcd[idx_src], src_feats[idx_src]
                if(tgt_pcd.size(0) > n_points):
                    idx = np.arange(tgt_pcd.size(0))
                    probs = (tgt_scores / tgt_scores.sum()).numpy().flatten()
                    idx_tgt = np.random.choice(idx, size= n_points, replace=False, p=probs)
                    tgt_pcd, tgt_feats = tgt_pcd[idx_tgt], tgt_feats[idx_tgt]

                ########################################
                # run ransac 
                distance_threshold = 0.3
                ts_est = ransac_pose_estimation(src_pcd, tgt_pcd, src_feats, tgt_feats, mutual=False, distance_threshold=distance_threshold, ransac_n = 4)
                tsfm_est.append(ts_est)

                # message = f"Inference time: prepare: {prepare_timer.avg}, reg: {feat_timer.avg}"
                # print(message)
                # self.logger.write(message + '\n')
        
        tsfm_est = np.array(tsfm_est)
        rot_est = tsfm_est[:,:3,:3]
        trans_est = tsfm_est[:,:3,3]
        rot_gt = np.array(rot_gt)
        trans_gt = np.array(trans_gt)[:,:,0]

        np.savez(f'{self.snapshot_dir}/results',rot_est=rot_est, rot_gt=rot_gt, trans_est = trans_est, trans_gt = trans_gt)

        r_deviation = get_angle_deviation(rot_est, rot_gt)
        translation_errors = np.linalg.norm(trans_est-trans_gt,axis=-1)

        flag_1=r_deviation<self.rot_threshold
        flag_2=translation_errors<self.trans_threshold
        correct=(flag_1 & flag_2).sum()
        precision=correct/rot_gt.shape[0]

        message=f'\n Registration recall: {precision:.3f}\n'

        # used for testing
        success_inds = flag_1 & flag_2
        dists = np.linalg.norm(trans_gt,axis=-1)
        np.save(f"{self.snapshot_dir}/success_dists.npy", dists[np.where(success_inds > 0)])
        np.save(f"{self.snapshot_dir}/fail_dists.npy", dists[np.where(success_inds < 1)])

        r_deviation = r_deviation[flag_1]
        translation_errors = translation_errors[flag_2]

        errors=dict()
        errors['rot_mean']=round(np.mean(r_deviation),3)
        errors['rot_median']=round(np.median(r_deviation),3)
        errors['trans_rmse'] = round(np.mean(translation_errors),3)
        errors['trans_rmedse']=round(np.median(translation_errors),3)
        errors['rot_std'] = round(np.std(r_deviation),3)
        errors['trans_std']= round(np.std(translation_errors),3)

        message+=str(errors)
        print(message)
        self.logger.write(message+'\n')


class NUSCENESTester(Trainer):
    """
    KITTI tester
    """
    def __init__(self,args):
        Trainer.__init__(self,args)
        if "rot_threshold" in [k for (k, v) in args.items()] and "trans_threshold" in [k for (k, v) in args.items()]:
            self.rot_threshold = args.rot_threshold
            self.trans_threshold = args.trans_threshold
        else:
            print('No rot & trans upper bound designated. Using default (5 degrees and 2 meters).')
            self.rot_threshold = 5
            self.trans_threshold = 2
    
    def test(self):
        print('Start to evaluate on test datasets...')
        tsfm_est = []
        num_iter = int(len(self.loader['test'].dataset) // self.loader['test'].batch_size)
        c_loader_iter = self.loader['test'].__iter__()
        
        self.model.eval()
        rot_gt, trans_gt =[],[]

        with torch.no_grad():
            for i in tqdm(range(num_iter)): # loop through this epoch
                prepare_timer, feat_timer = Timer(), Timer()
                prepare_timer.tic()
                inputs = c_loader_iter.next()
                prepare_timer.toc()
                ###############################################
                # forward pass
                for k, v in inputs.items():  
                    if type(v) == list:
                        inputs[k] = [item.to(self.device) for item in v]
                    else:
                        inputs[k] = v.to(self.device)

                feat_timer.tic()
                feats, scores_overlap, scores_saliency = self.model(inputs)  #[N1, C1], [N2, C2]
                feat_timer.toc()
                scores_overlap = scores_overlap.detach().cpu()
                scores_saliency = scores_saliency.detach().cpu()

                len_src = inputs['stack_lengths'][0][0]
                c_rot, c_trans = inputs['rot'], inputs['trans']
                rot_gt.append(c_rot.cpu().numpy())
                trans_gt.append(c_trans.cpu().numpy())
                src_feats, tgt_feats = feats[:len_src], feats[len_src:]
                src_pcd , tgt_pcd = inputs['src_pcd_raw'], inputs['tgt_pcd_raw']
                src_overlap, tgt_overlap = scores_overlap[:len_src], scores_overlap[len_src:]
                src_saliency, tgt_saliency = scores_saliency[:len_src], scores_saliency[len_src:]

                n_points = 5000
                ########################################
                # run random sampling or probabilistic sampling
                # src_pcd, src_feats = random_sample(src_pcd, src_feats, n_points)
                # tgt_pcd, tgt_feats = random_sample(tgt_pcd, tgt_feats, n_points)

                src_scores = src_overlap * src_saliency
                tgt_scores = tgt_overlap * tgt_saliency

                if(src_pcd.size(0) > n_points):
                    idx = np.arange(src_pcd.size(0))
                    probs = (src_scores / src_scores.sum()).numpy().flatten()
                    idx_src = np.random.choice(idx, size= n_points, replace=False, p=probs)
                    src_pcd, src_feats = src_pcd[idx_src], src_feats[idx_src]
                else:
                    idx_src = np.arange(src_pcd.size(0))
                if(tgt_pcd.size(0) > n_points):
                    idx = np.arange(tgt_pcd.size(0))
                    probs = (tgt_scores / tgt_scores.sum()).numpy().flatten()
                    idx_tgt = np.random.choice(idx, size= n_points, replace=False, p=probs)
                    tgt_pcd, tgt_feats = tgt_pcd[idx_tgt], tgt_feats[idx_tgt]
                else:
                    idx_tgt = np.arange(tgt_pcd.size(0))

                ########################################
                # run ransac 
                distance_threshold = 0.3
                ts_est = ransac_pose_estimation(src_pcd, tgt_pcd, src_feats, tgt_feats, mutual=False, distance_threshold=distance_threshold, ransac_n = 4)
                # ts_est, src_id, tgt_id = ransac_pose_estimation(src_pcd, tgt_pcd, src_feats, tgt_feats, mutual=True, distance_threshold=distance_threshold, ransac_n = 4)
                tsfm_est.append(ts_est)

                # message = f"Inference time: prepare: {prepare_timer.avg}, reg: {feat_timer.avg}"
                # print(message)
                # self.logger.write(message + '\n')
        
        tsfm_est = np.array(tsfm_est)
        rot_est = tsfm_est[:,:3,:3]
        trans_est = tsfm_est[:,:3,3]
        rot_gt = np.array(rot_gt)
        trans_gt = np.array(trans_gt)[:,:,0]

        np.savez(f'{self.snapshot_dir}/results',rot_est=rot_est, rot_gt=rot_gt, trans_est = trans_est, trans_gt = trans_gt)

        r_deviation = get_angle_deviation(rot_est, rot_gt)
        translation_errors = np.linalg.norm(trans_est-trans_gt,axis=-1)

        flag_1=r_deviation<self.rot_threshold
        flag_2=translation_errors<self.trans_threshold
        correct=(flag_1 & flag_2).sum()
        precision=correct/rot_gt.shape[0]

        message=f'\n Registration recall: {precision:.3f}\n'

        # used for testing
        success_inds = flag_1 & flag_2
        dists = np.linalg.norm(trans_gt,axis=-1)
        np.save(f"{self.snapshot_dir}/success_dists.npy", dists[np.where(success_inds > 0)])
        np.save(f"{self.snapshot_dir}/fail_dists.npy", dists[np.where(success_inds < 1)])

        r_deviation = r_deviation[flag_1]
        translation_errors = translation_errors[flag_2]

        errors=dict()
        errors['rot_mean']=round(np.mean(r_deviation),3)
        errors['rot_median']=round(np.median(r_deviation),3)
        errors['trans_rmse'] = round(np.mean(translation_errors),3)
        errors['trans_rmedse']=round(np.median(translation_errors),3)
        errors['rot_std'] = round(np.std(r_deviation),3)
        errors['trans_std']= round(np.std(translation_errors),3)

        message+=str(errors)
        print(message)
        self.logger.write(message+'\n')

        

def get_trainer(config):
    if(config.dataset == 'kitti'):
        return KITTITester(config)
    elif(config.dataset == 'nuscenes'):
        return NUSCENESTester(config)
    else:
        raise NotImplementedError
