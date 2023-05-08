# -*- coding: future_fstrings -*-
#
# Written by Chris Choy <chrischoy@ai.stanford.edu>
# Distributed under MIT License
import os
import os.path as osp
import gc
import logging
import numpy as np
import json

import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from model import load_model
import util.transform_estimation as te
from lib.metrics import pdist, corr_dist
from lib.timer import Timer, AverageMeter
from lib.eval import find_nn_gpu

from util.file import ensure_dir
from util.misc import _hash

import MinkowskiEngine as ME

# this chamfer distance lib is from https://github.com/krrish94/chamferdist
from chamferdist import ChamferDistance


class TwoStageTrainer:

    def __init__(
        self,
        config,
        data_loader,
        val_data_loader=None,
    ):
        num_feats = 1  # occupancy only for 3D Match dataset. For ScanNet, use RGB 3 channels.

        # Model initialization
        EncoderModel = load_model(config.encoder_model)
        encoder_model = EncoderModel(
            num_feats,
            config.model_n_out,
            bn_momentum=config.bn_momentum,
            normalize_feature=config.normalize_feature,
            conv1_kernel_size=config.conv1_kernel_size,
            D=3)
        
        if config.symmetric:
            GeneratorModel = load_model(config.generator_model)
            generator_model = GeneratorModel(
                config.model_n_out,
                config.point_generation_ratio * 3,
                bn_momentum=config.bn_momentum,
                normalize_feature=config.normalize_feature,
                conv1_kernel_size=config.conv1_kernel_size,
                D=3)
        else:
            GeneratorModel = load_model(config.generator_model)
            generator_model = GeneratorModel(
                in_channel=config.model_n_out,
                out_points=config.point_generation_ratio,
                bn_momentum=config.bn_momentum)

        if config.weights:
            checkpoint = torch.load(config.weights)
            encoder_model.load_state_dict(checkpoint['encoder_state_dict'])
            generator_model.load_state_dict(checkpoint['generator_state_dict'])

        logging.info(encoder_model)
        logging.info(generator_model)

        self.config = config
        self.encoder_model = encoder_model
        self.generator_model = generator_model
        self.max_epoch = config.max_epoch
        self.save_freq = config.save_freq_epoch
        self.val_max_iter = config.val_max_iter
        self.val_epoch_freq = config.val_epoch_freq
        self.voxel_size = config.voxel_size

        self.best_val_metric = config.best_val_metric
        self.best_val_epoch = -np.inf
        self.best_val = -np.inf

        if config.use_gpu and not torch.cuda.is_available():
            logging.warning('Warning: There\'s no CUDA support on this machine, '
                            'training is performed on CPU.')
            raise ValueError('GPU not available, but cuda flag set')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.optimizer = getattr(optim, config.optimizer)(
            params = [
                {'params': encoder_model.parameters()},
                {'params': generator_model.parameters()}
            ],
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay)

        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.exp_gamma)

        self.start_epoch = 1
        self.checkpoint_dir = config.out_dir

        ensure_dir(self.checkpoint_dir)
        json.dump(
            config,
            open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
            indent=4,
            sort_keys=False)

        self.iter_size = config.iter_size
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.val_data_loader = val_data_loader

        self.test_valid = True if self.val_data_loader is not None else False
        self.log_step = int(np.sqrt(self.config.batch_size))
        self.encoder_model = self.encoder_model.to(self.device)
        self.generator_model = self.generator_model.to(self.device)
        self.writer = SummaryWriter(logdir=config.out_dir)

        if config.resume is not None:
            if osp.isfile(config.resume):
                logging.info("=> loading checkpoint '{}'".format(config.resume))
                state = torch.load(config.resume)
                if not config.finetune_restart:
                    self.start_epoch = state['epoch']
                    self.scheduler.load_state_dict(state['scheduler'])
                    self.optimizer.load_state_dict(state['optimizer'])
                    if 'best_val' in state.keys():
                        self.best_val = state['best_val']
                        self.best_val_epoch = state['best_val_epoch']
                        self.best_val_metric = state['best_val_metric']
                else:
                    logging.info("=> Finetuning, will only load model weights.")
                encoder_model.load_state_dict(state['encoder_state_dict'])
                generator_model.load_state_dict(state['generator_state_dict'])


            else:
                raise ValueError(f"=> no checkpoint found at '{config.resume}'")

    def train(self):
        """
        Full training logic
        """
        # Baseline random feature performance
        if self.test_valid:
            pass
            # with torch.no_grad():
            #     val_dict = self._valid_epoch()
                
            # for k, v in val_dict.items():
            #     self.writer.add_scalar(f'val/{k}', v, 0)

        for epoch in range(self.start_epoch, self.max_epoch + 1):
            lr = self.scheduler.get_lr()
            logging.info(f" Epoch: {epoch}, LR: {lr}")
            self._train_epoch(epoch)
            self._save_checkpoint(epoch)
            self.scheduler.step()

            if self.test_valid and epoch % self.val_epoch_freq == 0:
                with torch.no_grad():
                    val_dict = self._valid_epoch()

                for k, v in val_dict.items():
                    self.writer.add_scalar(f'val/{k}', v, epoch)
                if (self.best_val < val_dict[self.best_val_metric]) or \
                   (self.best_val_metric=='loss' and self.best_val > val_dict[self.best_val_metric]):
                    logging.info(
                        f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
                    )
                    self.best_val = val_dict[self.best_val_metric]
                    self.best_val_epoch = epoch
                    self._save_checkpoint(epoch, 'best_val_checkpoint')
                else:
                    logging.info(
                        f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
                    )
                    
    def chamfer_distance(self, array1, array2):
        cd_dist = ChamferDistance()
        n1 = len(array1)
        n2 = len(array2)
        array1 = torch.unsqueeze(array1, dim=0).to(self.device)
        array2 = torch.unsqueeze(array2, dim=0).to(self.device)
        forward_cd_dist = cd_dist(array1, array2)
        backward_cd_dist = cd_dist(array2, array1)
        return forward_cd_dist / n1 + backward_cd_dist / n2

    # chamfer distance specifically designed for idea 6(d)-ii
    def chamfer_distance_splitCurrentFrame(self, curr_F, curr_T, nghb, weigh_F, weigh_T):
        cd_dist = ChamferDistance()
        n1_F = len(curr_F)
        n1_T = len(curr_T)
        n2 = len(nghb)
        array1_F = torch.unsqueeze(curr_F, dim=0).to(self.device)
        array1_T = torch.unsqueeze(curr_T, dim=0).to(self.device)
        array1 = torch.unsqueeze(torch.cat([curr_F, curr_T]), dim=0).to(self.device)
        array2 = torch.unsqueeze(nghb, dim=0).to(self.device)
        forward_cd_dist_F = cd_dist(array1_F, array2)
        forward_cd_dist_T = cd_dist(array1_T, array2)
        backward_cd_dist = cd_dist(array2, array1)
        return (forward_cd_dist_F * weigh_F + forward_cd_dist_T * weigh_T) / (n1_F * weigh_F + n1_T * weigh_T) + \
               (backward_cd_dist / n2)
    
    def find_corr(self, xyz0, xyz1, F0, F1, subsample_size=-1, return_other_half=False):
        subsample = len(F0) > subsample_size
        if subsample_size > 0 and subsample:
            N0 = min(len(F0), subsample_size)
            N1 = min(len(F1), subsample_size)
            inds0 = np.random.choice(len(F0), N0, replace=False)
            inds1 = np.random.choice(len(F1), N1, replace=False)
            F0, F1 = F0[inds0], F1[inds1]

        # Compute the nn
        nn_inds = find_nn_gpu(F0, F1, nn_max_n=self.config.nn_max_n)
        if not return_other_half:
            if subsample_size > 0 and subsample:
                return xyz0[inds0], xyz1[inds1[nn_inds]]
            else:
                return xyz0, xyz1[nn_inds]
        else:
            if subsample_size > 0 and subsample:
                inds0_unmatched = np.array(set(range(len(xyz0))) - set(inds0))
                inds1_unmatched = np.array(set(range(len(xyz1))) - set(inds1[nn_inds]))
                return xyz0[inds0], xyz1[inds1[nn_inds]], xyz0[inds0_unmatched], xyz1[inds1_unmatched]
            else:
                raise ValueError("This situation is not supported!")
                return xyz0, xyz1[nn_inds], np.array(range(len(xyz0))), np.array(range(len(xyz1)))[nn_inds]

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        return pts @ R.t() + T

    def evaluate_hit_ratio(self, xyz0, xyz1, T_gth, thresh=0.1):
        xyz0 = self.apply_transform(xyz0, T_gth)
        dist = np.sqrt(((xyz0 - xyz1)**2).sum(1) + 1e-6)
        return (dist < thresh).float().mean().item()

    def _save_checkpoint(self, epoch, filename='checkpoint'):
        state = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder_model.state_dict(),
            'generator_state_dict': self.generator_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'best_val': self.best_val,
            'best_val_epoch': self.best_val_epoch,
            'best_val_metric': self.best_val_metric
        }
        filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
        logging.info("Saving checkpoint: {} ...".format(filename))
        torch.save(state, filename)


class GenerativePairTrainer(TwoStageTrainer):
    def __init__(self, config, data_loader, val_data_loader):
        if val_data_loader is not None:
            assert val_data_loader.batch_size == 1, "Val set batch size must be 1 for now."
        super().__init__(config, data_loader, val_data_loader=val_data_loader)
        self.point_generation_ratio = config.point_generation_ratio
        self.regularization_strength = config.regularization_strength
        self.regularization_type = config.regularization_type
        self.neg_thresh = config.neg_thresh
        self.pos_thresh = config.pos_thresh
        self.neg_weight = config.neg_weight
        self.loss_ratio = config.loss_ratio
        self.alpha = 1e-1

    def generate_rand_negative_pairs(self, positive_pairs, hash_seed, N0, N1, N_neg=0):
        """
        Generate random negative pairs
        """
        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)
        if N_neg < 1:
            N_neg = positive_pairs.shape[0] * 2
        pos_keys = _hash(positive_pairs, hash_seed)

        neg_pairs = np.floor(np.random.rand(int(N_neg), 2) * np.array([[N0, N1]])).astype(
            np.int64)
        neg_keys = _hash(neg_pairs, hash_seed)
        mask = np.isin(neg_keys, pos_keys, assume_unique=False)
        return neg_pairs[np.logical_not(mask)]

    def contrastive_hardest_negative_loss(self,
                                          F0,
                                          F1,
                                          positive_pairs,
                                          num_pos=5192,
                                          num_hn_samples=2048,
                                          thresh=None):
        """
        Generate negative pairs
        """
        N0, N1 = len(F0), len(F1)
        N_pos_pairs = len(positive_pairs)
        hash_seed = max(N0, N1)
        sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
        sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

        if N_pos_pairs > num_pos:
            pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
            sample_pos_pairs = positive_pairs[pos_sel]
        else:
            sample_pos_pairs = positive_pairs

        # Find negatives for all F1[positive_pairs[:, 1]]
        subF0, subF1 = F0[sel0], F1[sel1]

        pos_ind0 = sample_pos_pairs[:, 0].long()
        pos_ind1 = sample_pos_pairs[:, 1].long()
        posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

        D01 = pdist(posF0, subF1, dist_type='L2')
        D10 = pdist(posF1, subF0, dist_type='L2')

        D01min, D01ind = D01.min(1)
        D10min, D10ind = D10.min(1)

        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)

        pos_keys = _hash(positive_pairs, hash_seed)

        D01ind = sel1[D01ind.cpu().numpy()]
        D10ind = sel0[D10ind.cpu().numpy()]
        neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
        neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

        mask0 = torch.from_numpy(
            np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
        mask1 = torch.from_numpy(
            np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
        pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
        neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
        neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)
        return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2

    def _train_epoch(self, epoch):
        gc.collect()
        self.encoder_model.train()
        self.generator_model.train()
        # Epoch starts from 1
        total_num = 0.0

        data_loader = self.data_loader
        data_loader_iter = self.data_loader.__iter__()

        iter_size = self.iter_size
        start_iter = (epoch - 1) * (len(data_loader) // iter_size)

        data_meter = AverageMeter()
        data_timer, total_timer, feat_timer, recon_timer, excess_timer = Timer(), Timer(), Timer(), Timer(), Timer()

        # torch.autograd.set_detect_anomaly(True)

        # Main training
        num_iter = len(data_loader) // iter_size
        for curr_iter in range(num_iter):
            self.optimizer.zero_grad()
            batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

            data_time = 0
            total_timer.tic()
            for iter_idx in range(iter_size):
                # Load a batch of data
                data_timer.tic()
                input_dict = data_loader_iter.next()
                data_time += data_timer.toc(average=False)

                # Go through the encoder network to generate per-point features
                feat_timer.tic()
                sinput0 = ME.SparseTensor(
                    input_dict['sinput0_F'].to(self.device),
                    coordinates=input_dict['sinput0_C'].to(self.device))
                encoded_0 = self.encoder_model(sinput0)
                F0 = encoded_0.F

                sinput1 = ME.SparseTensor(
                    input_dict['sinput1_F'].to(self.device),
                    coordinates=input_dict['sinput1_C'].to(self.device))
                encoded_1 = self.encoder_model(sinput1)
                F1 = encoded_1.F
                feat_timer.toc()

                # ----original FCGF Hardest Contrastive loss----
                pos_pairs = input_dict['correspondences']
                pos_loss, neg_loss = self.contrastive_hardest_negative_loss(
                    F0,
                    F1,
                    pos_pairs,
                    num_pos=self.config.num_pos_per_batch * self.config.batch_size,
                    num_hn_samples=self.config.num_hn_samples_per_batch *
                    self.config.batch_size)

                pos_loss /= iter_size
                neg_loss /= iter_size
                loss = pos_loss + self.neg_weight * neg_loss

                # ----generative loss----
                # we denote the output the FCGF as "encoded" coords and features
                if self.config.symmetric:
                    generated_0 = self.generator_model(encoded_0)
                    generated_1 = self.generator_model(encoded_1)
                    _, batch_gen_feats0 = \
                        generated_0.decomposed_coordinates_and_features
                    _, batch_gen_feats1 = \
                        generated_1.decomposed_coordinates_and_features

                # reconstruction loss for frame 0:
                batch_enc_coords, batch_enc_feats = \
                    encoded_0.decomposed_coordinates_and_features
                for i in range(len(batch_enc_coords)):
                    recon_timer.tic()
                    # let the generator generate the residual of new points
                    if self.config.symmetric:
                        generated = batch_gen_feats0[i] * self.voxel_size
                    else:
                        generated = self.generator_model(batch_enc_feats[i].to(self.device)) * self.voxel_size
                    
                    if self.regularization_type == 'L2':
                        regularize_loss = torch.mean(torch.sum((generated.reshape(-1, 3))**2, axis=-1))
                    elif self.regularization_type == 'RepelL2':
                        squared_tmp = torch.sum((generated.reshape(-1, 3))**2, axis=-1)
                        regularize_loss = torch.mean(squared_tmp) + \
                                          torch.mean(1.0 / (squared_tmp + self.alpha))
                    elif self.regularization_type == 'RepelL1':
                        lengths = torch.pow(torch.sum((generated.reshape(-1, 3))**2, axis=-1) + 1e-5, 0.25) - 1
                        regularize_loss = torch.mean(lengths**2)
                    mod_generated = (generated + self.voxel_size * \
                                     batch_enc_coords[i].repeat(1, self.point_generation_ratio)).reshape(-1, 3)
                    recon_timer.toc(accumulate=True)

                    excess_timer.tic()
                    loss += (self.chamfer_distance(mod_generated, 
                                                   input_dict['pcd_nghb0'][i]) + \
                            regularize_loss * self.regularization_strength) * self.loss_ratio
                    excess_timer.toc(accumulate=True)
                recon_timer.incCount()
                excess_timer.incCount()

                # reconstruction loss for frame 1:
                batch_enc_coords, batch_enc_feats = \
                    encoded_1.decomposed_coordinates_and_features
                for i in range(len(batch_enc_coords)):
                    recon_timer.tic()
                    # let the generator generate the residual of new points
                    if self.config.symmetric:
                        generated = batch_gen_feats1[i] * self.voxel_size
                    else:
                        generated = self.generator_model(batch_enc_feats[i].to(self.device)) * self.voxel_size

                    if self.regularization_type == 'L2':
                        regularize_loss = torch.mean(torch.sum((generated.reshape(-1, 3))**2, axis=-1))
                    elif self.regularization_type == 'RepelL2':
                        squared_tmp = torch.sum((generated.reshape(-1, 3))**2, axis=-1)
                        regularize_loss = torch.mean(squared_tmp) + \
                                          torch.mean(1.0 / (squared_tmp + self.alpha))
                    elif self.regularization_type == 'RepelL1':
                        lengths = torch.pow(torch.sum((generated.reshape(-1, 3))**2, axis=-1) + 1e-5, 0.25) - 1
                        regularize_loss = torch.mean(lengths**2)
                    mod_generated = (generated + self.voxel_size * \
                                     batch_enc_coords[i].repeat(1, self.point_generation_ratio)).reshape(-1, 3)
                    recon_timer.toc(accumulate=True)

                    excess_timer.tic()
                    loss += (self.chamfer_distance(mod_generated, 
                                                   input_dict['pcd_nghb1'][i]) + \
                            regularize_loss * self.regularization_strength) * self.loss_ratio
                    excess_timer.toc(accumulate=True)
                recon_timer.incCount()
                excess_timer.incCount()

                loss.backward()

                # accumulate batch loss for recording
                batch_loss += loss.item()
                batch_pos_loss += pos_loss.item()
                batch_neg_loss += neg_loss.item()

            self.optimizer.step()
            torch.cuda.empty_cache()

            total_num += 1.0
            total_timer.toc()
            data_meter.update(data_time)

            # Print logs
            if curr_iter % self.config.stat_freq == 0:
                self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
                self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
                self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
                logging.info(
                    "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
                    .format(epoch, curr_iter,
                            len(self.data_loader) //
                            iter_size, batch_loss, batch_pos_loss, batch_neg_loss) +
                    "\tData time: {:.4f}, Feat time: {:.4f}, Recon time: {:.4f}, Chamfer time: {:.4f} Iter time: {:.4f}".format(
                        data_timer.avg, feat_timer.avg, recon_timer.avg, excess_timer.avg, total_timer.avg))
                data_meter.reset()
                total_timer.reset()

    def _valid_epoch(self):
        # Change the network to evaluation mode
        self.encoder_model.eval()
        self.generator_model.eval()
        self.val_data_loader.dataset.reset_seed(0)
        num_data = 0
        loss_meter, chamfer_meter, regularize_meter, hit_ratio_meter, \
            feat_match_ratio, rte_meter, rre_meter = AverageMeter(), AverageMeter(), \
            AverageMeter(),  AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        data_timer, feat_timer, matching_timer, recon_timer, excess_timer = \
            Timer(), Timer(), Timer(), Timer(), Timer()
        tot_num_data = len(self.val_data_loader.dataset)
        if self.val_max_iter > 0:
            tot_num_data = min(self.val_max_iter, tot_num_data)
        data_loader_iter = self.val_data_loader.__iter__()

        for batch_idx in range(tot_num_data):
            data_timer.tic()
            try:
                input_dict = data_loader_iter.next()
            except StopIteration:
                break
            data_timer.toc()

            # Go through the encoder network to generate per-point features
            feat_timer.tic()
            sinput0 = ME.SparseTensor(
                input_dict['sinput0_F'].to(self.device),
                coordinates=input_dict['sinput0_C'].to(self.device))
            encoded_0 = self.encoder_model(sinput0)
            F0 = encoded_0.F

            sinput1 = ME.SparseTensor(
                input_dict['sinput1_F'].to(self.device),
                coordinates=input_dict['sinput1_C'].to(self.device))
            encoded_1 = self.encoder_model(sinput1)
            F1 = encoded_1.F
            feat_timer.toc()

            # match features and calculate transformation
            matching_timer.tic()
            xyz0, xyz1, T_gt = input_dict['pcd0'][0], input_dict['pcd1'][0], input_dict['T_gt']
            xyz0_corr, xyz1_corr = self.find_corr(xyz0, xyz1, F0, F1, subsample_size=5000)
            T_est = te.est_quad_linear_robust(xyz0_corr, xyz1_corr)

            loss = corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
            loss_meter.update(loss)

            rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
            rte_meter.update(rte)
            rre = np.arccos((np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
            if not np.isnan(rre):
                rre_meter.update(rre)

            hit_ratio = self.evaluate_hit_ratio(
                xyz0_corr, xyz1_corr, T_gt, thresh=self.config.hit_ratio_thresh)
            hit_ratio_meter.update(hit_ratio)
            feat_match_ratio.update(hit_ratio > 0.05)
            matching_timer.toc()

            # we denote the output the FCGF as "encoded" coords and features
            if self.config.symmetric:
                generated_0 = self.generator_model(encoded_0)
                generated_1 = self.generator_model(encoded_1)
                _, batch_gen_feats0 = \
                    generated_0.decomposed_coordinates_and_features
                _, batch_gen_feats1 = \
                    generated_1.decomposed_coordinates_and_features

            # loss for frame 0
            batch_enc_coords, batch_enc_feats = \
                encoded_0.decomposed_coordinates_and_features
            raw_reg_loss = 0
            loss_chamfer = 0
            for i in range(len(batch_enc_coords)):
                recon_timer.tic()
                if self.config.symmetric:
                    generated = batch_gen_feats0[i] * self.voxel_size
                else:
                    generated = self.generator_model(batch_enc_feats[i].to(self.device)) * self.voxel_size
                mod_generated = generated.reshape(-1, 3)
                if self.regularization_type == 'L2':
                    raw_reg_loss += torch.mean(torch.sum(mod_generated**2, axis=-1))
                elif self.regularization_type == 'RepelL2':
                    squared_tmp = torch.sum(mod_generated**2, axis=-1)
                    raw_reg_loss += torch.mean(squared_tmp) + \
                                    torch.mean(1.0 / (squared_tmp + self.alpha))
                elif self.regularization_type == 'RepelL1':
                    lengths = torch.pow(torch.sum((generated.reshape(-1, 3))**2, axis=-1) + 1e-5, 0.25) - 1
                    raw_reg_loss = torch.mean(lengths**2)
                del mod_generated
                # let the generator generate the residual of new points
                generated = (generated + self.voxel_size * \
                             batch_enc_coords[i].repeat(1, self.point_generation_ratio)).reshape(-1, 3)
                recon_timer.toc(accumulate=True)

                excess_timer.tic()
                loss_chamfer += self.chamfer_distance(generated, input_dict['pcd_nghb0'][i])
                excess_timer.toc(accumulate=True)

            recon_timer.incCount()
            excess_timer.incCount()

            # loss for frame 1
            batch_enc_coords, batch_enc_feats = \
                encoded_1.decomposed_coordinates_and_features
            for i in range(len(batch_enc_coords)):
                recon_timer.tic()
                if self.config.symmetric:
                    generated = batch_gen_feats1[i] * self.voxel_size
                else:
                    generated = self.generator_model(batch_enc_feats[i].to(self.device)) * self.voxel_size
                mod_generated = generated.reshape(-1, 3)
                if self.regularization_type == 'L2':
                    raw_reg_loss += torch.mean(torch.sum(mod_generated**2, axis=-1))
                elif self.regularization_type == 'RepelL2':
                    squared_tmp = torch.sum(mod_generated**2, axis=-1)
                    raw_reg_loss += torch.mean(squared_tmp) + \
                                    torch.mean(1.0 / (squared_tmp + self.alpha))
                elif self.regularization_type == 'RepelL1':
                    lengths = torch.pow(torch.sum((generated.reshape(-1, 3))**2, axis=-1) + 1e-5, 0.25) - 1
                    raw_reg_loss = torch.mean(lengths**2)
                del mod_generated
                # let the generator generate the residual of new points
                generated = (generated + self.voxel_size * \
                             batch_enc_coords[i].repeat(1, self.point_generation_ratio)).reshape(-1, 3)
                recon_timer.toc(accumulate=True)

                excess_timer.tic()
                loss_chamfer += self.chamfer_distance(generated, input_dict['pcd_nghb0'][i])
                excess_timer.toc(accumulate=True)
            recon_timer.incCount()
            excess_timer.incCount()
            raw_reg_loss /= (2 * len(batch_enc_coords))
            loss_chamfer /= (2 * len(batch_enc_coords))
            
            loss_regularize = raw_reg_loss * self.regularization_strength
            loss_meter.update((loss_chamfer + loss_regularize).item())
            chamfer_meter.update(loss_chamfer.item())
            regularize_meter.update(raw_reg_loss.item())
            num_data += 1
            torch.cuda.empty_cache()

            if batch_idx % 100 == 0:
                logging.info(' '.join([
                    f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
                    f"Feature Extraction Time: {feat_timer.avg:.3f}, Matching Time: {matching_timer.avg:.3f},",
                    f"Reconstruction Time: {recon_timer.avg:.3f}, Excess Time: {excess_timer.avg:.3f},",
                    f"Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
                    f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}",
                    f"Chamfer: {chamfer_meter.avg:.3f}, regularize: {regularize_meter.avg:.3f}"
                ]))
                data_timer.reset()

        logging.info(' '.join([
            f"Final Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
            f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}",
            f"Chamfer: {chamfer_meter.avg:.3f}, regularize: {regularize_meter.avg:.3f}"
        ]))
        return {
            "loss": loss_meter.avg,
            "rre": rre_meter.avg,
            "rte": rte_meter.avg,
            'feat_match_ratio': feat_match_ratio.avg,
            'hit_ratio': hit_ratio_meter.avg,
            "chamfer_distance": chamfer_meter.avg,
            "regularize_loss": regularize_meter.avg
        }
