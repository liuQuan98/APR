import time, os, torch,copy
import numpy as np
import torch.nn as nn
from chamferdist import ChamferDistance
from tensorboardX import SummaryWriter
from lib.timer import Timer, AverageMeter
from lib.utils import Logger,validate_gradient
# from lib.loss import GenerateChamferLoss

from tqdm import tqdm
import torch.nn.functional as F
import gc


class Trainer(object):
    def __init__(self, args):
        self.config = args
        # parameters
        self.mode = args.mode
        self.start_epoch = 1
        self.max_epoch = args.max_epoch
        self.save_dir = args.save_dir
        self.device = args.device
        self.verbose = args.verbose
        self.max_points = args.max_points

        self.model = args.model.to(self.device)
        self.model_name = args.model_name
        self.generative_model = args.generative_model.to(self.device)   # RCAR generative model
        self.loss_ratio = args.loss_ratio                               # RCAR loss mix ratio
        self.point_generation_ratio = args.point_generation_ratio       # RCAR point generation ratio
        self.regularization_strength = args.regularization_strength     # RCAR regularization strength
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_freq = args.scheduler_freq
        self.snapshot_freq = args.snapshot_freq
        self.snapshot_dir = args.snapshot_dir 
        self.benchmark = args.benchmark
        self.iter_size = args.iter_size
        self.verbose_freq= args.verbose_freq

        self.w_circle_loss = args.w_circle_loss
        self.w_overlap_loss = args.w_overlap_loss
        self.w_saliency_loss = args.w_saliency_loss 
        self.desc_loss = args.desc_loss
        # self.chamfer_loss = GenerateChamferLoss(args)

        self.symmetric = args.symmetric

        self.best_loss = 1e5
        self.best_recall = -1e5
        self.writer = SummaryWriter(log_dir=args.tboard_dir)
        self.logger = Logger(args.snapshot_dir)
        self.logger.write(f'#parameters {sum([x.nelement() for x in self.model.parameters()])/1000000.} M\n')

        print(self.model)

        if (args.pretrain !=''):
            self._load_pretrain(args.pretrain, args)
        
        self.loader =dict()
        self.loader['train']=args.train_loader
        self.loader['val']=args.val_loader
        self.loader['test'] = args.test_loader

        with open(f'{args.snapshot_dir}/model','w') as f:
            f.write(str(self.model))
        f.close()
 
    def _snapshot(self, epoch, name=None):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'generative_model_state_dict': self.generative_model.state_dict(), 
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_recall': self.best_recall
        }
        if name is None:
            filename = os.path.join(self.save_dir, f'model_{epoch}.pth')
        else:
            filename = os.path.join(self.save_dir, f'model_{name}.pth')
        self.logger.write(f"Save model to {filename}\n")
        torch.save(state, filename)

    def _load_pretrain(self, resume, config):
        if os.path.isfile(resume):
            state = torch.load(resume)
            if 'pretrain_restart' in [k for (k, v) in config.items()] and config.pretrain_restart:
                self.logger.write(f'Restart pretrain, only loading model weights.\n')
            else:
                self.start_epoch = state['epoch']
                self.scheduler.load_state_dict(state['scheduler'])
                self.optimizer.load_state_dict(state['optimizer'])
                self.best_loss = state['best_loss']
                self.best_recall = state['best_recall']
            self.model.load_state_dict(state['state_dict'])
            self.generative_model.load_state_dict(state['generative_model_state_dict'])
            
            self.logger.write(f'Successfully load pretrained model from {resume}!\n')
            self.logger.write(f'Current best loss {self.best_loss}\n')
            self.logger.write(f'Current best recall {self.best_recall}\n')
        else:
            raise ValueError(f"=> no checkpoint found at '{resume}'")

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']

    def stats_dict(self):
        stats=dict()
        stats['circle_loss']=0.
        stats['recall']=0.          # feature match recall, divided by number of ground truth pairs
        stats['chamfer_loss']=0.    # rcar chamfer loss
        stats['regularization_loss']=0. # rcar regularization loss
        stats['saliency_loss'] = 0.
        stats['saliency_recall'] = 0.
        stats['saliency_precision'] = 0.
        stats['overlap_loss'] = 0.
        stats['overlap_recall']=0.
        stats['overlap_precision']=0.
        return stats

    def stats_meter(self):
        meters=dict()
        stats=self.stats_dict()
        for key,_ in stats.items():
            meters[key]=AverageMeter()
        return meters

    def chamfer_distance(self, array1, array2):
        cd_dist = ChamferDistance()
        n1 = len(array1)
        n2 = len(array2)
        array1 = torch.unsqueeze(array1, dim=0).to(self.device)
        array2 = torch.unsqueeze(array2, dim=0).to(self.device)
        forward_cd_dist = cd_dist(array1, array2)
        backward_cd_dist = cd_dist(array2, array1)
        # print(f"reconstruction: len1: {n1}, len2: {n2}, forward: {forward_cd_dist}, backward: {backward_cd_dist}")
        return forward_cd_dist / n1 + backward_cd_dist / n2

    def inference_one_batch(self, inputs, phase):
        assert phase in ['train','val','test']
        ##################################
        # training
        if(phase == 'train'):
            self.model.train()
            ###############################################
            # forward pass
            if self.model_name == "KPFCNN":
                feats, scores_overlap, scores_saliency = self.model(inputs)  #[N1, C1], [N2, C2]
            pcd = inputs['points'][0]
            len_src = inputs['stack_lengths'][0][0]
            c_rot, c_trans = inputs['rot'], inputs['trans']
            correspondence = inputs['correspondences']

            src_pcd, tgt_pcd = inputs['src_pcd_raw'], inputs['tgt_pcd_raw']
            src_feats, tgt_feats = feats[:len_src], feats[len_src:]

            if self.symmetric:
                inputs['second_features'] = feats
                symmetric_generated = self.generative_model(inputs)
                generated_0 = symmetric_generated[:len_src]
                generated_1 = symmetric_generated[len_src:]

            # ----generative loss----
            generative_loss = 0 # weighted sum of chamfer loss and regularization loss
            chamfer_loss = 0
            regularization_loss = 0
            # reconstruction loss for frame 0:
            # let the generator generate the offsets of new points
            if self.symmetric:
                generated = generated_0
            else:
                generated = self.generative_model(src_feats)
            regularize_loss = torch.mean(torch.sum((generated.reshape(-1, 3))**2, axis=-1))
            regularization_loss += regularize_loss
            mod_generated = (generated + src_pcd.repeat(1, self.point_generation_ratio)).reshape(-1, 3)
            chamfer_loss_raw = self.chamfer_distance(mod_generated, 
                                                      inputs['src_nghb'].to(self.device))
            chamfer_loss += chamfer_loss_raw
            generative_loss += (chamfer_loss_raw + \
                                regularize_loss * self.regularization_strength) * self.loss_ratio

            invalid_flag = False
            if torch.isnan(chamfer_loss_raw):
                print(f"0 invalid: {len(mod_generated)}, {len(inputs['src_nghb'])}")
                invalid_flag = True

            # np.savez("points.npz", raw=inputs['src_pcd_raw'].cpu().detach().numpy(), 
            #                     recon=mod_generated.cpu().detach().numpy(),
            #                     APC=inputs['src_nghb'].cpu().detach().numpy())
            # raise ValueError

            # reconstruction loss for frame 1:
            if self.symmetric:
                generated = generated_1
            else:
                generated = self.generative_model(tgt_feats)
            regularize_loss = torch.mean(torch.sum((generated.reshape(-1, 3))**2, axis=-1))
            regularization_loss += regularize_loss
            mod_generated = (generated + tgt_pcd.repeat(1, self.point_generation_ratio)).reshape(-1, 3)
            chamfer_loss_raw = self.chamfer_distance(mod_generated, 
                                                      inputs['tgt_nghb'].to(self.device))
            chamfer_loss += chamfer_loss_raw
            generative_loss += (chamfer_loss_raw + \
                                regularize_loss * self.regularization_strength) * self.loss_ratio

            if torch.isnan(chamfer_loss_raw):
                print(f"1 invalid: {len(mod_generated)}, {len(inputs['tgt_nghb'])}")
                invalid_flag = True

            ###################################################
            # get loss
            stats= self.desc_loss(src_pcd, tgt_pcd, src_feats, tgt_feats,correspondence, c_rot, c_trans, scores_overlap, scores_saliency)

            c_loss = stats['circle_loss'] * self.w_circle_loss + stats['overlap_loss'] * self.w_overlap_loss + stats['saliency_loss'] * self.w_saliency_loss

            c_loss += generative_loss
            if not invalid_flag:
                c_loss.backward()

        else:
            self.model.eval()
            with torch.no_grad():
                ###############################################
                # forward pass
                feats, scores_overlap, scores_saliency = self.model(inputs)  #[N1, C1], [N2, C2]
                # pcd =  inputs['points'][0]
                len_src = inputs['stack_lengths'][0][0]
                c_rot, c_trans = inputs['rot'], inputs['trans']
                correspondence = inputs['correspondences']

                src_pcd, tgt_pcd = inputs['src_pcd_raw'], inputs['tgt_pcd_raw']
                src_feats, tgt_feats = feats[:len_src], feats[len_src:]

                if self.symmetric:
                    inputs['second_feature'] = feats
                    symmetric_generated, _, _ = self.generative_model(inputs)
                    generated_0 = symmetric_generated[:len_src]
                    generated_1 = symmetric_generated[len_src:]

                # ----generative loss----
                generative_loss = 0 # weighted sum of chamfer loss and regularization loss
                chamfer_loss = 0
                regularization_loss = 0
                # reconstruction loss for frame 0:
                # let the generator generate the offsets of new points
                if self.symmetric:
                    generated = generated_0
                else:
                    generated = self.generative_model(src_feats)
                regularization_loss += torch.mean(torch.sum((generated.reshape(-1, 3))**2, axis=-1))
                mod_generated = (generated + src_pcd.repeat(1, self.point_generation_ratio)).reshape(-1, 3)
                chamfer_loss += self.chamfer_distance(mod_generated, inputs['src_nghb'].to(self.device))

                # reconstruction loss for frame 1:
                if self.symmetric:
                    generated = generated_1
                else:
                    generated = self.generative_model(tgt_feats)
                regularization_loss += torch.mean(torch.sum((generated.reshape(-1, 3))**2, axis=-1))
                mod_generated = (generated + tgt_pcd.repeat(1, self.point_generation_ratio)).reshape(-1, 3)
                chamfer_loss += self.chamfer_distance(mod_generated, inputs['tgt_nghb'].to(self.device))
                invalid_flag = False

                ###################################################
                # get loss
                stats= self.desc_loss(src_pcd, tgt_pcd, src_feats, tgt_feats,correspondence, c_rot, c_trans, scores_overlap, scores_saliency)
                # print(stats)

        ##################################        
        # detach the gradients for loss terms
        stats['circle_loss'] = float(stats['circle_loss'].detach())
        stats['overlap_loss'] = float(stats['overlap_loss'].detach())
        stats['saliency_loss'] = float(stats['saliency_loss'].detach())
        stats['chamfer_loss'] = float(chamfer_loss.detach())
        stats['regularization_loss'] = float(regularization_loss.detach())
        
        return stats, invalid_flag


    def inference_one_epoch(self,epoch, phase):
        gc.collect()
        assert phase in ['train','val','test']

        # init stats meter
        stats_meter = self.stats_meter()

        num_iter = int(len(self.loader[phase].dataset) // self.loader[phase].batch_size)
        c_loader_iter = self.loader[phase].__iter__()

        self.optimizer.zero_grad()
        for c_iter in tqdm(range(num_iter)): # loop through this epoch   
            ##################################
            # load inputs to device.
            inputs = c_loader_iter.next()

            for k, v in inputs.items():  
                if type(v) == list:
                    inputs[k] = [item.to(self.device) for item in v]
                elif type(v) == dict:
                    pass
                else:
                    inputs[k] = v.to(self.device)
            try:
                ##################################
                # forward pass
                # with torch.autograd.detect_anomaly():
                stats, invalid_flag = self.inference_one_batch(inputs, phase)
                if invalid_flag:
                    raise ValueError("Empty pair found.")
                
                ###################################################
                # run optimisation
                if((c_iter+1) % self.iter_size == 0 and phase == 'train'):
                    gradient_valid = validate_gradient(self.model)
                    if(gradient_valid):
                        self.optimizer.step()
                    else:
                        self.logger.write('gradient not valid\n')
                    self.optimizer.zero_grad()
                
                ################################
                # update to stats_meter
                for key,value in stats.items():
                    stats_meter[key].update(value)
            except Exception as inst:
                print(inst)
                # raise ValueError
            
            torch.cuda.empty_cache()
            
            if (c_iter + 1) % self.verbose_freq == 0 and self.verbose:
                curr_iter = num_iter * (epoch - 1) + c_iter
                for key, value in stats_meter.items():
                    self.writer.add_scalar(f'{phase}/{key}', value.avg, curr_iter)
                
                message = f'{phase} Epoch: {epoch} [{c_iter+1:4d}/{num_iter}]'
                for key,value in stats_meter.items():
                    message += f'{key}: {value.avg:.2f}\t'

                self.logger.write(message + '\n')

        message = f'{phase} Epoch: {epoch}'
        for key,value in stats_meter.items():
            message += f'{key}: {value.avg:.2f}\t'
        self.logger.write(message+'\n')

        return stats_meter


    def train(self):
        print('start training...')
        print(self.model, flush=True)
        print(self.generative_model, flush=True)
        for epoch in range(self.start_epoch, self.max_epoch):
            self.inference_one_epoch(epoch,'train')
            self.scheduler.step()
            
            stats_meter = self.inference_one_epoch(epoch,'val')
            
            if stats_meter['circle_loss'].avg < self.best_loss:
                self.best_loss = stats_meter['circle_loss'].avg
                self._snapshot(epoch,'best_loss')
            if stats_meter['recall'].avg > self.best_recall:
                self.best_recall = stats_meter['recall'].avg
                self._snapshot(epoch,'best_recall')
            
            # we only add saliency loss when we get descent point-wise features
            if(stats_meter['recall'].avg>0.3):
                self.w_saliency_loss = 1.
            else:
                self.w_saliency_loss = 0.
                    
        # finish all epoch
        print("Training finish!")


    def eval(self):
        print('Start to evaluate on validation datasets...')
        stats_meter = self.inference_one_epoch(0,'val')
        
        for key, value in stats_meter.items():
            print(key, value.avg)
