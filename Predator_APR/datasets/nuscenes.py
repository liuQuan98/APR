# totally not modified yet.
# require significant effort to put to use

# Basic libs
import os, time, glob, random, pickle, copy, torch
import numpy as np
import open3d
import pathlib
import MinkowskiEngine as ME
from scipy.spatial.transform import Rotation

# Dataset parent class
from torch.utils.data import Dataset
from lib.timer import Timer
from lib.benchmark_utils import to_tsfm, to_o3d_pcd, get_correspondences


class NUSCENESDataset(Dataset):
    """
    We follow D3Feat to add data augmentation part.
    We first voxelize the pcd and get matches
    Then we apply data augmentation to pcds. KPConv runs over processed pcds, but later for loss computation, we use pcds before data augmentation
    """
    icp_voxel_size = 0.05 # 0.05 meters, i.e. 5cm

    def __init__(self,config,split,data_augmentation=True):
        super(NUSCENESDataset,self).__init__()
        self.config = config
        self.root = os.path.join(config.root, split)
        self.voxel_size = config.first_subsampling_dl
        self.matching_search_voxel_size = config.overlap_radius
        self.data_augmentation = data_augmentation
        self.augment_noise = config.augment_noise
        self.max_corr = config.max_points
        self.augment_shift_range = config.augment_shift_range
        self.augment_scale_max = config.augment_scale_max
        self.augment_scale_min = config.augment_scale_min
        self.max_correspondence_distance_fine = self.icp_voxel_size * 1.5
        self.load_neighbourhood = True
        if config.mode == 'test':
            self.load_neighbourhood = False
            try:
                self.downsample_single = config.downsample_single
            except:
                self.downsample_single = 1.0
            print(f'Doensample Ratio: {self.downsample_single}')

        try:
            self.mutate_neighbour = (config.mutate_neighbour_percentage != 0)
            self.mutate_neighbour_percentage = config.mutate_neighbour_percentage
        except:
            self.mutate_neighbour = False
            self.mutate_neighbour_percentage = 0

        from lib.utils import Logger
        self.logger = Logger(config.snapshot_dir)

        # rcar data config
        self.MIN_DIST = config.pair_min_dist
        self.MAX_DIST = config.pair_max_dist
        # self.min_sample_frame_dist = config.min_sample_frame_dist
        self.complement_pair_dist = config.complement_pair_dist
        self.num_complement_one_side = config.num_complement_one_side
        self.complement_range = self.num_complement_one_side * self.complement_pair_dist

        # pose configuration: use old or new
        assert config.use_old_pose is True, "no slam-based position available!"

        self.icp_path = os.path.join(config.root,'icp')
        pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

        # Initiate containers
        self.files = []
        self.nuscenes_icp_cache = {}
        self.nuscenes_cache = {}
        self.split = split
        # load LoNuscenes point cloud pairs, instead of generating them based on distance
        if split == 'test' and config.LoNUSCENES == True:
            self.files = np.load("configs/nuscenes/file_LoNUSCENES_50.npy", allow_pickle=True)
        else:
            self.prepare_nuscenes_ply(split)
        # downsample dataset
        if split == 'train':
            self.files = self.files[::3]
            self.files = self.files[:1200]
        print(self.__len__())

    def prepare_nuscenes_ply(self, split='train'):
        pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

        print(f"Loading the subset {split} from {self.root}")
        
        subset_names = os.listdir(os.path.join(self.root, 'sequences'))

        for dirname in subset_names:
            print(f"Processing log {dirname}")
            fnames = glob.glob(self.root + '/sequences/%s/velodyne/*.bin' % dirname)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            all_pos = self.get_video_odometry(dirname, return_all=True)
            self.Ts = all_pos[:, :3, 3]

            curr_time = inames[min(int(self.complement_range * 5), int(len(inames)/2))]

            np.random.seed(0)
            while curr_time in inames:
                # calculate the distance (by random or not)
                dist_tmp = self.MIN_DIST + np.random.rand() * (self.MAX_DIST - self.MIN_DIST)
                
                right_dist = np.sqrt(((self.Ts[curr_time: curr_time+int(10*self.complement_range)] - 
                                    self.Ts[curr_time].reshape(1, 3))**2).sum(-1))
                # Find the min index
                next_time = np.where(right_dist > dist_tmp)[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    # Follow https://github.com/yewzijian/3DFeatNet/blob/master/scripts_data_processing/nuscenes/process_nuscenes_data.m#L44
                    next_time = next_time[0] + curr_time - 1
                    skip_0, cmpl_0 = self._get_complement_frames(curr_time)
                    skip_1, cmpl_1 = self._get_complement_frames(next_time)
                    if skip_0 or skip_1:
                        curr_time += 1
                    else:
                        self.files.append((dirname, curr_time, next_time, cmpl_0, cmpl_1))
                        curr_time = next_time + 1

    def _get_complement_frames(self, frame):
        # list of frame ids belonging to the neighbourhood of the current frame
        list_complement = []
        # indicates that there aren't enough complement frames around this frame
        # so that we should skip this frame
        skip_flag = False
        # Find the frames behind me
        left_frame_bound = max(0, frame-int(10*self.complement_range))
        left_dist = (self.Ts[left_frame_bound:frame] - self.Ts[frame].reshape(1, 3))**2
        left_dist = np.sqrt(left_dist.sum(-1))
        for i in range(self.num_complement_one_side):
            dist_tmp = self.complement_pair_dist * (i+1)
            candidates = np.where(left_dist > dist_tmp)[0]
            # print(candidates)
            if len(candidates) == 0:
                # No left-side complement detected
                skip_flag = True
                break
            else:
                list_complement.append(left_frame_bound + candidates[-1])
        
        if skip_flag:
            return (True, [])

        # Find the frames in front of me   
        right_dist = (self.Ts[frame: frame+int(10*self.complement_range)] - self.Ts[frame].reshape(1, 3))**2
        right_dist = np.sqrt(right_dist.sum(-1))
        for i in range(self.num_complement_one_side):
            dist_tmp = self.complement_pair_dist * (i+1)
            candidates = np.where(right_dist > dist_tmp)[0]
            if len(candidates) == 0:
                # No right-side complement detected
                skip_flag = True
                list_complement = []
                break
            else:
                list_complement.append(frame + candidates[0])
        return (skip_flag, list_complement)

    def __len__(self):
        return len(self.files)

    # simple function for getting the xyz point-cloud w.r.t log and time
    def _get_xyz(self, dirname, time):
        fname = self._get_velodyne_fn(dirname, time)
        xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
        return xyzr[:, :3]

    def __getitem__(self, idx):
        if self.load_neighbourhood:
            prepare_timer, icp_timer, rot_crop_timer = Timer(), Timer(), Timer()
            prepare_timer.tic()
            dirname, t_0, t_1, t_cmpl_0, t_cmpl_1 = self.files[idx]
            # print(self.files[idx])
            positions = self.get_video_odometry(dirname, [t_0, t_1] + t_cmpl_0 + t_cmpl_1)

            pos_0, pos_1 = positions[0:2]
            pos_cmpl_all = positions[2:]
            pos_cmpl0 = pos_cmpl_all[:2*self.num_complement_one_side]
            pos_cmpl1 = pos_cmpl_all[2*self.num_complement_one_side:]

            if self.mutate_neighbour:
                for pos_cmpl in [pos_cmpl0, pos_cmpl1]:  # two frames: t0 & t1
                    # We denote the position-disturbed heighbourhood frames as 'victims'.
                    num_victims = int(self.mutate_neighbour_percentage*2*self.num_complement_one_side)
                    victim_idxs = np.random.choice(2*self.num_complement_one_side, num_victims, replace=False)
                    for vic_id in victim_idxs:
                        euler_angles=(np.random.rand(3)-0.5)*np.pi*2 # anglez, angley, anglex
                        rot_mutate= Rotation.from_euler('zyx', euler_angles).as_matrix()
                        pos_cmpl[vic_id][:3,:3] = np.dot(pos_cmpl[vic_id][:3,:3], rot_mutate)

            # load two center point clouds
            xyz_0 = self._get_xyz(dirname, t_0)
            xyz_1 = self._get_xyz(dirname, t_1)

            # load neighbourhood point clouds
            xyz_cmpl_0 = []
            xyz_cmpl_1 = []
            for (t_tmp_0, t_tmp_1) in zip(t_cmpl_0, t_cmpl_1):
                xyz_cmpl_0.append(self._get_xyz(dirname, t_tmp_0))
                xyz_cmpl_1.append(self._get_xyz(dirname, t_tmp_1))
            prepare_timer.toc()

            icp_timer.tic()

            def GetListM(pos_core, pos_cmpls):
                return [np.linalg.inv(pos_core) @ pos_cmpls[i] for i in range(0, self.num_complement_one_side)] + \
                        [np.linalg.inv(pos_core) @ pos_cmpls[i] for i in range(self.num_complement_one_side, len(pos_cmpls))]
            list_M_0 = GetListM(pos_0, pos_cmpl0)
            list_M_1 = GetListM(pos_1, pos_cmpl1)

            # determine rot&trans matrices of the complement frames
            xyz_cmpl_0 = [self.apply_transform(xyz_k, M_k) 
                        for xyz_k, M_k in zip(xyz_cmpl_0, list_M_0)]
            xyz_cmpl_1 = [self.apply_transform(xyz_k, M_k) 
                        for xyz_k, M_k in zip(xyz_cmpl_1, list_M_1)]

            # obtain relative pose
            M2 = np.linalg.inv(positions[1]) @ positions[0]

            icp_timer.toc()
                
            rot_crop_timer.tic()
            # abandon all points that lie out of the scope of the center frame
            # this is because we cannot ask the network to fully imagine
            #   what's there where not even one supporting point exists
            max_dist_square_0 = np.max((xyz_0**2).sum(-1))
            max_dist_square_1 = np.max((xyz_1**2).sum(-1))
            xyz_cmpl_0 = np.concatenate(xyz_cmpl_0, axis=0)
            xyz_cmpl_1 = np.concatenate(xyz_cmpl_1, axis=0)
            xyz_nghb_0 = xyz_cmpl_0[np.where((xyz_cmpl_0**2).sum(-1) < max_dist_square_0)[0]]
            xyz_nghb_1 = xyz_cmpl_1[np.where((xyz_cmpl_1**2).sum(-1) < max_dist_square_1)[0]]
            del xyz_cmpl_0
            del xyz_cmpl_1

            # refined pose is denoted as trans
            tsfm = M2
            rot = tsfm[:3,:3]
            trans = tsfm[:3,3][:,None]

            # apply downsampling on one side during testing
            if self.split == 'test' and self.downsample_single != 1.0:
                indices = np.random.choice(len(xyz_0), int(len(xyz_0)*self.downsample_single))
                xyz_0 = xyz_0[indices]

            # voxelize the point clouds here
            pcd0 = self.make_open3d_point_cloud(xyz_0)
            pcd1 = self.make_open3d_point_cloud(xyz_1)
            pcd0_nghb = self.make_open3d_point_cloud(xyz_nghb_0)
            pcd1_nghb = self.make_open3d_point_cloud(xyz_nghb_1)
            pcd0 = pcd0.voxel_down_sample(voxel_size=self.voxel_size)
            pcd1 = pcd1.voxel_down_sample(voxel_size=self.voxel_size)
            pcd0_nghb = pcd0_nghb.voxel_down_sample(voxel_size=self.voxel_size)
            pcd1_nghb = pcd1_nghb.voxel_down_sample(voxel_size=self.voxel_size)
            src_pcd = np.array(pcd0.points)
            tgt_pcd = np.array(pcd1.points)
            src_nghb = np.array(pcd0_nghb.points)
            tgt_nghb = np.array(pcd1_nghb.points)

            # Get matches
            matching_inds = get_correspondences(pcd0, pcd1, tsfm, self.matching_search_voxel_size)
            if(matching_inds.size(0) < self.max_corr and self.split == 'train'):
                return self.__getitem__(np.random.choice(len(self.files),1)[0])

            src_feats=np.ones_like(src_pcd[:,:1]).astype(np.float32)
            tgt_feats=np.ones_like(tgt_pcd[:,:1]).astype(np.float32)

            rot = rot.astype(np.float32)
            trans = trans.astype(np.float32)

            # add data augmentation
            src_pcd_input = copy.deepcopy(src_pcd)
            tgt_pcd_input = copy.deepcopy(tgt_pcd)
            if(self.data_augmentation):
                # add gaussian noise
                src_pcd_input += (np.random.rand(src_pcd_input.shape[0],3) - 0.5) * self.augment_noise
                tgt_pcd_input += (np.random.rand(tgt_pcd_input.shape[0],3) - 0.5) * self.augment_noise

                # rotate the point cloud
                euler_ab=np.random.rand(3)*np.pi*2 # anglez, angley, anglex
                rot_ab= Rotation.from_euler('zyx', euler_ab).as_matrix()
                if(np.random.rand(1)[0]>0.5):
                    src_pcd_input = np.dot(rot_ab, src_pcd_input.T).T
                else:
                    tgt_pcd_input = np.dot(rot_ab, tgt_pcd_input.T).T
                
                # scale the pcd
                scale = self.augment_scale_min + (self.augment_scale_max - self.augment_scale_min) * random.random()
                src_pcd_input = src_pcd_input * scale
                tgt_pcd_input = tgt_pcd_input * scale

                # shift the pcd
                shift_src = np.random.uniform(-self.augment_shift_range, self.augment_shift_range, 3)
                shift_tgt = np.random.uniform(-self.augment_shift_range, self.augment_shift_range, 3)

                src_pcd_input = src_pcd_input + shift_src
                tgt_pcd_input = tgt_pcd_input + shift_tgt
            rot_crop_timer.toc()
            # message = f"Data loading time: prepare: {prepare_timer.avg}, icp: {icp_timer.avg}, " + \
            #           f"rotate & crop: {rot_crop_timer.avg}"
            # print(message)
            # self.logger.write(message + '\n')

            return src_pcd_input, tgt_pcd_input, src_feats, tgt_feats, rot, trans, matching_inds, src_pcd, tgt_pcd, src_nghb, tgt_nghb, torch.ones(1)
        else:   # if we are doing inference, then the neighbourhood framed are not needed anymore
            prepare_timer, icp_timer, rot_crop_timer = Timer(), Timer(), Timer()
            prepare_timer.tic()
            
            try:
                dirname, t_0, t_1, t_cmpl_0, t_cmpl_1 = self.files[idx]
            except:
                dirname, t_0, t_1 = self.files[idx]
            positions = self.get_video_odometry(dirname, [t_0, t_1])

            pos_0, pos_1 = positions[0:2]

            # load two center point clouds
            xyz_0 = self._get_xyz(dirname, t_0)
            xyz_1 = self._get_xyz(dirname, t_1)
            prepare_timer.toc()

            icp_timer.tic()
            M2 = np.linalg.inv(positions[1]) @ positions[0]
            icp_timer.toc()
                
            rot_crop_timer.tic()
            # refined pose is denoted as trans
            tsfm = M2
            rot = tsfm[:3,:3]
            trans = tsfm[:3,3][:,None]

            # apply downsampling on one side during testing
            if self.split == 'test' and self.downsample_single != 1.0:
                indices = np.random.choice(len(xyz_0), int(len(xyz_0)*self.downsample_single))
                xyz_0 = xyz_0[indices]

            # voxelize the point clouds here
            pcd0 = self.make_open3d_point_cloud(xyz_0)
            pcd1 = self.make_open3d_point_cloud(xyz_1)
            pcd0 = pcd0.voxel_down_sample(voxel_size=self.voxel_size)
            pcd1 = pcd1.voxel_down_sample(voxel_size=self.voxel_size)
            src_pcd = np.array(pcd0.points)
            tgt_pcd = np.array(pcd1.points)

            # Get matches
            matching_inds = get_correspondences(pcd0, pcd1, tsfm, self.matching_search_voxel_size)
            if(matching_inds.size(0) < self.max_corr and self.split == 'train'):
                return self.__getitem__(np.random.choice(len(self.files),1)[0])

            src_feats=np.ones_like(src_pcd[:,:1]).astype(np.float32)
            tgt_feats=np.ones_like(tgt_pcd[:,:1]).astype(np.float32)

            rot = rot.astype(np.float32)
            trans = trans.astype(np.float32)

            # add data augmentation
            src_pcd_input = copy.deepcopy(src_pcd)
            tgt_pcd_input = copy.deepcopy(tgt_pcd)
            if(self.data_augmentation):
                # add gaussian noise
                src_pcd_input += (np.random.rand(src_pcd_input.shape[0],3) - 0.5) * self.augment_noise
                tgt_pcd_input += (np.random.rand(tgt_pcd_input.shape[0],3) - 0.5) * self.augment_noise

                # rotate the point cloud
                euler_ab=np.random.rand(3)*np.pi*2 # anglez, angley, anglex
                rot_ab= Rotation.from_euler('zyx', euler_ab).as_matrix()
                if(np.random.rand(1)[0]>0.5):
                    src_pcd_input = np.dot(rot_ab, src_pcd_input.T).T
                else:
                    tgt_pcd_input = np.dot(rot_ab, tgt_pcd_input.T).T
                
                # scale the pcd
                scale = self.augment_scale_min + (self.augment_scale_max - self.augment_scale_min) * random.random()
                src_pcd_input = src_pcd_input * scale
                tgt_pcd_input = tgt_pcd_input * scale

                # shift the pcd
                shift_src = np.random.uniform(-self.augment_shift_range, self.augment_shift_range, 3)
                shift_tgt = np.random.uniform(-self.augment_shift_range, self.augment_shift_range, 3)

                src_pcd_input = src_pcd_input + shift_src
                tgt_pcd_input = tgt_pcd_input + shift_tgt
            rot_crop_timer.toc()
            # message = f"Data loading time: prepare: {prepare_timer.avg}, icp: {icp_timer.avg}, r&c: {rot_crop_timer.avg}, total: {prepare_timer.avg+icp_timer.avg+rot_crop_timer.avg}"
            # # print(message)
            # self.logger.write(message + '\n')

            return src_pcd_input, tgt_pcd_input, src_feats, tgt_feats, rot, trans, matching_inds, src_pcd, tgt_pcd, np.array([]), np.array([]), torch.ones(1)


    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def get_video_odometry(self, dirname, indices=None, ext='.txt', return_all=False):
        data_path = os.path.join(self.root, 'sequences', dirname, 'poses.npy')
        if data_path not in self.nuscenes_cache:
            self.nuscenes_cache[data_path] = np.load(data_path)
        if return_all:
            return self.nuscenes_cache[data_path]
        else:
            return self.nuscenes_cache[data_path][indices]

    def _get_velodyne_fn(self, dirname, t):
        fname = self.root + '/sequences/%s/velodyne/%06d.bin' % (dirname, t)
        return fname

    def make_open3d_point_cloud(self, xyz, color=None):
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(xyz)
        if color is not None:
            pcd.colors = open3d.utility.Vector3dVector(color)
        return pcd