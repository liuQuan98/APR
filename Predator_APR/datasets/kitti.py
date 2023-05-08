# Basic libs
from multiprocessing.sharedctypes import Value
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


class KITTIDataset(Dataset):
    """
    We follow D3Feat to add data augmentation part.
    We first voxelize the pcd and get matches
    Then we apply data augmentation to pcds. KPConv runs over processed pcds, but later for loss computation, we use pcds before data augmentation
    """
    DATA_FILES = {
        'train': [0,1,2,3,4,5],
        'val': [6,7],
        'test': [8,9,10]
    }
    discard_pairs =[(5, 1151, 1220), (2, 926, 962), (2, 2022, 2054), \
                    (1, 250, 266), (0, 3576, 3609), (2, 2943, 2979), \
                    (1, 411, 423), (2, 2241, 2271), (0, 1536, 1607), \
                    (0, 1338, 1439), (7, 784, 810), (2, 1471, 1498), \
                    (2, 3829, 3862), (0, 1780, 1840), (2, 3294, 3356), \
                    (2, 2420, 2453), (2, 4146, 4206), (0, 2781, 2829), \
                    (0, 3351, 3451), (1, 428, 444), (0, 3073, 3147)]
    icp_voxel_size = 0.05 # 0.05 meters, i.e. 5cm

    def __init__(self,config,split,data_augmentation=True):
        super(KITTIDataset,self).__init__()
        self.config = config
        self.root = os.path.join(config.root,'dataset')
        self.voxel_size = config.first_subsampling_dl
        self.matching_search_voxel_size = config.overlap_radius
        self.data_augmentation = data_augmentation or config.test_augmentation
        self.augment_noise = config.augment_noise
        self.IS_ODOMETRY = True
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
        try:
            self.use_old_pose = config.use_old_pose
        except:
            self.use_old_pose = True

        try:
            self.mutate_neighbour = (config.mutate_neighbour_percentage != 0)
            self.mutate_neighbour_percentage = config.mutate_neighbour_percentage
        except:
            self.mutate_neighbour = False
            self.mutate_neighbour_percentage = 0

        if self.use_old_pose:
            self.icp_path = os.path.join(config.root,'icp')
        else:
            self.icp_path = os.path.join(config.root,'icp_slam')
        pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

        # Initiate containers
        self.files = []
        self.kitti_icp_cache = {}
        self.kitti_cache = {}

        # load LoKITTI point cloud pairs, instead of generating them based on distance
        if split == 'test' and config.LoKITTI == True:
            self.files = np.load("configs/kitti/file_LoKITTI_50.npy")
            print("Loaded LoKITTI for split test!", flush=True)
        else:
            self.prepare_kitti_ply(split)
        self.split = split
        print(self.__len__())

    def prepare_kitti_ply(self, split='train'):
        pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

        print(f"Loading the subset {split} from {self.root}")

        subset_names = self.DATA_FILES[split]
        for dirname in subset_names:
            drive_id = int(dirname)
            print(f"Processing drive {drive_id}")
            fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            if self.use_old_pose:
                all_odo = self.get_video_odometry(drive_id, return_all=True)
                all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
            else:
                all_pos = self.get_slam_odometry(drive_id, return_all=True)
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
                    # Follow https://github.com/yewzijian/3DFeatNet/blob/master/scripts_data_processing/kitti/process_kitti_data.m#L44
                    next_time = next_time[0] + curr_time - 1
                    skip_0, cmpl_0 = self._get_complement_frames(curr_time)
                    skip_1, cmpl_1 = self._get_complement_frames(next_time)
                    skip_2 = (drive_id, curr_time, next_time) in self.discard_pairs
                    if skip_0 or skip_1 or (skip_2 and self.use_old_pose):
                        curr_time += 1
                    else:
                        self.files.append((drive_id, curr_time, next_time, cmpl_0, cmpl_1))
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

    # simple function for getting the xyz point-cloud w.r.t drive and time
    def _get_xyz(self, drive, time):
        fname = self._get_velodyne_fn(drive, time)
        xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
        return xyzr[:, :3]

    # registers source onto target (used by multi-way registration)
    def pairwise_registration(self, source, target, pos_source, pos_target):
        # -----------The following code piece is copied from open3d official documentation
        M = (self.velo2cam @ pos_source.T @ np.linalg.inv(pos_target.T)
             @ np.linalg.inv(self.velo2cam)).T
        icp_fine = open3d.registration.registration_icp(
            source, target, 0.2, M,
            open3d.registration.TransformationEstimationPointToPoint(),
            open3d.registration.ICPConvergenceCriteria(max_iteration=200))
        transformation_icp = icp_fine.transformation
        information_icp = open3d.registration.get_information_matrix_from_point_clouds(
            source, target, self.max_correspondence_distance_fine,
            icp_fine.transformation)
        return transformation_icp, information_icp

    # give the multi-way registration result on one side
    def full_registration(self, pcds, poses):
        # -----------The following code piece is copied from open3d official documentation
        pose_graph = open3d.registration.PoseGraph()
        odometry = np.identity(4)
        pose_graph.nodes.append(open3d.registration.PoseGraphNode(odometry))
        n_pcds = len(pcds)
        for source_id in range(n_pcds):
            for target_id in range(source_id + 1, n_pcds):
                transformation_icp, information_icp = self.pairwise_registration(
                    pcds[source_id], pcds[target_id], poses[source_id], poses[target_id])
                # print("Build open3d.registration.PoseGraph")
                if target_id == source_id + 1:  # odometry case
                    odometry = np.dot(transformation_icp, odometry)
                    pose_graph.nodes.append(
                        open3d.registration.PoseGraphNode(
                            np.linalg.inv(odometry)))
                    pose_graph.edges.append(
                        open3d.registration.PoseGraphEdge(source_id,
                                                                target_id,
                                                                transformation_icp,
                                                                information_icp,
                                                                uncertain=False))
                else:  # loop closure case
                    pose_graph.edges.append(
                        open3d.registration.PoseGraphEdge(source_id,
                                                                target_id,
                                                                transformation_icp,
                                                                information_icp,
                                                                uncertain=True))

        option = open3d.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.max_correspondence_distance_fine,
            edge_prune_threshold=0.25,
            reference_node=0)
        open3d.registration.global_optimization(
            pose_graph,
            open3d.registration.GlobalOptimizationLevenbergMarquardt(),
            open3d.registration.GlobalOptimizationConvergenceCriteria(),
            option)
        
        return [pose_graph.nodes[i].pose for i in range(len(pcds))]

    # a piece of socket program between my implementation and open3d official implemnetation
    def multiway_registration(self, drive, t_curr, t_cmpls, xyz_curr, xyz_cmpls, pos_curr, pos_cmpls):
        # check if any of the matrices are not calculated or loaded into cache 
        recalc_flag = False
        reload_flag = False
        for t_next in t_cmpls:
            key = '%d_%d_%d' % (drive, t_next, t_curr)
            filename = self.icp_path + '/' + key + '.npy'
            if key not in self.kitti_icp_cache:
                if not os.path.exists(filename):
                    recalc_flag = True
                else:
                    reload_flag = True
        
        # if inside cache, then retrieve it
        if not recalc_flag and not reload_flag:
            keys = ['%d_%d_%d' % (drive, t_next, t_curr) for t_next in t_cmpls]
            return [self.kitti_icp_cache[keys[i]] for i in range(len(keys))]
        
        # if already calculated, but not in cache, then load them
        if not recalc_flag and reload_flag:
            filenames = [self.icp_path + '/%d_%d_%d.npy' % (drive, t_next, t_curr) for t_next in t_cmpls]
            listMs = [np.load(filename) for filename in filenames]
            for i, t_next in enumerate(t_cmpls):
                key = '%d_%d_%d' % (drive, t_next, t_curr)
                self.kitti_icp_cache[key] = listMs[i]
            return listMs

        # if some of the matrices are not calculated, then re-calculate all of them.
        _, sel = ME.utils.sparse_quantize(xyz_curr / self.icp_voxel_size, return_index=True)
        pcds_left  = [self.make_open3d_point_cloud(xyz_curr[sel])]
        pcds_right = [self.make_open3d_point_cloud(xyz_curr[sel])]
        poses_left  = [pos_curr] + pos_cmpls[:self.num_complement_one_side]
        poses_right = [pos_curr] + pos_cmpls[self.num_complement_one_side:]
        for i in range(self.num_complement_one_side):
            _, sel_left = ME.utils.sparse_quantize(xyz_cmpls[i] / self.icp_voxel_size, return_index=True)
            pcds_left.append(self.make_open3d_point_cloud(xyz_cmpls[i][sel_left]))
            _, sel_right = ME.utils.sparse_quantize(xyz_cmpls[i + self.num_complement_one_side] / self.icp_voxel_size, return_index=True)
            pcds_right.append(self.make_open3d_point_cloud(xyz_cmpls[i + self.num_complement_one_side][sel_right]))
        
        listM_left = self.full_registration(pcds_left, poses_left)
        listM_right = self.full_registration(pcds_right, poses_right)
        
        listMs = [np.linalg.inv(listM_left[0]) @ listM_left[i] for i in range(1, len(listM_left))] + \
                 [np.linalg.inv(listM_right[0]) @ listM_right[i] for i in range(1, len(listM_right))]
        
        for i, t_next in enumerate(t_cmpls):
            key = '%d_%d_%d' % (drive, t_next, t_curr)
            filename = self.icp_path + '/' + key + '.npy'
            np.save(filename, listMs[i])
            self.kitti_icp_cache[key] = listMs[i]
        return listMs

    def parse_calibration(self, filename):
        calib = {}
        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()
        return calib
    
    def get_slam_odometry(self, drive, indices=None, return_all=False):
        data_path = self.root + '/sequences/%02d' % drive
        calib_filename = data_path + '/calib.txt'
        pose_filename = data_path + '/poses.txt'
        calibration = self.parse_calibration(calib_filename)

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        poses = []
        pose_file = open(pose_filename)
        for line in pose_file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
        
        if pose_filename not in self.kitti_icp_cache:
            self.kitti_icp_cache[pose_filename] = np.array(poses)
        if return_all:
            return self.kitti_icp_cache[pose_filename]
        else:
            return self.kitti_icp_cache[pose_filename][indices]

    def __getitem__(self, idx):
        if self.load_neighbourhood:
            prepare_timer, icp_timer, rot_crop_timer = Timer(), Timer(), Timer()
            prepare_timer.tic()
            drive, t_0, t_1, t_cmpl_0, t_cmpl_1 = self.files[idx]
            # print(self.files[idx])
            if self.use_old_pose:
                all_odometry = self.get_video_odometry(drive, [t_0, t_1] + t_cmpl_0 + t_cmpl_1)
                positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
            else:
                positions = self.get_slam_odometry(drive, [t_0, t_1] + t_cmpl_0 + t_cmpl_1)
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
            xyz_0 = self._get_xyz(drive, t_0)
            xyz_1 = self._get_xyz(drive, t_1)

            # load neighbourhood point clouds
            xyz_cmpl_0 = []
            xyz_cmpl_1 = []
            for (t_tmp_0, t_tmp_1) in zip(t_cmpl_0, t_cmpl_1):
                xyz_cmpl_0.append(self._get_xyz(drive, t_tmp_0))
                xyz_cmpl_1.append(self._get_xyz(drive, t_tmp_1))
            prepare_timer.toc()

            icp_timer.tic()

            if not self.use_old_pose:
                def GetListM(pos_core, pos_cmpls):
                    return [np.linalg.inv(pos_core) @ pos_cmpls[i] for i in range(0, self.num_complement_one_side)] + \
                           [np.linalg.inv(pos_core) @ pos_cmpls[i] for i in range(self.num_complement_one_side, len(pos_cmpls))]
                list_M_0 = GetListM(pos_0, pos_cmpl0)
                list_M_1 = GetListM(pos_1, pos_cmpl1)
            # determine and refine rot&trans matrices of the complement frames by icp
            # adopted from the open3d official implementation of multi-way registration
            else:
                list_M_0 = self.multiway_registration(drive, t_0, t_cmpl_0, xyz_0, xyz_cmpl_0, pos_0, pos_cmpl0)
                list_M_1 = self.multiway_registration(drive, t_1, t_cmpl_1, xyz_1, xyz_cmpl_1, pos_1, pos_cmpl1)

            xyz_cmpl_0 = [self.apply_transform(xyz_k, M_k) 
                        for xyz_k, M_k in zip(xyz_cmpl_0, list_M_0)]
            xyz_cmpl_1 = [self.apply_transform(xyz_k, M_k) 
                        for xyz_k, M_k in zip(xyz_cmpl_1, list_M_1)]

            # use ICP to refine the ground_truth pose, for ICP we don't voxllize the point clouds
            key = '%d_%d_%d' % (drive, t_0, t_1)
            filename = self.icp_path + '/' + key + '.npy'
            if key not in self.kitti_icp_cache:
                if not os.path.exists(filename):
                    print('missing ICP files, recompute it')
                    if self.use_old_pose:
                        # self.logger.write('missing ICP files, recompute it\n')
                        M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
                                    @ np.linalg.inv(self.velo2cam)).T
                        xyz0_t = self.apply_transform(xyz_0, M)
                        pcd0 = to_o3d_pcd(xyz0_t)
                        pcd1 = to_o3d_pcd(xyz_1)
                        reg = open3d.registration.registration_icp(pcd0, pcd1, 0.2, np.eye(4),
                                                                open3d.registration.TransformationEstimationPointToPoint(),
                                                                open3d.registration.ICPConvergenceCriteria(max_iteration=200))
                        # pcd0.transform(reg.transformation)
                        M2 = M @ reg.transformation
                    else:
                        M2 = np.linalg.inv(positions[1]) @ positions[0]
                    np.save(filename, M2)
                else:
                    M2 = np.load(filename)
                self.kitti_icp_cache[key] = M2
            else:
                M2 = self.kitti_icp_cache[key]

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

            # apply downsampling on one side during testing
            if self.split == 'test' and self.downsample_single != 1.0:
                indices = np.random.choice(len(xyz_0), int(len(xyz_0)*self.downsample_single))
                xyz_0 = xyz_0[indices]

            # refined pose is denoted as trans
            tsfm = M2
            rot = tsfm[:3,:3]
            trans = tsfm[:3,3][:,None]

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

            if len(src_nghb) == 0 or len(tgt_nghb) == 0:
                self.logger.write(f"errornous: ({idx}, {t_0}, {t_1})\n")

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
                drive, t_0, t_1, t_cmpl_0, t_cmpl_1 = self.files[idx]
            except:
                drive, t_0, t_1 = self.files[idx]
            # print(self.files[idx])
            if self.use_old_pose:
                all_odometry = self.get_video_odometry(drive, [t_0, t_1])
                positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
            else:
                positions = self.get_slam_odometry(drive, [t_0, t_1])
            pos_0, pos_1 = positions[0:2]

            # load two center point clouds
            xyz_0 = self._get_xyz(drive, t_0)
            xyz_1 = self._get_xyz(drive, t_1)
            prepare_timer.toc()

            icp_timer.tic()
            # use ICP to refine the ground_truth pose, for ICP we don't voxllize the point clouds
            key = '%d_%d_%d' % (drive, t_0, t_1)
            filename = self.icp_path + '/' + key + '.npy'
            if key not in self.kitti_icp_cache:
                if not os.path.exists(filename):
                    if self.use_old_pose:
                        # self.logger.write('missing ICP files, recompute it\n')
                        M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
                                    @ np.linalg.inv(self.velo2cam)).T
                        xyz0_t = self.apply_transform(xyz_0, M)
                        pcd0 = to_o3d_pcd(xyz0_t)
                        pcd1 = to_o3d_pcd(xyz_1)
                        reg = open3d.registration.registration_icp(pcd0, pcd1, 0.2, np.eye(4),
                                                                open3d.registration.TransformationEstimationPointToPoint(),
                                                                open3d.registration.ICPConvergenceCriteria(max_iteration=200))
                        # pcd0.transform(reg.transformation)
                        M2 = M @ reg.transformation
                    else:
                        M2 = np.linalg.inv(positions[1]) @ positions[0]
                    np.save(filename, M2)
                else:
                    M2 = np.load(filename)
                self.kitti_icp_cache[key] = M2
            else:
                M2 = self.kitti_icp_cache[key]
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
            # apply normal voxel downsampling
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

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        if self.IS_ODOMETRY:
            data_path = self.root + '/poses/%02d.txt' % drive
            if data_path not in self.kitti_cache:
                self.kitti_cache[data_path] = np.genfromtxt(data_path)
            if return_all:
                return self.kitti_cache[data_path]
            else:
                return self.kitti_cache[data_path][indices]

    def odometry_to_positions(self, odometry):
        if self.IS_ODOMETRY:
            T_w_cam0 = odometry.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            return T_w_cam0

    def _get_velodyne_fn(self, drive, t):
        if self.IS_ODOMETRY:
            fname = self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        return fname

    def get_position_transform(self, pos0, pos1, invert=False):
        T0 = self.pos_transform(pos0)
        T1 = self.pos_transform(pos1)
        return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
            np.linalg.inv(T1), T0).T)


    def make_open3d_point_cloud(self, xyz, color=None):
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(xyz)
        if color is not None:
            pcd.colors = open3d.utility.Vector3dVector(color)
        return pcd