'''
Author: Bharath Kumar Ramesh Babu
Email: kumar7bharath@gmail.com
'''

import sys
sys.path.append('..')

import os
import cv2
import numpy as np

from torch.utils.data import Dataset
import torch

from config.config import dataset_config, visualize_config
from utils.common_utils import quaternion_from_rotation, sample_and_concatenate_points, homo_trans_mat_from_rot_trans
from utils.common_utils import convert_to_homo_coords, homo_trans_mat_inv, rotation_from_quaternion, normalize_rotation_matrix
from utils.common_utils import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle
from utils.projection_utils import project_birdseye, scan_to_image, project_unproject


def load_kitti_velo_scan(velo_dir, sequence = 0, index = 0):
    '''
    Loads lidar points from .bin files in semantic kitti dataset
    '''
    _scan = np.fromfile(os.path.join(velo_dir, str(sequence).zfill(2), "velodyne", str(index).zfill(6) + ".bin"), dtype=np.float32)
    _scan = _scan.reshape((-1, 4))
    return _scan


def load_kitti_odometry_gt(gt_dir, sequence = 0, index = 0):
    '''
    Loads odometry ground truth from the Kitti dataset
    '''
    
    def get_trans_from_str(ss):
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])

        rot = np.array([[float(ss[0]), float(ss[1]), float(ss[2])],
                    [float(ss[4]), float(ss[5]), float(ss[6])],
                    [float(ss[8]), float(ss[9]), float(ss[10])]])
        rot = normalize_rotation_matrix(rot)
        
        return x, y, z, rot

    
    gt_path = os.path.join(gt_dir, str(sequence).zfill(2) + ".txt")
    with open(gt_path) as f:
        data = f.readlines()

    ss = data[index].strip().split()
    x, y, z, rot = get_trans_from_str(ss)

    return x, y, z, rot


def read_calib_file(filepath):
    '''
    Loads data from calib file
    '''
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    obj = {}
    for line in lines:
        name, value = line.strip().split(':', 1)
        obj[name] = np.array([float(x) for x in value.split()])
    
    return obj
    
    
def shortlist_points_with_mkp_ranking(scans, mkp_ranking):
    '''
    Shortlist points based on the MKP ranking
    
    :param scans: (B, 6, N) tensor, MKP matches  
    :param ranking: (B, 1, N) tensor, MKP match ranking
    
    :return shortlist: (B, 6, M) tensor, shortlisted MKPs based on ranking
    '''
    batch_size = scans.shape[0]
    shorlist_list = []
    max_points = 0
    for i in range(batch_size):
        cur_scan = scans[i, :, :]
        cur_mkp_ranking = mkp_ranking[i, :, :]
        cur_shortlist = cur_scan[:, cur_mkp_ranking.squeeze()==1]
        
        if cur_shortlist.shape[1] < 3:
            rand_indices = torch.randint(0, cur_scan.shape[1], (3,))
            cur_shortlist = cur_scan[:, rand_indices]
        
        max_points = max(max_points, cur_shortlist.shape[1])
        shorlist_list.append(cur_shortlist)

    for i in range(batch_size):
        cur_scan = shorlist_list[i]
        
        if cur_scan.shape[1] < max_points/2:
            rand_indices = torch.randint(0, cur_scan.shape[1], (max_points-cur_scan.shape[1],))
        else:
            rand_indices = torch.randperm(cur_scan.shape[1])[:max_points-cur_scan.shape[1]]
        points_to_add = cur_scan[:, rand_indices]
        
        if points_to_add.shape[1] != 0:
            shorlist_list[i] = torch.cat([cur_scan, points_to_add], dim=1)
    
    return torch.stack(shorlist_list, dim=0)


class KittiLidarOdomDataset(Dataset):
    '''
    Download and Extract the following data from the Kitti website (https://www.cvlibs.net/datasets/kitti/eval_odometry.php) 
    and place it in kitti_lidar_data directory
    1. Lidar data
    2. Odometry ground truth
    Maximum number of points in a point cloud: 129392 [sequence 0 - sequence 11]
    
    :param include_sequences: List of sequences to be included from the dataset
    :param max_points: Maximum number of points in a point cloud across the dataset
    :param pre_process: If true, proprocess keypoints detection
    :param pts_per_batch: Equalizing number of keypoints to accomodate batch training
    :param augment: If true, augment the data
    :param augment_yaw_max: Maximum yaw angle for rotation augmentation
    :param noise_std: Standard deviation for noise augmentation
    :param top_k: Number of keypoints to be selected for MKP selection module
    :param draw: If true, gets data for visualization
    '''
    def __init__(self, velo_dir, gt_dir, include_sequences=[0], max_points=129392, pre_process=False, \
        pts_per_batch=400, augument=False, augment_yaw_max=0.05, noise_std=0.0, top_k=100, draw=False):
        
        self.velo_dir = velo_dir                        
        self.gt_dir = gt_dir
        self.max_points = max_points                   
        self.pre_process = pre_process
        self.pts_per_batch = pts_per_batch
        self.noise_std = noise_std
        self.augument = augument
        self.augument_yaw_max = augment_yaw_max
        self.top_k = top_k
        self.draw = draw

        self.lidar_pairs = []

        # Get paths for lidar pairs and ground truth from all the sequences
        sequences = sorted(os.listdir(self.velo_dir))
        for sequence_idx in include_sequences:
            sequence = sequences[sequence_idx]

            scans = sorted(os.listdir(os.path.join(self.velo_dir, sequence, "velodyne")))
            for i in range(len(scans)):
                if i == 0:
                    continue
                self.lidar_pairs.append([(int(sequence), i - 1), (int(sequence), i)])


    def __len__(self):
        return len(self.lidar_pairs)


    def __getitem__(self, index):
        '''
        Projects point cloud into shperical coordinates, matches keypoints and reprojects them back to original coordinates
        
        :param index: Index of the lidar pair and ground truth to be loaded
        :return points_unproj: (N, 6) array, reprojected 3D points
        :return gt: (N, 6) array, ground truth containing translation and rotation
        :return mkp_gt: ground truth for MKP selection module
        
        The following are optional return values and are only returned if draw is set to True
        :return matches image: Image containing matches
        :return depth_image_1: (H, W) array, unnormalized depth projection of image 1
        :return depth_image_2: (H, W) array, unnormalized depth projection of image 2
        :return keypoints_image_1: (H, W) array, keypoints in image 1
        :return keypoints_image_2: (H, W) array, keypoints in image 2
        :return initial_scans: (N, 6) array, initial scans
        '''
    
        # Loads lidar pair and ground truth
        prev_scan, cur_scan, homo_trans_diff, velo_to_cam_trans = self.get_lidar_pair(index)
            
        # Convert points from lidar to camera frame
        prev_scan = ((velo_to_cam_trans@convert_to_homo_coords(prev_scan).T).T)[:,:3]
        cur_scan  = ((velo_to_cam_trans@convert_to_homo_coords(cur_scan).T).T)[:,:3]
        
        if not self.pre_process:
            prev_scan = sample_and_concatenate_points(prev_scan, self.max_points)
            cur_scan = sample_and_concatenate_points(cur_scan, self.max_points)
            
            trans_diff = np.squeeze(homo_trans_diff[0:3,3])
            rot_diff = homo_trans_diff[:3,:3]

            axis_angle_diff = rotation_matrix_to_axis_angle(rot_diff)
            gt = np.hstack((trans_diff, axis_angle_diff))
            return np.hstack((prev_scan, cur_scan)), gt
        
        else:            
            if not self.draw:
                points_unproj = project_unproject(
                            prev_scan, cur_scan, visualize_config.ROW_SCALE, visualize_config.COL_SCALE)
            else:
                points_unproj, matches_image, image_proj, image_proj_next, kps_image, kps_next_image = project_unproject(
                            prev_scan, cur_scan, visualize_config.ROW_SCALE, visualize_config.COL_SCALE, draw=True)
                        
            # Duplicates points to make sure every pointcloud in the batch has the same number of points
            # TODO Beware of this weird stuff. This could lead to a lot of bugs
            if(points_unproj.shape[0] < self.pts_per_batch):
                points_unproj = sample_and_concatenate_points(points_unproj, self.pts_per_batch)
            else:
                points_unproj = points_unproj[:self.pts_per_batch, :]

            # Adds gaussian noise to the points
            points_unproj += np.random.normal(0, self.noise_std, points_unproj.shape)            
            
            if self.augument:
                points_unproj_aug, homo_trans_diff_aug = self.apply_data_augmentation(points_unproj, homo_trans_diff, self.augument_yaw_max)   
                trans_diff_aug = np.squeeze(homo_trans_diff_aug[:3,3])
                rot_diff_aug = homo_trans_diff_aug[:3,:3]
                
                axis_angle_diff_aug = rotation_matrix_to_axis_angle(rot_diff_aug)
                gt_aug = np.hstack((trans_diff_aug, axis_angle_diff_aug))
            
            # Generate MKP ground truth based on ground truth
            mkp_gt = self.get_mkp_gt(points_unproj, homo_trans_diff, self.top_k)
                        
            trans_diff = np.squeeze(homo_trans_diff[0:3,3])
            rot_diff = homo_trans_diff[:3,:3]
            
            axis_angle_diff = rotation_matrix_to_axis_angle(rot_diff)
            gt = np.hstack((trans_diff, axis_angle_diff))
            
            if self.draw:
                prev_scan = sample_and_concatenate_points(prev_scan, self.max_points)
                cur_scan = sample_and_concatenate_points(cur_scan, self.max_points)

                return points_unproj, gt, mkp_gt, matches_image, image_proj, image_proj_next, kps_image, kps_next_image, \
                        np.hstack((prev_scan[:,:3], cur_scan[:,:3]))
            
            if self.augument:
                return points_unproj, gt, mkp_gt, points_unproj_aug, gt_aug 
            
            return points_unproj, gt, mkp_gt


    def get_lidar_pair(self, index):
        '''
        Loads data and ground truth for a given index
        '''
        # Load lidar scans and calibration data
        prev_scan = load_kitti_velo_scan(self.velo_dir, self.lidar_pairs[index][0][0], self.lidar_pairs[index][0][1])[:,:3]
        cur_scan = load_kitti_velo_scan(self.velo_dir, self.lidar_pairs[index][1][0], self.lidar_pairs[index][1][1])[:,:3]
        velo_to_cam_trans = read_calib_file(os.path.join(dataset_config.CALIB_PATH, str(self.lidar_pairs[index][0][0]).zfill(2), 'calib.txt'))["Tr"]
        velo_to_cam_trans = np.hstack([velo_to_cam_trans, np.array([0, 0, 0, 1])]).reshape((4, 4))

        # Load ground truth
        x_prev, y_prev, z_prev, rot_prev = load_kitti_odometry_gt(self.gt_dir, self.lidar_pairs[index][0][0], self.lidar_pairs[index][0][1])
        x, y, z, rot = load_kitti_odometry_gt(self.gt_dir, self.lidar_pairs[index][1][0], self.lidar_pairs[index][1][1])

        # Get translation and rotation relative to the previous frame
        prev_trans = np.expand_dims(np.array([x_prev, y_prev, z_prev]), axis=1)
        prev_homo_trans = homo_trans_mat_from_rot_trans(rot_prev, prev_trans)        
        
        trans = np.expand_dims(np.array([x, y, z]), axis=1)
        homo_trans = homo_trans_mat_from_rot_trans(rot, trans)
    
        homo_trans_diff = homo_trans_mat_inv(prev_homo_trans)@homo_trans
    
        return prev_scan, cur_scan, homo_trans_diff, velo_to_cam_trans


    def apply_data_augmentation(self, points_unproj, homo_trans_diff, augument_yaw_max):
        '''
        Applies data augmentation to the points and ground truth
        '''
        points_unproj = points_unproj.copy()
        homo_trans_diff = homo_trans_diff.copy()
        
        # Use this if you are using sample_quat_y algo
        # reference_quat = quaternion_from_rotation(homo_trans_diff[:3, :3])
        # reference_quat[2] = sample_quat_y(reference_quat[2])
        # transformed_homo_trans_diff = rotation_from_quaternion(reference_quat)
        # aug_rot = transformed_homo_trans_diff @ np.linalg.inv(homo_trans_diff[:3, :3])
        # aug_mat = homo_trans_mat_from_rot_trans(aug_rot, np.array([[0, 0, 0]]).T)
        
        yaw = np.random.uniform(-augument_yaw_max, augument_yaw_max)
        # yaw = np.random.normal(0, augument_yaw_max/3)
        # yaw = max(-0.06, min(0.06, yaw))
        
        aug_mat = np.array([[np.cos(yaw),  0, np.sin(yaw), 0], 
                            [0,            1,           0, 0],
                            [-np.sin(yaw), 0, np.cos(yaw), 0],
                            [0,            0, 0,           1]])
        
        homo_trans_diff = aug_mat @ homo_trans_diff
        points_unproj[:,:3] = ((aug_mat @ convert_to_homo_coords(points_unproj[:,:3]).T).T)[:,:3] 
        
        return points_unproj, homo_trans_diff


    def get_mkp_gt(self, points_unproj, homo_trans_diff, top_k):
        '''
        Generates MKP ground truth based odometry ground truth 
        '''
        distances = np.linalg.norm(homo_trans_diff @ convert_to_homo_coords(points_unproj[:,3:6]).T - convert_to_homo_coords(points_unproj[:,0:3]).T, axis=0)        
        mkp_gt = np.zeros((points_unproj.shape[0], 1))
        sorted_distances_indices = np.argpartition(distances, top_k)[:top_k]
        mkp_gt[sorted_distances_indices] = 1

        return mkp_gt
        

if __name__ == "__main__":
    # Visualize the Kitti lidar dataset and its ground truth trajectory
    dataset = KittiLidarOdomDataset(dataset_config.LIDAR_PATH, dataset_config.GT_PATH, include_sequences=[0])
    velo_to_cam_trans = read_calib_file(os.path.join(dataset_config.CALIB_PATH, '00', 'calib.txt'))["Tr"]
    velo_to_cam_trans = np.hstack([velo_to_cam_trans, np.array([0, 0, 0, 1])]).reshape((4, 4))

    traj_img_size = 800
    traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
    draw_scale = 0.7
    
    gt_trans_wrt_world = np.zeros(3)
    gt_rot_wrt_world = np.eye(3)
    gt_homo_trans_wrt_world = homo_trans_mat_from_rot_trans(gt_rot_wrt_world, np.expand_dims(gt_trans_wrt_world, axis=1))
    
    for i in range(int(len(dataset))):
        scans, gt = dataset[i]
        gt_rot = rotation_from_quaternion(gt[3:])
        gt_trans = np.expand_dims(gt[:3], axis=1)
        gt_homo_trans = homo_trans_mat_from_rot_trans(gt_rot, gt_trans)
        
        gt_homo_trans_wrt_world = gt_homo_trans_wrt_world@gt_homo_trans 
        scan_to_visualize = scans[:, 3:6]

        # groundtruth in red
        x_gt = gt_homo_trans_wrt_world[0, 3]
        y_gt = gt_homo_trans_wrt_world[1, 3]
        z_gt = gt_homo_trans_wrt_world[2, 3]
        
        # z axis towards top and x axis to the right
        draw_x_gt, draw_y_gt = int(traj_img_size/2) + int(draw_scale*x_gt), int(traj_img_size/2) - int(draw_scale*z_gt)
        cv2.circle(traj_img, (draw_x_gt, draw_y_gt), 1,(0, 0, 255), 1)  

        # Print the coordinates in the top left corner
        cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
        text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x_gt, y_gt, z_gt)
        cv2.putText(traj_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

        birdseye = project_birdseye(scan_to_visualize)
        cur_spherical_proj, _, _ = scan_to_image(scan_to_visualize, inpaint=True)

        scan_to_visualize = scan_to_visualize[scan_to_visualize[:, 1] < 1]
        scan_wrt_world = ((gt_homo_trans_wrt_world@convert_to_homo_coords(scan_to_visualize).T).T)[:,:3] 
        traj_with_points = project_birdseye(scan_wrt_world, init_img=traj_img, draw_scale=draw_scale)

        cv2.imshow('Birdseye Projection', birdseye)
        cv2.imshow('Trajectory', traj_img)
        cv2.imshow('Spherical Projection Currrent Frame', cur_spherical_proj)
        cv2.imshow('Trajectory with Points', traj_with_points)
        cv2.waitKey(1)
