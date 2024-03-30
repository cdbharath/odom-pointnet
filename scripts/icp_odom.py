'''
Author: Bharath Kumar Ramesh Babu
Email: kumar7bharath@gmail.com
'''

import sys
sys.path.append('..')

import numpy as np
import cv2

from utils.dataset_utils import KittiLidarOdomDataset
from config.config import dataset_config
from utils.common_utils import rotation_from_quaternion, homo_trans_mat_from_rot_trans, quaternion_from_rotation
from utils.misc_utils import plot_vo_frame
from utils.common_utils import convert_to_homo_coords, homo_trans_mat_inv, axis_angle_to_rotation_matrix

def apply_data_augmentation(points_unproj, homo_trans_diff, augument_yaw_max):
    '''
    Applies data augmentation to the points and ground truth
    '''
    points_unproj = points_unproj.copy()
    homo_trans_diff = homo_trans_diff.copy()
        
    yaw = np.random.uniform(-augument_yaw_max, augument_yaw_max)
    
    aug_mat = np.array([[np.cos(yaw),  0, np.sin(yaw), 0], 
                        [0,            1,           0, 0],
                        [-np.sin(yaw), 0, np.cos(yaw), 0],
                        [0,            0, 0,           1]])
    
    homo_trans_diff = aug_mat @ homo_trans_diff
    points_unproj[:,:3] = ((aug_mat @ convert_to_homo_coords(points_unproj[:,:3]).T).T)[:,:3] 
    
    return points_unproj, homo_trans_diff

def icp(current_pc, previous_pc):
    transformation_matrix = np.identity(4)

    # TODO ideally this should be iterative bringing the current_pc to the previous_pc closer and closer
    # Calculate the transformation matrix using Singular Value Decomposition (SVD)
    mean_current_pc = np.mean(current_pc, axis=0)
    mean_previous_pc = np.mean(previous_pc, axis=0)
    
    centered_current_pc = current_pc - mean_current_pc
    centered_previous_pc = previous_pc - mean_previous_pc
    
    # Compute the covariance matrix
    A = np.dot(centered_previous_pc.T, centered_current_pc)
    U, _, Vt = np.linalg.svd(A)
    R = np.dot(U, Vt)
    t = mean_previous_pc - np.dot(R, mean_current_pc)

    # Update the transformation matrix
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = t

    return transformation_matrix
    

if __name__ == "__main__":
    city = 4
    frame = 0
    
    traj_img_size = 800
    traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
    draw_scale = 0.7
    
    gt_homo_trans_wrt_world = homo_trans_mat_from_rot_trans(np.eye(3), np.expand_dims(np.zeros(3), axis=1))
    homo_trans_wrt_world = homo_trans_mat_from_rot_trans(np.eye(3), np.expand_dims(np.zeros(3), axis=1))

    while True:
        dataset = KittiLidarOdomDataset(dataset_config.LIDAR_PATH, dataset_config.GT_PATH, include_sequences=[city], pre_process=True, top_k=20)
        scans, gt, mkp_gt = dataset[frame]
        scans = scans[mkp_gt.squeeze() == 1]
        
        gt_rot = axis_angle_to_rotation_matrix(gt[3:])
        gt_trans = np.expand_dims(gt[:3], axis=1)
        cur_gt_homo_trans = homo_trans_mat_from_rot_trans(gt_rot, gt_trans)

        # if(frame == 50):
        #     scans, cur_gt_homo_trans = apply_data_augmentation(scans, cur_gt_homo_trans, 1.57)

        cur_icp_homo_trans = icp(scans[:, 3:], scans[:, :3])
        
        gt_homo_trans_wrt_world = gt_homo_trans_wrt_world @ cur_gt_homo_trans        
        homo_trans_wrt_world = homo_trans_wrt_world @ cur_icp_homo_trans
                
        traj_img = plot_vo_frame(gt_homo_trans_wrt_world[:3, 3], homo_trans_wrt_world[:3, 3], traj_img, traj_img_size, draw_scale)
        cv2.imshow('Trajectory', traj_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame = frame + 1