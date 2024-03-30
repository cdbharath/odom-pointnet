'''
Author: Bharath Kumar Ramesh Babu
Email: kumar7bharath@gmail.com
'''

import sys
sys.path.append('..')

import os
import cv2
import numpy as np

from config.config import dataset_config
from utils.common_utils import homo_trans_mat_from_rot_trans, convert_to_homo_coords, rotation_from_quaternion
from utils.projection_utils import project_birdseye, scan_to_image
from utils.dataset_utils import KittiLidarOdomDataset, read_calib_file


def apply_data_augmentation(points_unproj, homo_trans_diff, augument_yaw_max):
    '''
    Applies data augmentation to the points and ground truth
    '''
    points_unproj = points_unproj.copy()
    homo_trans_diff = homo_trans_diff.copy()
        
    yaw = augument_yaw_max
    
    aug_mat = np.array([[np.cos(yaw),  0, np.sin(yaw), 0], 
                        [0,            1,           0, 0],
                        [-np.sin(yaw), 0, np.cos(yaw), 0],
                        [0,            0, 0,           1]])
    
    homo_trans_diff = aug_mat @ homo_trans_diff
    points_unproj[:,:3] = ((aug_mat @ convert_to_homo_coords(points_unproj[:,:3]).T).T)[:,:3] 
    
    return points_unproj, homo_trans_diff


if __name__ == "__main__":
    index_to_augment = 50
    angle = 0.7

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
        
        if i == index_to_augment:
            scans, gt_homo_trans = apply_data_augmentation(scans, gt_homo_trans, angle)
        
        gt_homo_trans_wrt_world = gt_homo_trans_wrt_world@gt_homo_trans 
        scan_to_visualize = scans[:, :3]

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
        birdseye = project_birdseye(scans[:, 3:], color="red", init_img=birdseye)

        # scan_to_visualize = scan_to_visualize[scan_to_visualize[:, 1] < 1]
        # scan_wrt_world = ((gt_homo_trans_wrt_world@convert_to_homo_coords(scan_to_visualize).T).T)[:,:3] 
        # traj_with_points = project_birdseye(scan_wrt_world, init_img=traj_img, draw_scale=draw_scale)

        cv2.imshow('Birdseye Projection', birdseye)
        cv2.imshow('Trajectory', traj_img)
        # cv2.imshow('Trajectory with Points', traj_with_points)
        cv2.waitKey(1)

        if i == index_to_augment or i == index_to_augment - 1 or i == index_to_augment + 1 :
            count = 0
            while(count < 100000000):
                count = count + 1
