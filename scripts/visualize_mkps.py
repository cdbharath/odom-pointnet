'''
Author: Bharath Kumar Ramesh Babu
Email: kumar7bharath@gmail.com
'''

import sys
sys.path.append('..')

import cv2
import numpy as np
import os

# from utils.projection_utils import project_birdseye
from utils.dataset_utils import KittiLidarOdomDataset, read_calib_file
from utils.common_utils import rotation_from_quaternion, homo_trans_mat_from_rot_trans, convert_to_homo_coords
from config.config import dataset_config
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    velo_to_cam_trans = read_calib_file(os.path.join(dataset_config.CALIB_PATH, '00', 'calib.txt'))["Tr"]
    velo_to_cam_trans = np.hstack([velo_to_cam_trans, np.array([0, 0, 0, 1])]).reshape((4, 4))

    dataset = KittiLidarOdomDataset(dataset_config.LIDAR_PATH, dataset_config.GT_PATH, pre_process=True)
    mkps, gt, mkp_gt = dataset[0] 

    translation = np.expand_dims(gt[:3], axis=1)
    quat = gt[3:]
    rotation = rotation_from_quaternion(quat)
    trans_mat = homo_trans_mat_from_rot_trans(rotation, translation)

    # prev_mkps = project_birdseye(mkps[:, :3], dilate=True, draw_scale=15)
    # cur_mkps = project_birdseye(mkps[:, 3:], dilate=True, color="red", init_img=prev_mkps, draw_scale=15)
    # cv2.imshow('Before ground truth transform', cur_mkps)
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    ax1.scatter(mkps[:,0], mkps[:,1], mkps[:,2], c='blue')
    ax1.scatter(mkps[:,3], mkps[:,4], mkps[:,5], c='red')
    for i in range(mkps.shape[0]):
        ax1.plot([mkps[i,0], mkps[i,3]], [mkps[i,1], mkps[i,4]], [mkps[i,2], mkps[i,5]], color='gray')

    mkps[:, 3:] = ((trans_mat@convert_to_homo_coords(mkps[:, 3:]).T).T)[:, :3] 

    # prev_mkps = project_birdseye(mkps[:, :3], dilate=True, draw_scale=15)
    # cur_mkps = project_birdseye(mkps[:, 3:], dilate=True, color="red", init_img=prev_mkps, draw_scale=15)
    # cv2.imshow('After ground truth transform', cur_mkps)
    ax2.scatter(mkps[:,0], mkps[:,1], mkps[:,2], c='blue')
    ax2.scatter(mkps[:,3], mkps[:,4], mkps[:,5], c='red')
    for i in range(mkps.shape[0]):
        ax2.plot([mkps[i,0], mkps[i,3]], [mkps[i,1], mkps[i,4]], [mkps[i,2], mkps[i,5]], color='gray')
    
    # cv2.waitKey(0)
    ax1.axes.set_xlim3d(left=-30, right=30) 
    ax1.axes.set_ylim3d(bottom=-30, top=30) 
    ax1.axes.set_zlim3d(bottom=-30, top=30) 
    ax2.axes.set_xlim3d(left=-30, right=30) 
    ax2.axes.set_ylim3d(bottom=-30, top=30) 
    ax2.axes.set_zlim3d(bottom=-30, top=30) 
    plt.show()