'''
Author: Bharath Kumar Ramesh Babu
Email: kumar7bharath@gmail.com
'''

import sys
sys.path.append('..')

import os
import numpy as np
import matplotlib.pyplot as plt

from config.config import dataset_config, log_config
from utils.common_utils import rotation_from_quaternion, quaternion_from_rotation, w_from_xyz, axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle
from utils.dataset_utils import load_kitti_velo_scan, load_kitti_odometry_gt, read_calib_file


def write_inference_to_file(translation, rotation_mat, sequence):
    '''
    Write inference to log file for benchmark evaluation
    '''
        
    with open(os.path.join(log_config.LOG_DIR, str(sequence).zfill(2) + ".txt"), 'a+') as f:
        f.write(f"{rotation_mat[0][0]} {rotation_mat[0][1]} {rotation_mat[0][2]} {translation[0]}" +
        f" {rotation_mat[1][0]} {rotation_mat[1][1]} {rotation_mat[1][2]} {translation[1]}" + 
        f" {rotation_mat[2][0]} {rotation_mat[2][1]} {rotation_mat[2][2]} {translation[2]}\n")        

def get_lidar_pairs(sequence_idx, velo_dir):
    '''
    Get lidar pairs for a given sequence
    '''
    lidar_pairs = []
    sequences = sorted(os.listdir(velo_dir))
    sequence = sequences[sequence_idx]

    scans = sorted(os.listdir(os.path.join(velo_dir, sequence, "velodyne")))
    for i in range(len(scans)):
        if i == 0:
            continue
        lidar_pairs.append([(int(sequence), i - 1), (int(sequence), i)])
    
    return lidar_pairs

if __name__ == "__main__":
    velo_dir = dataset_config.LIDAR_PATH
    sequence_idx = 6

    lidar_pairs = get_lidar_pairs(sequence_idx, velo_dir)
    
    translation = np.array([0, 0, 0])
    rotation_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    write_inference_to_file(translation, rotation_mat, sequence_idx)
    
    w_ = []
    x_ = []
    y_ = []
    z_ = []
    prev_rotation_mat = rotation_mat
    
    for pair in lidar_pairs:
        # print(f"Processing pair: {pair}")
        x, y, z, rot = load_kitti_odometry_gt(dataset_config.GT_PATH, pair[1][0], pair[1][1])
        translation = np.array([x, y, z])
        rotation_mat = rot
        
        axis_angle = rotation_matrix_to_axis_angle(rotation_mat)
        rotation_mat = axis_angle_to_rotation_matrix(axis_angle[0], axis_angle[1])
        
        # quat = quaternion_from_rotation(rotation_mat)
        # rotation_mat = rotation_from_quaternion(quat)
        
        rot_diff = prev_rotation_mat.T @ rotation_mat
        quat_diff = quaternion_from_rotation(rot_diff)
        
        w_.append(quat_diff[0])
        x_.append(quat_diff[1])
        y_.append(quat_diff[2])
        z_.append(quat_diff[3])
        print(quat_diff)
        prev_rotation_mat = rotation_mat
        
        write_inference_to_file(translation, rotation_mat, sequence_idx)
            
    plt.figure(1)
    # Create a histogram
    plt.hist(w_, bins=50)
    
    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('w')
    
    plt.figure(2)
    # Create a histogram
    plt.hist(x_, bins=50)
    
    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('x')

    plt.figure(3)
    # Create a histogram
    plt.hist(y_, bins=50)
    
    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('y')

    plt.figure(4)
    # Create a histogram
    plt.hist(z_, bins=50)
    
    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('z')

    # Show the plot
    plt.show()