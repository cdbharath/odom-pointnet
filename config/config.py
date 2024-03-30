'''
Author: Bharath Kumar Ramesh Babu
Email: kumar7bharath@gmail.com
'''

import os
import sys
sys.path.append('..')

cwd = os.path.dirname(os.path.abspath(__file__))
cwd = os.path.join(cwd, "..")


class train_inference_config:
    BATCH_SIZE = 128
    NUM_WORKERS = 8
    LEARNING_RATE_R = 0.001
    LEARNING_RATE_T = 0.001
    LEARNING_RATE_MKP = 0.001
    LEARNING_RATE_DECAY_R = 0.5
    LEARNING_RATE_DECAY_T = 0.5
    LEARNING_RATE_DECAY_MKP = 0.5
    MILESTONE_R = [5, 10]  
    MILESTONE_T = [5, 10]
    MILESTONE_MKP = [5, 10]
    MOMENTUM = 0.0
    N_EPOCHS = 100
    VAL_EVERY = 10
    TOP_N = 100
    WEIGHT_DECAY_MKP = 0.0000
    WEIGHT_DECAY_T = 0.0000
    WEIGHT_DECAY_R = 0.0000
    AUGUMENT = False
    TRAIN_CITIES = [0, 1, 2, 3, 4, 5, 8, 9]
    VAL_CITIES = [6, 7]
    AUG_ANGLE = 0.05
    DROPOUT = 0.0
    MODEL = "PointNet"


class dataset_config:
    CWD = cwd
    LIDAR_PATH = os.path.join(cwd, "kitti_lidar_data/data_odometry_velodyne/dataset/sequences")
    IMAGE_PATH = os.path.join(cwd, "kitti_lidar_data/data_odometry_gray/dataset/sequences/")
    GT_PATH = os.path.join(cwd, "kitti_lidar_data/data_odometry_poses/dataset/poses")
    CALIB_PATH = os.path.join(cwd, "kitti_lidar_data/data_odometry_calib/dataset/sequences")
    TRAIN_SEQUENCES = 11


class log_config:
    CKPT_PATH = os.path.join(cwd, "ckpt")
    CKPT_EVERY = 1
    LOG_DIR = os.path.join(cwd, "logs")


class visualize_config:
    FILENAME = os.path.join(cwd, "kitti_lidar_data/data_odometry_velodyne/dataset/sequences/00/velodyne/000000.bin")
    LIDAR_DATA_PATH = os.path.join(cwd, "kitti_lidar_data/data_odometry_velodyne/dataset/sequences/")
    LABEL_DATA_PATH = os.path.join(cwd, "kitti_lidar_data/data_odometry_labels/dataset/sequences/")
    ROW_SCALE = 64
    COL_SCALE = 1024
    TRAJ_IMAGE_SIZE = 800
    DRAW_SCALE = 0.7
