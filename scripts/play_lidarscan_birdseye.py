'''
Author: Bharath Kumar Ramesh Babu
Email: kumar7bharath@gmail.com
'''

import sys
sys.path.append('..')

import cv2
import numpy as np
import os

from utils.dataset_utils import load_kitti_velo_scan, read_calib_file
from utils.projection_utils import scan_to_image, project_birdseye
from utils.common_utils import convert_to_homo_coords
from config.config import dataset_config, visualize_config

if __name__ == "__main__":
    for i in range(4540):
        scan = load_kitti_velo_scan(dataset_config.LIDAR_PATH, 0, i)[:, :3]

        velo_to_cam_trans = read_calib_file(os.path.join(dataset_config.CALIB_PATH, '00', 'calib.txt'))["Tr"]
        velo_to_cam_trans = np.hstack([velo_to_cam_trans, np.array([0, 0, 0, 1])]).reshape((4, 4))

        scan = ((velo_to_cam_trans@convert_to_homo_coords(scan).T).T)[:,:3]     # Convert points from lidar to camera frame

        spherical_proj, _, _ = scan_to_image(scan[:, :3], visualize_config.ROW_SCALE, visualize_config.COL_SCALE, True)
        birdseye = project_birdseye(scan[:, :3])

        cv2.imshow("Spherical Projection", spherical_proj)
        cv2.imshow("Birds Eye Projection", birdseye)

        cv2.waitKey(1)

