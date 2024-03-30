'''
Author: Bharath Kumar Ramesh Babu
Email: kumar7bharath@gmail.com
'''

import sys
sys.path.append('..')

import open3d as o3d
import cv2
import numpy as np
import os

from utils.dataset_utils import load_kitti_velo_scan, read_calib_file
from utils.common_utils import convert_to_homo_coords
from utils.projection_utils import scan_to_image
from config.config import dataset_config, visualize_config

if __name__ == "__main__":
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for i in range(4540):
        scan = load_kitti_velo_scan(dataset_config.LIDAR_PATH, 0, i)[:, :3]
        
        velo_to_cam_trans = read_calib_file(os.path.join(dataset_config.CALIB_PATH, '00', 'calib.txt'))["Tr"]
        velo_to_cam_trans = np.hstack([velo_to_cam_trans, np.array([0, 0, 0, 1])]).reshape((4, 4))
        scan_transformed  = ((velo_to_cam_trans@convert_to_homo_coords(scan).T).T)[:,:3]

        spherical_proj, _, _ = scan_to_image(scan_transformed[:, :3], visualize_config.ROW_SCALE, visualize_config.COL_SCALE, True)
        cv2.imshow("Spherical Projection", spherical_proj)
        cv2.waitKey(1)

        source = o3d.geometry.PointCloud()
        v3d = o3d.utility.Vector3dVector
        source.points = v3d(scan[:, :3])

        vis.clear_geometries()
        vis.add_geometry(source)

        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()
