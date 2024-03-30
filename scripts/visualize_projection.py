'''
Author: Bharath Kumar Ramesh Babu
Email: kumar7bharath@gmail.com
'''

import sys
sys.path.append('..')

import numpy as np
import cv2
from scipy import ndimage
import time
import os

from utils.dataset_utils import load_kitti_velo_scan, read_calib_file
from utils.common_utils import convert_to_homo_coords, homo_trans_mat_inv
from utils.projection_utils import scan_to_image, get_sift_features, kp_to_points, match_sift_features, project_birdseye
from config.config import dataset_config, visualize_config


def visualize_projection(sequence = 0, frame = 100, waitkey = 0):
    # Visualizes the following 
    # 1. Projection of a pair of consecutive lidar scans to image space
    # 2. Get SIFT features from the resulting images
    # 3. Match the SIFT features 
    # 4. Reproject the matched feature points back to 3D space 

    start_time = time.time()

    velo_to_cam_trans = read_calib_file(os.path.join(dataset_config.CALIB_PATH, '00', 'calib.txt'))["Tr"]
    velo_to_cam_trans = np.hstack([velo_to_cam_trans, np.array([0, 0, 0, 1])]).reshape((4, 4))

    # Load and visualize scan
    scan = load_kitti_velo_scan(dataset_config.LIDAR_PATH, sequence, frame)[:, :3]
    scan = ((velo_to_cam_trans@convert_to_homo_coords(scan).T).T)[:,:3]     # Convert points from lidar to camera frame
    print(f"number of lidar points: {scan.shape[0]}")
    
    inital_pointcloud = project_birdseye(scan)

    # Lidar to Depth projection 
    image_proj, pyr, unnorm_image_proj = scan_to_image(scan, visualize_config.ROW_SCALE, visualize_config.COL_SCALE, inpaint=True)

    # Get SIFT features from depth projection
    kps, des, sift_vis = get_sift_features(image_proj)
    kps_array = np.array([[kp.pt[0], kp.pt[1]] for kp in kps])
    
    # Reproject back to 3D space 
    points_reproj = kp_to_points(unnorm_image_proj, kps_array, pyr, visualize_config.ROW_SCALE, visualize_config.COL_SCALE)
    print(f"number of reprojected points: {points_reproj.shape[0]}")

    reproj_pointcloud = project_birdseye(points_reproj, dilate=True, color="red", init_img=inital_pointcloud)

    # # keypoint matches 
    scan_next = load_kitti_velo_scan(dataset_config.LIDAR_PATH, sequence, frame + 1)[:, :3]    
    scan_next = ((velo_to_cam_trans@convert_to_homo_coords(scan_next).T).T)[:,:3]     # Convert points from lidar to camera frame

    image_proj_next, pyr_next, unnorm_image_proj_next = scan_to_image(scan_next, visualize_config.ROW_SCALE, visualize_config.COL_SCALE, inpaint=True)

    # # Images rotated for better matching visualization 
    image_proj_rot = ndimage.rotate(image_proj, 90)      
    image_proj_rot_next = ndimage.rotate(image_proj_next, 90)
    kps, des, sift_vis = get_sift_features(image_proj_rot)
    kps_next, des_next, _ = get_sift_features(image_proj_rot_next)

    matches, matches_image = match_sift_features(image_proj_rot, kps, des, image_proj_rot_next, kps_next, des_next, filter="feat", thresh=0.8) 
    print("Number of SIFT matches: " + str(len(matches)))

    end_time = time.time()
    print(f"Time taken to run this script: {end_time - start_time} secs")

    cv2.imshow("Initial Pointcloud", inital_pointcloud)
    cv2.imshow("Spherical Projection", image_proj)
    cv2.imshow("Sift Matches", ndimage.rotate(matches_image, -90))
    cv2.imshow("Reprojected Pointcloud", reproj_pointcloud)

    cv2.waitKey(waitkey)


if __name__ == "__main__":
    mode = "frame"
    city = 1
    frame = 100

    if mode == "frame":
        visualize_projection(city, frame)
    elif mode == "video":
        while True:
            visualize_projection(city, frame, 1)
            frame = frame + 1

    cv2.destroyAllWindows()

