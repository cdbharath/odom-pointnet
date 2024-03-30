'''
Author: Bharath Kumar Ramesh Babu
Email: kumar7bharath@gmail.com
'''

import sys
sys.path.append('..')

import numpy as np
import cv2

from utils.common_utils import wrap_angle
from utils.depth_completion_utils import nearest_kdtree, cv_inpaint
from scipy.spatial import KDTree as KDTree


def proj_3d_to_2d(pts_3d, row_scale = 63, col_scale = 1023, enable_roi=True, roi_radius=100.0):
    '''
    Project 3D point cloud to 2D image space
    
    :param pts_3d: Nx3 array of 3D points
    :return pts_2d: Nx2 array of 2D points
    
    :return pyr: Nx3 array of pitch, yaw, and range 
    '''
    x, y, z = pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2]

    R = np.sqrt(x*x + y*y + z*z)
    pitch = -np.arcsin(y/R)
    yaw = np.arctan2(z, x)
    
    if enable_roi:
        pitch = pitch[R < roi_radius]
        yaw = yaw[R < roi_radius]
        R = R[R < roi_radius]
            
    # Normalize and scale the indices
    _pitch = (row_scale - 1) * ((np.max(pitch) - pitch)/(abs(np.max(pitch)) + abs(np.min(pitch))))
    _yaw = (col_scale - 1) * (0.5 * (-(yaw/np.pi) + 1))

    # Round the indices and depth to integers
    pts_2d_x = np.rint(_pitch).astype("int") 
    pts_2d_y = np.rint(_yaw).astype("int")

    return np.hstack((pts_2d_x[:, np.newaxis], pts_2d_y[:, np.newaxis])), [pitch, yaw, R, pts_2d_x, pts_2d_y, pts_3d, pts_3d]


def scan_to_image(scan, row_scale=63, col_scale=1023, inpaint=False):
    '''
    Project 3D point cloud to 2D image space
    Note: The input point clouds are in the lidar frame
    Refer: https://www.cvlibs.net/datasets/kitti/setup.php
    
    :param scan: Nx3 array of 3D points
    :param inpaint: Boolean flag to apply depth completion to the sparse spherical image
    
    :return norm_image: Normalized 2D image of the spherical projection
    :return pyr: Nx3 array of pitch, yaw, and range
    :return _image: 2D image of the spherical projection
    '''
    
    scan_2d, pyr = proj_3d_to_2d(scan, row_scale, col_scale)
    
    _pitch = scan_2d[:, 0] 
    _yaw = scan_2d[:, 1]
    _R = pyr[2]

    init_val = 0.0
    _image = np.full([row_scale, col_scale], init_val)
    _image[_pitch, _yaw] = _R

    if inpaint:
        # Fills the pixels with missing data to complete the sparse spherical image

        _image = nearest_kdtree(_image)

        # Convert to uint8 images, aka normalization
        depth_complete = _image*255/np.max(_image)
        depth_complete = np.round(depth_complete).astype("uint8")

        # depth_complete = cv_inpaint(depth_complete)
        # depth_complete = cv_dilate(depth_complete)

    # norm_image = depth_complete
    norm_image = cv2.equalizeHist(depth_complete)
    return norm_image, pyr, _image


def get_sift_features(image, n_feat=2000, draw_kp=True):
    '''
    Get sift keypoints from the input image
    
    :param image: Input image
    
    :return kps: List of keypoints
    :return des: List of descriptors
    '''
    # sift_features = cv2.SIFT_create(nfeatures=n_feat, nOctaveLayers=3, contrastThreshold=0.001, edgeThreshold=15, sigma=1)
    # kps, des = sift_features.detectAndCompute(image, None)
    
    orb = cv2.ORB_create(nfeatures=n_feat, scoreType=cv2.ORB_FAST_SCORE, 
                         scaleFactor=1.2, nlevels=8, edgeThreshold=0, fastThreshold=0)
    kps = orb.detect(image, None)
    kps, des = orb.compute(image, kps)
    
    if draw_kp:
        kp_image = cv2.drawKeypoints(image, kps, image)
        return kps, des, kp_image

    return kps, des


def match_sift_features(img1, kps1, des1, img2, kps2, des2, filter="default", thresh=0.9, optional_thresh=60, draw_matches=True):
    '''
    Matches SIFT key points from one image to another
    
    :param img1: First image
    :param kp1: List of keypoints from the first image
    :param des1: List of descriptors from the first image
    :param img2: Second image
    :param kps2: List of keypoints from the second image
    :param des2: List of descriptors from the second image
    :param filter: Filter to apply to the matches
    
    :return good: List of good matches
    :return matches_image: Image with the matches drawn
    '''
    
    # BF matcher
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1,des2,k=2)
    
    # FLANN matcher
    flann = cv2.FlannBasedMatcher_create()
    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)
    matches = flann.knnMatch(des1, des2, k=2)

    draw_good = []
    good = []
    for m,n in matches:
        if filter == "feat":
            if m.distance < thresh*n.distance:
                # TODO this is a very hacky way to do this
                if ((kps1[m.queryIdx].pt[0] - kps2[m.trainIdx].pt[0])**2 + (kps1[m.queryIdx].pt[1] - kps2[m.trainIdx].pt[1])**2)**0.5 < optional_thresh:
                    draw_good.append([m])
                    good.append(list(kps1[m.queryIdx].pt) + list(kps2[m.trainIdx].pt))
        elif filter == "dist":
            if ((kps1[m.queryIdx].pt[0] - kps2[m.trainIdx].pt[0])**2 + (kps1[m.queryIdx].pt[1] - kps2[m.trainIdx].pt[1])**2)**0.5 < thresh:
                draw_good.append([m])
                good.append(list(kps1[m.queryIdx].pt) + list(kps2[m.trainIdx].pt)) 
        else:
            draw_good.append([m])
            good.append(list(kps1[m.queryIdx].pt) + list(kps2[m.trainIdx].pt))

    good = sorted(good, key=lambda x: draw_good[good.index(x)][0].distance)

    if draw_matches:
        # draw_good = sorted(draw_good, key=lambda x: x[0].distance)
        matches_image = cv2.drawMatchesKnn(img1, kps1, img2, kps2, draw_good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return np.array(good), matches_image
    else:
        return np.array(good)


def kp_to_points(image, kp, pyr, row_scale=64, col_scale=1024):
    '''
    Project keypoints back to pointcloud
    
    :param image: Input image
    :param kp: Keypoints
    :param pyr: Nx3 array of pitch, yaw, and range
    :param only_true_points: Boolean flag to return only the keypoints that are present in the original pointcloud
    
    :return x, y, z: 3D points
    '''
    
    # Original pitch
    pitch = pyr[0]
    pts_3d = pyr[6]
    
    # Keypoint pitch and yaw
    _pitch = kp[:,1]
    _yaw = kp[:,0]

    # Calculate range of 3D point
    _R = image[np.rint(_pitch).astype("int"), np.rint(_yaw).astype("int")]

    # Convert 2D keypoint to pitch and yaw angles
    _pitch = np.max(pitch) - (_pitch/(row_scale-1)) * (abs(np.max(pitch)) + abs(np.min(pitch)))
    _yaw = -((2*_yaw/(col_scale-1) - 1) * np.pi)

    # Convert pitch, yaw, and range to x, y, and z coordinates
    x = _R * np.cos(_pitch) * np.cos(_yaw)
    y = -_R * np.sin(_pitch)
    z = _R * np.cos(_pitch) * np.sin(_yaw)
    
    reprojected = np.vstack((x, y, z)).T
    
    kdtree = KDTree(pts_3d)
    _, pre_indices = kdtree.query(reprojected, k=1)
    reprojected = pts_3d[pre_indices]
    
    return reprojected


def project_birdseye(scan, dilate=False, color="default", init_img=None, draw_scale=4.5):
    '''
    Projects pointcloud to birdseye view image
    Note: The input point clouds are in the camera frame as opposed to the lidar frame
    Refer: https://www.cvlibs.net/datasets/kitti/setup.php
    
    :param scan: Nx3 array of x, y, z coordinates
    :init_img: Initial image to draw on
    
    :return proj_img: Birdseye view image
    '''
    FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
    proj_img_size = 800
    proj_img = np.zeros((proj_img_size, proj_img_size, 3), dtype=np.uint8)

    x, y = scan[:, 2], -scan[:, 0]
    
    if color == "red":
        rgb_val = np.array([0, 0, 255])
    else:
        rgb_val = np.array([0, 100, 100])

    draw_x_, draw_y_ = (proj_img_size/2 - draw_scale*x).astype(int), (proj_img_size/2 - draw_scale*y).astype(int)
    draw_x = draw_x_[np.logical_and(draw_x_ < proj_img_size, draw_y_ < proj_img_size)]
    draw_y = draw_y_[np.logical_and(draw_x_ < proj_img_size, draw_y_ < proj_img_size)]
    draw_x = np.clip(draw_x, 0, proj_img_size-1)
    draw_y = np.clip(draw_y, 0, proj_img_size-1)
    
    proj_img[draw_x, draw_y] = rgb_val
    if dilate:
        proj_img = cv2.dilate(proj_img, FULL_KERNEL_3)

    # TODO overlay part is badly implemented, take care of it later
    if init_img is not None:
        init_img = init_img.copy()
        init_img[np.nonzero(proj_img)] = 255
        return init_img

    return proj_img


def project_unproject(scan, scan_next, row_scale=64, col_scale=1024, inpaint=True, filter="feat", thresh=0.9, verbose=False, draw=False):
    '''
    This function implements the following
    1. Project lidar scan to image space
    2. Get SIFT features from the image
    3. Match SIFT features from one image to another
    4. Project the matched keypoints back to 3D space
    
    :param scan: Nx3 array of x, y, z coordinates 
    :param scan_next: Nx3 array of x, y, z coordinates
    
    :return x, y, z: 3D points from the matched keypoints
    '''
    
    # Lidar to Depth projection 
    image_proj, pyr, unnorm_image_proj = scan_to_image(scan, row_scale, col_scale, inpaint=inpaint)
    image_proj_next, pyr_next, unnorm_image_proj_next = scan_to_image(scan_next, row_scale, col_scale, inpaint=inpaint)

    # Get SIFT features from depth projection
    kps, des = get_sift_features(image_proj, draw_kp=False)
    kps_next, des_next = get_sift_features(image_proj_next, draw_kp=False)
    
    matches = match_sift_features(image_proj, kps, des, image_proj_next, kps_next, des_next, filter=filter, thresh=thresh, draw_matches=False) 
    if verbose:
        print(f"number of matches: {matches.shape[0]}")

    # Reproject back to 3D space 
    points_reproj = kp_to_points(unnorm_image_proj, matches[:,:2], pyr, row_scale, col_scale)
    points_reproj_next = kp_to_points(unnorm_image_proj_next, matches[:,2:], pyr_next, row_scale, col_scale)
    if verbose:
        print(f"number of reprojected points: {points_reproj.shape[0]}")

    # For visualizing SIFT match  
    if draw:
        image_proj_rot = cv2.rotate(image_proj, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image_proj_next_rot = cv2.rotate(image_proj_next, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        kps_rot, des_rot, kps_image_rot = get_sift_features(image_proj_rot)
        kps_next_rot, des_next_rot, kps_next_image_rot = get_sift_features(image_proj_next_rot)

        _, matches_image_rot = match_sift_features(image_proj_rot, kps_rot, des_rot, 
                                                    image_proj_next_rot, kps_next_rot, des_next_rot, filter=filter, thresh=thresh) 
        
        # returns reprojected 3D points, matches image, unnormalized depth projection 1, unnormalized depth projection 2,
        # keypoints image1, keypoints image2
        return np.hstack((points_reproj, points_reproj_next)), cv2.rotate(matches_image_rot, cv2.ROTATE_90_CLOCKWISE), \
            unnorm_image_proj, unnorm_image_proj_next, cv2.rotate(kps_image_rot, cv2.ROTATE_90_CLOCKWISE), \
            cv2.rotate(kps_next_image_rot, cv2.ROTATE_90_CLOCKWISE)

    return np.hstack((points_reproj, points_reproj_next))
