'''
Author: Bharath Kumar Ramesh Babu
Email: kumar7bharath@gmail.com
'''

import sys
sys.path.append('..')

import cv2
import numpy as np
import os

from utils.projection_utils import project_birdseye
from utils.dataset_utils import KittiLidarOdomDataset
from utils.common_utils import homo_trans_mat_from_rot_trans, convert_to_homo_coords, rotation_from_quaternion
from config.config import dataset_config, visualize_config

def plot_vo_frame(gt, traj_img, traj_img_size, draw_scale):
    '''
    Plots the regressed VO trajectory and ground truth trajectory 
    '''
    x_gt = gt[0]
    y_gt = gt[1]
    z_gt = gt[2]
    draw_x_gt, draw_y_gt = int(draw_scale*x_gt) + int(traj_img_size/2), int(traj_img_size/2) - int(draw_scale*z_gt)
    cv2.circle(traj_img, (draw_x_gt, draw_y_gt), 1,(0, 0, 255), 1)  # groundtruth in red

    cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
    text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x_gt, y_gt, z_gt)
    cv2.putText(traj_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

    return traj_img

def visualize_match(city = 0, frame = 150, draw_scale=10):
    dataset = KittiLidarOdomDataset(dataset_config.LIDAR_PATH, dataset_config.GT_PATH, include_sequences=[city], pre_process=True, augument=True)
    scans, gt, mkp_gt = dataset[frame]
        
    rot = rotation_from_quaternion(gt[3:])
    trans = np.expand_dims(gt[:3], axis=1)
    homo_trans_diff = homo_trans_mat_from_rot_trans(rot, trans)

    prev_scan = scans[:, :3]
    cur_scan = scans[:, 3:]
    
    prev_scan = prev_scan[prev_scan[:, 1] < 1]
    cur_scan = cur_scan[cur_scan[:, 1] < 1]
    
    shortlist_pc = scans[mkp_gt.squeeze().astype("bool")]

    # MKP birdseye view, prev scan
    prev_scan_img = project_birdseye(prev_scan, dilate=True, draw_scale=draw_scale)
    cur_scan_img = project_birdseye(cur_scan, dilate=True, color="red", init_img=prev_scan_img, draw_scale=draw_scale)
    before_gt_trans = cur_scan_img.copy()

    cur_scan = ((homo_trans_diff@convert_to_homo_coords(cur_scan).T).T)[:, :3] 

    # MKP birdseye view, cur scan
    prev_scan_img = project_birdseye(prev_scan, dilate=True, draw_scale=draw_scale)
    cur_scan_img = project_birdseye(cur_scan, dilate=True, color="red", init_img=prev_scan_img, draw_scale=draw_scale)
    after_gt_trans = cur_scan_img.copy()
    
    # MKP shortlist, prev scan 
    prev_scan_sl = shortlist_pc[:, :3]
    cur_scan_sl = shortlist_pc[:, 3:]
    prev_scan_img = project_birdseye(prev_scan_sl, dilate=True, draw_scale=draw_scale)
    cur_scan_img = project_birdseye(cur_scan_sl, dilate=True, color="red", init_img=prev_scan_img, draw_scale=draw_scale)
    before_gt_trans_sl = cur_scan_img.copy()

    cur_scan_sl = ((homo_trans_diff@convert_to_homo_coords(cur_scan_sl).T).T)[:, :3]

    # MKP shortlist, cur scan 
    prev_scan_img = project_birdseye(prev_scan_sl, dilate=True, draw_scale=draw_scale)
    cur_scan_img = project_birdseye(cur_scan_sl, dilate=True, color="red", init_img=prev_scan_img, draw_scale=draw_scale)
    after_gt_trans_sl = cur_scan_img.copy()

    return before_gt_trans, after_gt_trans, gt, before_gt_trans_sl, after_gt_trans_sl 


if __name__ == "__main__":
    mode = "video"
    city = 0
    frame = 0

    if mode == "frame":
        before_gt_trans, after_gt_trans, gt, before_gt_trans_sl, after_gt_trans_sl = visualize_match(city, frame)
        cv2.imshow('Before ground truth transform', before_gt_trans)
        cv2.imshow('After ground truth transform', after_gt_trans)    

        cv2.waitKey(0)
    elif mode == "video":
        gt_trans_wrt_world = np.zeros(3)
        gt_rot_wrt_world = np.eye(3)
        gt_homo_trans_wrt_world = homo_trans_mat_from_rot_trans(gt_rot_wrt_world, np.expand_dims(gt_trans_wrt_world, axis=1))
    
        traj_img_size = visualize_config.TRAJ_IMAGE_SIZE
        traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
        draw_scale = visualize_config.DRAW_SCALE

        while True:
            before_gt_trans, after_gt_trans, gt, before_gt_trans_sl, after_gt_trans_sl = visualize_match(city, frame)
            frame = frame + 1
            print(frame)
            
            gt_rot = rotation_from_quaternion(gt[3:])
            gt_trans = np.expand_dims(gt[:3], axis=1)
            gt_homo_trans = homo_trans_mat_from_rot_trans(gt_rot, gt_trans)            
            gt_homo_trans_wrt_world = gt_homo_trans_wrt_world@gt_homo_trans 
            gt = [gt_homo_trans_wrt_world[0, 3], 
                  gt_homo_trans_wrt_world[1, 3],
                  gt_homo_trans_wrt_world[2, 3]]

            traj_img = plot_vo_frame(gt, traj_img, traj_img_size, draw_scale)
            cv2.imshow('Trajectory', traj_img)
            
            cv2.imshow('Before ground truth transform', before_gt_trans)
            cv2.imshow('After ground truth transform', after_gt_trans)    
            
            cv2.imshow('Shortlist before ground truth transform', before_gt_trans_sl)
            cv2.imshow('Shortlist after ground truth transform', after_gt_trans_sl)    
            cv2.waitKey(1)

