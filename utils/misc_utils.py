'''
Author: Bharath Kumar Ramesh Babu
Email: kumar7bharath@gmail.com
'''

import numpy as np
import cv2

def sample_quat_y(x):
    '''
    Hardcoded sampling strategy for quaternion y component
    '''
    if -0.0015 <= x <= 0.0015:
        if np.random.random() < 0.4:
            return x
        else:
            if np.random.random() < 0.5:
                return np.random.uniform(-0.04, -0.0025)
            else:
                return np.random.uniform(0.0025, 0.04)
    else:
        return x
    
    
def plot_vo_frame(gt, pred, traj_img, traj_img_size, draw_scale):
    '''
    Plots the regressed VO trajectory and ground truth trajectory 
    '''
    x = pred[0]
    y = pred[1]
    z = pred[2]

    draw_x, draw_y = int(draw_scale*x) + int(traj_img_size/2), int(traj_img_size/2) - int(draw_scale*z)
    cv2.circle(traj_img, (draw_x, draw_y), 1,(0, 255, 0), 1)  

    x_gt = gt[0]
    z_gt = gt[2]
    draw_x_gt, draw_y_gt = int(draw_scale*x_gt) + int(traj_img_size/2), int(traj_img_size/2) - int(draw_scale*z_gt)
    cv2.circle(traj_img, (draw_x_gt, draw_y_gt), 1,(0, 0, 255), 1)  # groundtruth in red

    cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
    text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
    cv2.putText(traj_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

    return traj_img
