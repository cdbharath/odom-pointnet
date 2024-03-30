import sys
sys.path.append('..')

import cv2
import numpy as np
from torch.utils.data import Dataset
import os
from config.config import dataset_config
from utils.misc_utils import plot_vo_frame
from utils.common_utils import homo_trans_mat_from_rot_trans, homo_trans_mat_inv
from utils.dataset_utils import load_kitti_odometry_gt
from utils.projection_utils import get_sift_features, match_sift_features

WIDTH = 1241.0
HEIGHT = 376.0
FX = 718.8560
FY = 718.8560
CX = 607.1928
CY = 185.2157

CAMERA_MATRIX = np.array([[FX, 0,  CX],
                          [0,  FY, CY],
                          [0,  0,  1]])

def match_keypoints(frame1, frame2, draw_matches=False):
    # Find the keypoints and descriptors with ORB
    kp1, des1 = get_sift_features(frame1, n_feat=5000, draw_kp=False)
    kp2, des2 = get_sift_features(frame2, n_feat=5000, draw_kp=False)

    # Match descriptors
    matches = match_sift_features(frame1, kp1, des1, frame2, kp2, des2, filter="feat", thresh=0.7, optional_thresh=100, draw_matches=draw_matches)    
    return matches


def calculate_odometry(frame1, frame2):

    good_matches = match_keypoints(frame1, frame2)

    # Estimate essential matrix using RANSAC
    src_pts = good_matches[:, 2:].reshape(-1, 1, 2)
    dst_pts = good_matches[:, :2].reshape(-1, 1, 2)

    if len(good_matches) > 10:
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=FX, pp=(CX, CY), method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, focal=FX, pp=(CX, CY))
        status = True
    else:
        R = np.eye(3)
        t = np.zeros((3, 1))
        status = False

    # Draw matches
    frame1_rot = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame2_rot = cv2.rotate(frame2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    _, img_matches = match_keypoints(frame1_rot, frame2_rot, draw_matches=True)
    img_matches = cv2.rotate(img_matches, cv2.ROTATE_90_CLOCKWISE)

    return R, t, img_matches, status


def get_image(sequence, index):
    image =  cv2.imread(os.path.join(dataset_config.IMAGE_PATH, str(sequence).zfill(2), "image_0", str(index).zfill(6) + ".png"), cv2.IMREAD_GRAYSCALE)
    return image


class KittiImageOdomDataset(Dataset):
    def __init__(self, include_sequences=[0]):
        self.image_pairs = []
        self.image_dir = os.path.join(dataset_config.IMAGE_PATH)

        # Get paths for lidar pairs and ground truth from all the sequences
        sequences = sorted(os.listdir(self.image_dir))
        for sequence_idx in include_sequences:
            sequence = sequences[sequence_idx]

            scans = sorted(os.listdir(os.path.join(self.image_dir, sequence, "image_0")))
            for i in range(len(scans)):
                if i == 0:
                    continue
                self.image_pairs.append([(int(sequence), i - 1), (int(sequence), i)])

    def __len__(self):
        return len(self.image_pairs)
                
    def __getitem__(self, idx):
        prev_image = get_image(self.image_pairs[idx][0][0], self.image_pairs[idx][0][1])
        cur_image = get_image(self.image_pairs[idx][1][0], self.image_pairs[idx][1][1])
        
        # Load ground truth
        x_prev, y_prev, z_prev, rot_prev = load_kitti_odometry_gt(dataset_config.GT_PATH, self.image_pairs[idx][0][0], self.image_pairs[idx][0][1])
        x, y, z, rot = load_kitti_odometry_gt(dataset_config.GT_PATH, self.image_pairs[idx][1][0], self.image_pairs[idx][1][1])

        # Get translation and rotation relative to the previous frame
        prev_trans = np.expand_dims(np.array([x_prev, y_prev, z_prev]), axis=1)
        prev_homo_trans = homo_trans_mat_from_rot_trans(rot_prev, prev_trans)        
        
        trans = np.expand_dims(np.array([x, y, z]), axis=1)
        homo_trans = homo_trans_mat_from_rot_trans(rot, trans)
    
        homo_trans_diff = homo_trans_mat_inv(prev_homo_trans)@homo_trans
        return prev_image, cur_image, homo_trans_diff


if __name__ == "__main__":
    sequence = 0
    dataset = KittiImageOdomDataset(include_sequences=[sequence])
    
    traj_img_size = 800
    traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
    draw_scale = 0.7

    transformation_wrt_world_gt = homo_trans_mat_from_rot_trans(np.eye(3), np.expand_dims(np.zeros(3), axis=1))
    transformation_wrt_world = homo_trans_mat_from_rot_trans(np.eye(3), np.expand_dims(np.zeros(3), axis=1))

    for i in range(len(dataset)):
        prev_image, cur_image, transformation = dataset[i]        
        transformation_wrt_world_gt = transformation_wrt_world_gt@transformation 
        
        R, t, matches_image, _ = calculate_odometry(prev_image, cur_image)
        cur_transformation = homo_trans_mat_from_rot_trans(R, t)
        
        scale = np.sqrt(np.sum(transformation[:3,3]*transformation[:3,3]))
        if scale > 0.1:
            cur_transformation[:3, 3] = cur_transformation[:3, 3] * scale
            transformation_wrt_world = transformation_wrt_world@cur_transformation

        traj_img = plot_vo_frame(transformation_wrt_world_gt[:3, 3], transformation_wrt_world[:3, 3], traj_img, traj_img_size, draw_scale)
        cv2.imshow('Trajectory', traj_img)
        cv2.imshow('Match Image', matches_image)
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        