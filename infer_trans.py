'''
Author: Bharath Kumar Ramesh Babu
Email: kumar7bharath@gmail.com
'''

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

import torch 
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from utils.dataset_utils import KittiLidarOdomDataset, shortlist_points_with_mkp_ranking
from net.lodonet import MKPSelectionModule, RotationTranslationEstimationModule, LWPoseEstimationModule
from config.config import dataset_config, visualize_config, train_inference_config, log_config
from utils.common_utils import w_from_xyz, rotation_from_quaternion, homo_trans_mat_from_rot_trans, axis_angle_to_rotation_matrix
from utils.misc_utils import plot_vo_frame


def write_inference_to_file(translation, rotation_mat, sequence):
    '''
    Write inference to log file for benchmark evaluation
    '''
        
    with open(os.path.join(log_config.LOG_DIR, str(sequence).zfill(2) + ".txt"), 'a+') as f:
        f.write(f"{rotation_mat[0][0]} {rotation_mat[0][1]} {rotation_mat[0][2]} {translation[0]}" +
        f" {rotation_mat[1][0]} {rotation_mat[1][1]} {rotation_mat[1][2]} {translation[1]}" + 
        f" {rotation_mat[2][0]} {rotation_mat[2][1]} {rotation_mat[2][2]} {translation[2]}\n")        
    
def get_trans(translation, rotation_quat):
    '''
    Gets rotation matrix from quaternion tensor
    Gets translation from tensor
    '''
    rotation_quat_cpu = rotation_quat.cpu()[0].numpy()
    translation_cpu = translation.cpu()[0].numpy()
    
    w = w_from_xyz(rotation_quat_cpu)
    rotation_quat_cpu = np.insert(rotation_quat_cpu, 0, w)
    
    rotation_mat = rotation_from_quaternion(rotation_quat_cpu)
    return translation_cpu, rotation_mat

def inference(model, dataloader, metric, sequence):
    r_model, t_model = model
    r_metric, t_metric = metric

    r_model.eval()
    t_model.eval()

    traj_img_size = visualize_config.TRAJ_IMAGE_SIZE
    traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
    draw_scale = visualize_config.DRAW_SCALE
    
    batch_count = 0
    trans_running_rmse = 0.0
    rot_running_rmse = 0.0

    gt_homo_trans_wrt_world = homo_trans_mat_from_rot_trans(np.eye(3), np.expand_dims(np.zeros(3), axis=1))
    homo_trans_wrt_world = homo_trans_mat_from_rot_trans(np.eye(3), np.expand_dims(np.zeros(3), axis=1))
    
    write_inference_to_file(np.zeros(3), np.eye(3), sequence)
    
    for sample in dataloader:
        scans = sample[0].to(device)
        scans = scans.transpose(1, 2)

        gt = sample[1].to(device)
        
        mkp_gt = sample[2].to(device)
        mkp_gt = mkp_gt.transpose(1, 2)
        
        # mkp_ranking = mkp_model(x)
        # sigmoid_ranking = torch.round(torch.sigmoid(mkp_ranking)).int()
        
        shortlist_points = shortlist_points_with_mkp_ranking(scans, mkp_gt)
        rotation = r_model(shortlist_points)        
        translation = t_model(shortlist_points)
        
        cur_translation, cur_rotation = get_trans(translation, rotation)
        
        cur_homo_trans = homo_trans_mat_from_rot_trans(cur_rotation, np.expand_dims(cur_translation, axis=1))
        homo_trans_wrt_world = homo_trans_wrt_world@cur_homo_trans
        
        cur_gt_translation, cur_gt_rotation = get_trans(gt[:, :3], gt[:, 4:])
        cur_gt_homo_trans = homo_trans_mat_from_rot_trans(cur_gt_rotation, np.expand_dims(cur_gt_translation, axis=1))
        gt_homo_trans_wrt_world = gt_homo_trans_wrt_world@cur_gt_homo_trans
                
        write_inference_to_file(homo_trans_wrt_world[:3, 3], homo_trans_wrt_world[:3, :3], sequence)
        
        batch_count += 1
                
        traj_img = plot_vo_frame(gt_homo_trans_wrt_world[:3, 3], homo_trans_wrt_world[:3, 3], traj_img, traj_img_size, draw_scale)
        cv2.imshow('Trajectory', traj_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
   
    return trans_running_rmse/batch_count, rot_running_rmse/batch_count

if __name__ == "__main__":
    batch_size = 1
    num_workers = 1
    sequence = 0
    top_n = train_inference_config.TOP_N

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    inference_data = KittiLidarOdomDataset(dataset_config.LIDAR_PATH, dataset_config.GT_PATH, include_sequences=[sequence], pre_process=True, top_k=top_n)
    inference_dataloader = DataLoader(inference_data, batch_size=batch_size, num_workers=num_workers)

    r_model = LWPoseEstimationModule()
    t_model = LWPoseEstimationModule()
    
    r_ckpt = torch.load(os.path.join(dataset_config.CWD, 'ckpt/r_checkpoint_epoch70.pth'), map_location=device)
    t_ckpt = torch.load(os.path.join(dataset_config.CWD, 'ckpt/t_checkpoint_epoch70.pth'), map_location=device)

    r_model.load_state_dict(r_ckpt, strict=True)
    t_model.load_state_dict(t_ckpt, strict=True)

    r_model.to(device)
    t_model.to(device)
    
    r_model.double()
    t_model.double()

    r_loss = nn.MSELoss()
    t_loss = nn.MSELoss()

    with torch.no_grad():
        trans_rmse, rot_rmse = inference(model=[r_model, t_model], dataloader=inference_dataloader, metric=[r_loss, t_loss], sequence=sequence)
        print(f"Translation RMSE: {trans_rmse}\n Rotational RMSE: {rot_rmse}")