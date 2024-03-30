'''
Author: Bharath Kumar Ramesh Babu
Email: kumar7bharath@gmail.com
'''

import numpy as np
import os
import cv2

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

from net.lodonet import MKPSelectionModule, RotationTranslationEstimationModule
from config.config import dataset_config, train_inference_config
from utils.projection_utils import project_birdseye, proj_3d_to_2d
from utils.dataset_utils import KittiLidarOdomDataset, read_calib_file, shortlist_points_with_mkp_ranking
from utils.common_utils import homo_trans_mat_inv, convert_to_homo_coords
    
def draw_kp(smatches, image_proj, image_proj_next, gt):
    smatches = smatches.transpose(1, 2)
    smatches = smatches.detach().cpu().numpy().squeeze()
    
    smatches_kps1 = smatches[:, :3]
    smatches_kps2 = smatches[:, 3:]
    
    print("Ground Truth validation")
    print(f"centroid difference: {np.mean(smatches_kps1 - smatches_kps2, axis=0)}")
    print(f"odometry ground truth translation: {gt.numpy().squeeze()[:3]}")
            
    skps1_np, _ = proj_3d_to_2d(smatches_kps1) 
    skps2_np, _ = proj_3d_to_2d(smatches_kps2)
                
    skps1 = [cv2.KeyPoint(int(skps1_np[i][0]), 1023 - int(skps1_np[i][1]), 1) for i in range(skps1_np.shape[0])]
    skps2 = [cv2.KeyPoint(int(skps2_np[i][0]), 1023 - int(skps2_np[i][1]), 1) for i in range(skps2_np.shape[0])]
    
    matches = []
    for i in range(skps1_np.shape[0]):
        matches.append([cv2.DMatch(i, i, 0)])
                
    pp_matches_image = cv2.drawMatchesKnn(cv2.rotate(image_proj, cv2.ROTATE_90_COUNTERCLOCKWISE), skps1, \
      cv2.rotate(image_proj_next, cv2.ROTATE_90_COUNTERCLOCKWISE), skps2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
    rotated_pp_matches = cv2.rotate(pp_matches_image, cv2.ROTATE_90_CLOCKWISE)
    return rotated_pp_matches
 
def preprocess_single_pass(dataloader, frame):
    dataloader_iter = iter(dataloader) 
    for i in range(frame):
      next(dataloader_iter)
    sample = next(dataloader_iter)
    
    scans = sample[0]
    gt = sample[1]
    mkp_gt = sample[2]

    # Matches image from OpenCV function
    matches_image = sample[3].numpy().squeeze()
    
    # Image projection of consecutive frames
    image_proj = sample[4].numpy().squeeze()
    image_proj = image_proj*255/np.max(image_proj)
    image_proj = np.round(image_proj).astype("uint8")
    image_proj = cv2.equalizeHist(image_proj)

    image_proj_next = sample[5].numpy().squeeze()
    image_proj_next = image_proj_next*255/np.max(image_proj_next)
    image_proj_next = np.round(image_proj_next).astype("uint8")
    image_proj_next = cv2.equalizeHist(image_proj_next)

    # Keypoints image from OpenCV function 
    kps_image = sample[6].numpy().squeeze()
    kps_next_image = sample[7].numpy().squeeze()

    # Scans from the KittiDataset class
    init_scan = sample[8].numpy().squeeze()
    
    return scans, gt, mkp_gt, matches_image, image_proj, image_proj_next, kps_image, kps_next_image, init_scan

def visualize(scans, gt, matches_image, image_proj, image_proj_next, kps_image, kps_next_image, init_scan):
    # Scans from the DataLoader
    scans = scans.cpu().detach().numpy().squeeze()
    scan1 = scans[:,:3]      
    scan2 = scans[:,3:]
    
    init_scan1 = init_scan[:,:3]
    init_scan2 = init_scan[:,3:]

    # Birdseye view of the scans
    birdseye_initscan1 = project_birdseye(init_scan1)
    birdseye_initscan2 = project_birdseye(init_scan2)

    # Overlay preprocessed MKPs
    birdseye_initscan1 = project_birdseye(scan1, dilate=True, color="red", init_img=birdseye_initscan1)
    birdseye_initscan2 = project_birdseye(scan2, dilate=True, color="red", init_img=birdseye_initscan2)

    scan1_np, _ = proj_3d_to_2d(scan1) 
    scan2_np, _ = proj_3d_to_2d(scan2)

    scan1_kp = [cv2.KeyPoint(int(scan1_np[i][0]), 1023 - int(scan1_np[i][1]), 1) for i in range(scan1_np.shape[0])]
    scan2_kp = [cv2.KeyPoint(int(scan2_np[i][0]), 1023 - int(scan2_np[i][1]), 1) for i in range(scan2_np.shape[0])]
    
    scan_matches = []
    for i in range(scan1_np.shape[0]):
        scan_matches.append([cv2.DMatch(i, i, 0)])
                
    scan_matches_image = cv2.drawMatchesKnn(cv2.rotate(image_proj, cv2.ROTATE_90_COUNTERCLOCKWISE), scan1_kp, \
      cv2.rotate(image_proj_next, cv2.ROTATE_90_COUNTERCLOCKWISE), scan2_kp, scan_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    scan_matches_image = cv2.rotate(scan_matches_image, cv2.ROTATE_90_CLOCKWISE)
                  
    cv2.imshow('Preprocessed Sift Matches', matches_image)
    cv2.imshow('Input Matches', scan_matches_image)

    # cv2.imshow('Birds Eye Scan 1', birdseye_initscan1)
    # cv2.imshow('Birds Eye Scan 2', birdseye_initscan2)
    
    # cv2.imshow('Projected Image 1', image_proj)
    # cv2.imshow('Projected Image 2', image_proj_next)
    
    # cv2.imshow('Keypoints Image 1', kps_image)
    # cv2.imshow('Keypoints Image 2', kps_next_image)

    cv2.waitKey(0) 

def inference_single_pass(scans, model, loss, gt, mkp_gt, device):
    mkp_model, r_model, t_model = model
    mkp_loss, r_loss, t_loss = loss

    mkp_model.eval()
    r_model.eval()
    t_model.eval()
    
    scans = scans.transpose(1, 2).to(device)
    ranking = mkp_model(scans)
    sigmoid_ranking = torch.round(torch.sigmoid(ranking)).int()
    
    print(f"Number of MKP after selection: {sigmoid_ranking.sum()}")
    print(f"Number of MKP in ground truth: {mkp_gt.sum()}")
    
    # Check ground truth 
    smatches_gt = shortlist_points_with_mkp_ranking(scans, mkp_gt)    
    smatches = shortlist_points_with_mkp_ranking(scans, sigmoid_ranking)
        
    rotated_pp_matches = draw_kp(smatches, image_proj, image_proj_next, gt)
    rotated_pp_matches_gt = draw_kp(smatches_gt, image_proj, image_proj_next, gt)
    
    cv2.imshow('Ground Truth', rotated_pp_matches_gt)
    cv2.imshow('Matches by LodoNet', rotated_pp_matches)
    return rotated_pp_matches

if __name__ == "__main__":
    batch_size = 1
    num_workers = 1
    sequence = 0
    frame = 25
    top_n = train_inference_config.TOP_N

    inference_data = KittiLidarOdomDataset(dataset_config.LIDAR_PATH, dataset_config.GT_PATH, include_sequences=[sequence], pre_process=True, augument=False, top_k=top_n, draw=True)
    inference_dataloader = DataLoader(inference_data, batch_size=batch_size, num_workers=num_workers)

    scans, gt, mkp_gt, matches_image, image_proj, image_proj_next, kps_image, kps_next_image, init_scan = preprocess_single_pass(inference_dataloader, frame)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    mkp_model = MKPSelectionModule()
    mkp_model.to(device)
    mkp_model.double()

    r_model = RotationTranslationEstimationModule()
    r_model.to(device)
    r_model.double()

    t_model = RotationTranslationEstimationModule()
    t_model.to(device)
    t_model.double()
    
    mkp_ckpt = torch.load(os.path.join(dataset_config.CWD, 'ckpt/mkp_checkpoint_epoch80.pth'), map_location=device)
    r_ckpt = torch.load(os.path.join(dataset_config.CWD, 'ckpt/r_checkpoint_epoch80.pth'), map_location=device)
    t_ckpt = torch.load(os.path.join(dataset_config.CWD, 'ckpt/t_checkpoint_epoch80.pth'), map_location=device)

    mkp_model.load_state_dict(mkp_ckpt, strict=True)
    r_model.load_state_dict(r_ckpt, strict=True)
    t_model.load_state_dict(t_ckpt, strict=True)

    mkp_loss = nn.BCEWithLogitsLoss()
    r_loss = nn.MSELoss()
    t_loss = nn.MSELoss()

    with torch.no_grad():
        smatches = inference_single_pass(scans, [mkp_model, r_model, t_model], [mkp_loss, r_loss, t_loss], 
                                         gt, mkp_gt, device)
    
    visualize(scans, gt, matches_image, image_proj, image_proj_next, kps_image, kps_next_image, init_scan)
