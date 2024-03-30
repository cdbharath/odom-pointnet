'''
Author: Bharath Kumar Ramesh Babu
Email: kumar7bharath@gmail.com
'''

import sys
sys.path.append('..')

import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import time

from utils.dataset_utils import load_kitti_velo_scan
from config.config import dataset_config

VOXEL_SIZE = 0.05
VISUALIZE = True

def pcd2xyz(pcd):
    return np.asarray(pcd.points).T

def extract_fpfh(pcd, voxel_size):
  radius_normal = voxel_size * 2
  pcd.estimate_normals(
      o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

  radius_feature = voxel_size * 5
  fpfh = o3d.pipelines.registration.compute_fpfh_feature(
      pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
  return np.array(fpfh.data).T

def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
  feat1tree = cKDTree(feat1)
  dists, nn_inds = feat1tree.query(feat0, k=knn)
  if return_distance:
    return nn_inds, dists
  else:
    return nn_inds

def find_correspondences(feats0, feats1, mutual_filter=True):
  nns01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
  corres01_idx0 = np.arange(len(nns01))
  corres01_idx1 = nns01

  if not mutual_filter:
    return corres01_idx0, corres01_idx1

  nns10 = find_knn_cpu(feats1, feats0, knn=1, return_distance=False)
  corres10_idx1 = np.arange(len(nns10))
  corres10_idx0 = nns10

  mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
  corres_idx0 = corres01_idx0[mutual_filter]
  corres_idx1 = corres01_idx1[mutual_filter]

  return corres_idx0, corres_idx1

if __name__ == "__main__":
    start = time.time()

    # Load and visualize scan
    scan = load_kitti_velo_scan(dataset_config.LIDAR_PATH, 0, 100)[:, :3]
    scan_next = load_kitti_velo_scan(dataset_config.LIDAR_PATH, 0, 101)[:, :3]    

    A_pcd_raw = o3d.geometry.PointCloud()
    A_pcd_raw.points = o3d.utility.Vector3dVector(scan)
    B_pcd_raw = o3d.geometry.PointCloud()
    B_pcd_raw.points = o3d.utility.Vector3dVector(scan_next)
    
    A_pcd_raw.paint_uniform_color([0.0, 0.0, 1.0]) # show A_pcd in blue
    B_pcd_raw.paint_uniform_color([1.0, 0.0, 0.0]) # show B_pcd in red
    if VISUALIZE:
        o3d.visualization.draw_geometries([A_pcd_raw,B_pcd_raw]) # plot A and B 
    
    # voxel downsample both clouds
    A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    if VISUALIZE:
        o3d.visualization.draw_geometries([A_pcd,B_pcd]) # plot downsampled A and B 
    
    A_xyz = pcd2xyz(A_pcd) # np array of size 3 by N
    B_xyz = pcd2xyz(B_pcd) # np array of size 3 by M
    
    # extract FPFH features
    A_feats = extract_fpfh(A_pcd,VOXEL_SIZE)
    B_feats = extract_fpfh(B_pcd,VOXEL_SIZE)
    
    # establish correspondences by nearest neighbour search in feature space
    corrs_A, corrs_B = find_correspondences(
        A_feats, B_feats, mutual_filter=True)
    A_corr = A_xyz[:,corrs_A] # np array of size 3 by num_corrs
    B_corr = B_xyz[:,corrs_B] # np array of size 3 by num_corrs
    
    num_corrs = A_corr.shape[1]
    print(f'FPFH generates {num_corrs} putative correspondences.')
    
    end = time.time()
    print(f'time taken: {end - start} s')
    
    # visualize the point clouds together with feature correspondences
    points = np.concatenate((A_corr.T,B_corr.T),axis=0)
    lines = []
    for i in range(num_corrs):
        lines.append([i,i+num_corrs])
    colors = [[0, 1, 0] for i in range(len(lines))] # lines are shown in green
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([A_pcd,B_pcd,line_set])
        