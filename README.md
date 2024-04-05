# Pointnet based lidar odometry prediction in 2D space

### Odometry prediction with KITTI Lidar dataset

<div align="left">
  <img src="odom.gif" alt="Image 1" width="450" height="400" style="margin-right: 5px;">
  <img src="map.png" alt="Image 2" width="400" height="400">
</div>

This work implements the following steps
1. Extract lidar scans
2. Convert to depth image with spherical projection
3. KD-Tree based depth completion
4. Find feature matches with ORB features
5. Reproject back to euclidean space 
3. PointNet based architecture for realtime odometry prediction

### Results

<div align="left">
  <img src="model_output.png" alt="Image 1" width="450" height="400" style="margin-right: 5px;">
  <img src="icp_output.png" alt="Image 2" width="400" height="400">
</div>
Graph in the left shows the odometry prediction of the point-based model relative to ground truth in KITTI city 0. The graph in the right shows the performance for ICP for the same city. 


