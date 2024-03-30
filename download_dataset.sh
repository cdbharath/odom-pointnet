mkdir kitti_lidar_data
cd kitti_lidar_data

# Download KITTI dataset
mkdir data_odometry_calib
cd data_odometry_calib
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip
unzip data_odometry_calib.zip
cd ..

mkdir data_odometry_poses
cd data_odometry_poses
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip
unzip data_odometry_poses.zip
cd ..

mkdir data_odometry_velodyne
cd data_odometry_velodyne
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip
unzip data_odometry_velodyne.zip
cd ..