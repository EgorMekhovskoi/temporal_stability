# Temporal filtering occupancy change based dynamic points removal

## Overview
Small ROS2 repo related to the dynamic points removal temporal filtering algorithm. It provides as the output two topics with dynamic and static points from the initial pointcloud and vehicle pose.

## Input

 - PointCloud2 topic named /loxo/point_cloud
 - PoseStamped topic named /loxo/vehicle_pose

## Output

 - PointCloud2 topic named /points_static
 - PointCloud2 topic named /points_static

## Build instructions

 1. Pull the repo
 2. cd to the pulled folder 
 3. Build the node
	 Source the workspace:
    ```sh
    colcon build
    ```
 4. Source the workspace:
    ```sh
    source install/setup.bash
    ```
## Usage
todo

