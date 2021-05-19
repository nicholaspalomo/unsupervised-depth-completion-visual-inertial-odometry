#!/bin/bash

python src/utils.py \
    --create_dataset \
    --hdf5_path data/costar.h5 \
    --robot_name spot2 \
    --camera_name camera_front \
    --camera_image_topic color/image_raw_throttle/compressed \
    --depth_camera_image_topic aligned_depth_to_color/image_raw_throttle/compressedDepth \
    --camera_info_topic color/camera_info \
    --depth_camera_info_topic aligned_depth_to_color/camera_info \
    --point_cloud_topic velodyne_points \
    --bags_path src/bags/ \
    --max_idx 800 \
    --img_idx 0