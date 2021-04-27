#!/bin/bash

mkdir -p costar_data

mkdir -p costar_data/kitti_depth_completion
mkdir -p costar_data/kitti_depth_completion/train_val_split
mkdir -p costar_data/kitti_depth_completion/train_val_split/sparse_depth
mkdir -p costar_data/kitti_depth_completion/train_val_split/ground_truth
mkdir -p costar_data/kitti_depth_completion/validation
mkdir -p costar_data/kitti_depth_completion/testing
mkdir -p costar_data/kitti_depth_completion/tmp

unzip data/data_depth_velodyne.zip -d data/kitti_depth_completion/train_val_split/sparse_depth
unzip data/data_depth_annotated.zip -d data/kitti_depth_completion/train_val_split/ground_truth
unzip data/data_depth_selection.zip -d data/kitti_depth_completion/tmp

mv data/kitti_depth_completion/tmp/depth_selection/val_selection_cropped/image data/kitti_depth_completion/validation/image
mv data/kitti_depth_completion/tmp/depth_selection/val_selection_cropped/velodyne_raw data/kitti_depth_completion/validation/sparse_depth
mv data/kitti_depth_completion/tmp/depth_selection/val_selection_cropped/groundtruth_depth data/kitti_depth_completion/validation/ground_truth
mv data/kitti_depth_completion/tmp/depth_selection/val_selection_cropped/intrinsics data/kitti_depth_completion/validation/intrinsics

mv data/kitti_depth_completion/tmp/depth_selection/test_depth_completion_anonymous/image data/kitti_depth_completion/testing/image
mv data/kitti_depth_completion/tmp/depth_selection/test_depth_completion_anonymous/velodyne_raw data/kitti_depth_completion/testing/sparse_depth
mv data/kitti_depth_completion/tmp/depth_selection/test_depth_completion_anonymous/intrinsics data/kitti_depth_completion/testing/intrinsics

rm -r data/kitti_depth_completion/tmp

python setup/setup_dataset_kitti.py
