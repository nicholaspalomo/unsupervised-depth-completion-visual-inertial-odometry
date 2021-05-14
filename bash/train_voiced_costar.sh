#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_voiced.py \
--train_image_path /media/nico/46B8200EB81FFB5F/workspace/costar_data/train/image/costar_train_image.txt \
--train_interp_depth_path /media/nico/46B8200EB81FFB5F/workspace/costar_data/train/interp_depth/costar_train_interp_depth.txt \
--train_validity_map_path /media/nico/46B8200EB81FFB5F/workspace/costar_data/train/validity_map/costar_train_validity_map.txt \
--train_intrinsics_path /media/nico/46B8200EB81FFB5F/workspace/costar_data/intrinsics/costar_train_intrinsics.txt \
--n_batch 2 \
--n_height 240 \
--n_width 424 \
--n_channel 3 \
--n_epoch 30 \
--learning_rates 1.2e-4,0.6e-4,0.3e-4 \
--learning_bounds 18,24 \
--occ_threshold 1.5 \
--occ_ksize 7 \
--net_type vggnet11 \
--im_filter_pct 0.75 \
--sz_filter_pct 0.25 \
--min_predict_z 1.5 \
--max_predict_z 100.0 \
--w_ph 1.00 \
--w_co 0.20 \
--w_st 0.80 \
--w_sm 0.01 \
--w_sz 0.20 \
--w_pc 0.10 \
--pose_norm frobenius \
--rot_param exponential \
--n_summary 1000 \
--n_checkpoint 5000 \
--checkpoint_path /media/nico/46B8200EB81FFB5F/workspace/unsupervised-depth-completion-visual-inertial-odometry/log
# --restore_path /media/nico/46B8200EB81FFB5F/workspace/costar_data/trained_models/vggnet11_kitti_model \

