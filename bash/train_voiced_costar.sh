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
--n_epoch 50 \
--learning_rates 0.5e-4,0.25e-4,0.125e-4 \
--learning_bounds 18,24 \
--occ_threshold 1.5 \
--occ_ksize 7 \
--net_type vggnet08 \
--im_filter_pct 0.75 \
--sz_filter_pct 0.25 \
--min_predict_z 0.1 \
--max_predict_z 10.0 \
--w_ph 1.00 \
--w_co 0.20 \
--w_st 0.80 \
--w_sm 0.15 \
--w_sz 2.0 \
--w_pc 0.10 \
--pose_norm frobenius \
--rot_param exponential \
--n_summary 10 \
--n_checkpoint 10 \
--checkpoint_path log \
--restore_path /media/nico/46B8200EB81FFB5F/workspace/unsupervised-depth-completion-visual-inertial-odometry/log/model.ckpt-1840
