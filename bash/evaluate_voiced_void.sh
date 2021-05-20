#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/evaluate_model.py \
--image_path /media/nico/46B8200EB81FFB5F/workspace/costar_data/validation/image/costar_val_image.txt \
--interp_depth_path /media/nico/46B8200EB81FFB5F/workspace/costar_data/validation/interp_depth/costar_val_interp_depth.txt \
--validity_map_path /media/nico/46B8200EB81FFB5F/workspace/costar_data/validation/validity_map/costar_val_validity_map.txt \
--ground_truth_path /media/nico/46B8200EB81FFB5F/workspace/costar_data/validation/validity_map/costar_val_validity_map.txt \
--start_idx 63 \
--end_idx 80 \
--n_batch 3 \
--n_height 240 \
--n_width 424 \
--occ_threshold 1.5 \
--occ_ksize 7 \
--net_type vggnet08 \
--im_filter_pct 0.75 \
--sz_filter_pct 0.25 \
--min_predict_z 0.1 \
--max_predict_z 10.0 \
--min_evaluate_z 0.1 \
--max_evaluate_z 10.0 \
--save_depth \
--output_path /media/nico/46B8200EB81FFB5F/workspace/unsupervised-depth-completion-visual-inertial-odometry/log/output_path \
--intrinsics_file  /media/nico/46B8200EB81FFB5F/workspace/costar_data/intrinsics/intrinsics.npy \
--restore_path /media/nico/46B8200EB81FFB5F/workspace/unsupervised-depth-completion-visual-inertial-odometry/log/model.ckpt-5940
