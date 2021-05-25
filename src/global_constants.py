'''
Authors: Alex Wong <alexw@cs.ucla.edu>, Xiaohan Fei <feixh@cs.ucla.edu>
If you use this code, please cite the following paper:
A. Wong, X. Fei, S. Tsuei, and S. Soatto. Unsupervised Depth Completion from Visual Inertial Odometry.
https://arxiv.org/pdf/1905.08616.pdf
@article{wong2020unsupervised,
  title={Unsupervised Depth Completion From Visual Inertial Odometry},
  author={Wong, Alex and Fei, Xiaohan and Tsuei, Stephanie and Soatto, Stefano},
  journal={IEEE Robotics and Automation Letters},
  volume={5},
  number={2},
  pages={1899--1906},
  year={2020},
  publisher={IEEE}
}
'''
'''
  Global constants
'''
# Input image dimensions
N_BATCH             = 2
N_HEIGHT            = 240
N_WIDTH             = 424
N_CHANNEL           = 3
# Network Hyperparameters
N_EPOCH             = 50
LEARNING_RATES      = [1.2e-4, 0.6e-4, 0.3e-4]
LEARNING_BOUNDS     = [18, 24]
LEARNING_RATES_TXT  = '1.2e-4,0.6e-4,0.3e-4'
LEARNING_BOUNDS_TXT = '18,24'
MIN_Z               = 0.1
MAX_Z               = 10.0
POSE_NORM           = 'frobenius'
ROT_PARAM           = 'exponential'
# Preprocessing
OCC_THRESHOLD       = 1.5
OCC_KSIZE           = 7
# Network Structure
NET_TYPE            = 'vggnet08'
IM_FILTER_PCT       = 0.75
SZ_FILTER_PCT       = 0.25
# Weights for loss function
W_PH                = 1.00
W_CO                = 0.20
W_ST                = 0.80
W_SM                = 0.01
W_SZ                = 0.20
W_PC                = 0.10
# Checkpoint paths
CHECKPOINT_PATH     = 'log'
RESTORE_PATH        = ''
N_CHECKPOINT        = 5000
N_SUMMARY           = 100
OUTPUT_PATH         = 'out'
# Hardware settings
N_THREAD            = 100
EPSILON             = 1e-10

# For debugging evaluation script - hardcoded parameters
INTRINSICS_FILE     = '/media/nico/46B8200EB81FFB5F/workspace/costar_data/intrinsics/intrinsics.npy'
RESTORE_PATH        = '/media/nico/46B8200EB81FFB5F/workspace/unsupervised-depth-completion-visual-inertial-odometry/log/model.ckpt-5940'
IMAGE_PATH          = '/media/nico/46B8200EB81FFB5F/workspace/costar_data/validation/image/costar_val_image.txt'
INTERP_DEPTH_PATH   = '/media/nico/46B8200EB81FFB5F/workspace/costar_data/validation/interp_depth/costar_val_interp_depth.txt'
VALIDITY_MAP_PATH   = '/media/nico/46B8200EB81FFB5F/workspace/costar_data/validation/validity_map/costar_val_validity_map.txt'
GROUND_TRUTH_PATH   = '/media/nico/46B8200EB81FFB5F/workspace/costar_data/validation/sparse_depth/costar_val_sparse_depth.txt'
START_IDX           = 63
END_IDX             = 65
PLT_DENSE_PC        = True