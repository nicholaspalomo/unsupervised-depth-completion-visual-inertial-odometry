'''
Author: Nicholas Palomo

Create Costar dataset in KITTI format from raw measurements file obtained from ROS bags.
'''

import sys, os, glob, argparse
sys.path.insert(0, 'src')
import numpy as np
import math
import h5py
import multiprocessing as mp
import cv2, data_utils

'''
    Paths for Costar dataset
'''
TRAIN_SPLIT = 3./4.
VALIDATION_SPLIT = 1./4.

COSTAR_DATA_DIRPATH = os.path.join('..', 'costar.h5')
SPARSE_DEPTH_INPUT_PATH = os.path.join('..', 'costar_data', 'sparse_depth')
INTERP_DEPTH_OUTPUT_PATH = os.path.join('..', 'costar_data', 'interp_depth')
VALIDITY_MAP_OUTPUT_PATH = os.path.join('..', 'costar_data', 'validity_map')
IMAGE_OUTPUT_PATH = os.path.join('..', 'costar_data', 'image')
SEMI_DENSE_DEPTH_OUTPUT_PATH = os.path.join('..', 'costar_data', 'semi_dense_depth')
TRAIN_OUTPUT_REF_DIRPATH = os.path.join('..', 'costar_data', 'train')
VAL_OUTPUT_REF_DIRPATH = os.path.join('..', 'costar_data', 'validation')
TEST_OUTPUT_REF_DIRPATH = os.path.join('..', 'costar_data', 'test')
INTRINSICS_PATH = os.path.join('..', 'costar_data', 'data', 'intrinsics.npy')

def process_frame(params, debug=True):
    image0_idx, image1_idx, image2_idx = params

    with h5py.File(COSTAR_DATA_DIRPATH, 'r') as hf:
        # Read images and concatenate together
        image0 = np.reshape(hf['image']['image'][int(image0_idx), :], (hf['image']['resolution'][0], hf['image']['resolution'][1], 3))
        image1 = np.reshape(hf['image']['image'][int(image1_idx), :], (hf['image']['resolution'][0], hf['image']['resolution'][1], 3))
        image2 = np.reshape(hf['image']['image'][int(image2_idx), :], (hf['image']['resolution'][0], hf['image']['resolution'][1], 3))
        image = np.concatenate([image1, image0, image2], axis=1)

        sparse_point_cloud = hf['lidar']['lidar'][image0_idx, :]
        sparse_point_cloud = sparse_point_cloud[~np.isnan(sparse_point_cloud)]
        sparse_point_cloud = np.reshape(sparse_point_cloud, (-1,4))
        sparse_point_cloud = sparse_point_cloud[:, :3]

        # transform point cloud into camera frame
        sparse_point_cloud = np.matmul(
            np.hstack((sparse_point_cloud, np.ones((sparse_point_cloud.shape[0], 1)))),
            np.reshape(hf['lidar']['tf_velo2cam'][:], (4, 4)).T
        )

        # get the camera intrinsic matrix
        K = np.reshape(hf['image']['intrinsics'][:], (3, 3))
        h, w = hf['image']['resolution'][0], hf['image']['resolution'][1]

    # filter out the points which lie outside the camera FOV
    pc_cam_pos_z_idx = np.where((sparse_point_cloud[:, 2] > 0))[0]
    sparse_point_cloud = sparse_point_cloud[pc_cam_pos_z_idx, :] # points in front of the image plane
    sparse_point_cloud_img = np.matmul(sparse_point_cloud[:,:3], K.T) # point cloud given in the image frame
    sparse_point_cloud_img = sparse_point_cloud_img / sparse_point_cloud_img[:, -1, None]
    
    width_mask = ((sparse_point_cloud_img[:,0] >= 0) & (sparse_point_cloud_img[:,0] <= w))
    height_mask = ((sparse_point_cloud_img[:,1] >= 0) & (sparse_point_cloud_img[:,1] <= h))
    pc_img_mask_idx = np.where(width_mask & height_mask)[0]

    sparse_point_cloud = sparse_point_cloud[pc_img_mask_idx, :]

    sz, vm = data_utils.create_depth_with_validity_map(sparse_point_cloud, h, w, K)
    iz = data_utils.interpolate_depth(sz, vm)

    if debug:
        import cv2
        cv2.imshow('interpolated depth image', iz)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

    # Create output directories
    for output_path in [IMAGE_OUTPUT_PATH, INTERP_DEPTH_OUTPUT_PATH, VALIDITY_MAP_OUTPUT_PATH]:
        output_dirpath = os.path.dirname(output_path)
        if not os.path.exists(output_dirpath):
            try:
                os.makedirs(output_dirpath)
            except FileExistsError:
                pass

    # Write to disk
    data_utils.save_depth(iz, INTERP_DEPTH_OUTPUT_PATH)
    data_utils.save_validity_map(vm, VALIDITY_MAP_OUTPUT_PATH)
    cv2.imwrite(IMAGE_OUTPUT_PATH, image)

    return (IMAGE_OUTPUT_PATH, SPARSE_DEPTH_INPUT_PATH, INTERP_DEPTH_OUTPUT_PATH, VALIDITY_MAP_OUTPUT_PATH, SEMI_DENSE_DEPTH_OUTPUT_PATH)

# parser = argparse.ArgumentParser()

# parser.add_argument('--n_thread', type=int, default=8)
# args = parser.parse_args()

for dirpath in [TRAIN_OUTPUT_REF_DIRPATH, VAL_OUTPUT_REF_DIRPATH, TEST_OUTPUT_REF_DIRPATH]:
  if not os.path.exists(dirpath):
    os.makedirs(dirpath)

# Build a mapping between the camera intrinsics to the directories
with h5py.File(COSTAR_DATA_DIRPATH, 'r') as hf:
    # get the camera intrinsic matrix
    intrinsics = np.hstack((
        np.reshape(hf['image']['intrinsics'][:], (3, 3)),
        np.zeros((3, 1))
    ))
    # intrinsics["T_velo2cam"] = np.reshape(hf['lidar']['tf_velo2cam'][:], (4, 4))
    # intrinsics["h"], intrinsics["w"] = hf['image']['resolution'][0], hf['image']['resolution'][1]
np.save(INTRINSICS_PATH, intrinsics)

'''
    Create validity maps and paths for sparse and semi dense depth for training
'''
with h5py.File(COSTAR_DATA_DIRPATH, 'r') as hf:
    image_indexes = hf["image"]["image_idx"]
    lidar_indexes = hf["lidar"]["lidar_idx"]

train_indexes = image_indexes[:int(image_indexes.shape[0] * TRAIN_SPLIT)]
validation_indexes = image_indexes[int(image_indexes.shape[0] * TRAIN_SPLIT):]

for train_idx in train_indexes[:(3 * int(math.floor(train_indexes.shape[0] / 3)))]:
    if train_idx > 0 and train_idx < train_indexes.shape[0]-1:

        params = (train_idx, train_idx-1, train_idx+1)

        results = process_frame(params)

        image_output_path, sparse_depth_path, interp_depth_output_path, \
              validity_map_output_path, semi_dense_depth_path = results

print('Training data: Completed processing {} samples'.format(3 * int(math.floor(train_indexes.shape[0] / 3))))

'''
    Create validity maps and paths for sparse and semi dense depth for validation
'''
for validation_idx in validation_indexes:
    params = (validation_idx, validation_idx, validation_idx)

    results = process_frame(params)

    image_output_path, sparse_depth_path, interp_depth_output_path, \
            validity_map_output_path, semi_dense_depth_path = results

print('Validation data: Completed processing {} samples'.format(validation_indexes.shape[0]))


# if __name__ == '__main__':
