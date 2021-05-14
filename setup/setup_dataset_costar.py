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
import cv2
import data_utils

'''
    Paths for Costar dataset
'''
CAMERA = 'front' # 'front' or 'rear'
HAS_DEPTH_CAMERA = False

TRAIN_SPLIT = 0.75
VALIDATION_SPLIT = 0.25

COSTAR_DATA_DIRPATH = os.path.join('costar.h5')
TRAIN_OUTPUT_REF_DIRPATH = os.path.join('..', 'costar_data', 'train')
VAL_OUTPUT_REF_DIRPATH = os.path.join('..', 'costar_data', 'validation')
TEST_OUTPUT_REF_DIRPATH = os.path.join('..', 'costar_data', 'test')

SPARSE_DEPTH_INPUT_PATH = 'sparse_depth'
INTERP_DEPTH_OUTPUT_PATH = 'interp_depth'
VALIDITY_MAP_OUTPUT_PATH = 'validity_map'
IMAGE_OUTPUT_PATH = 'image'
SEMI_DENSE_OUTPUT_PATH = 'semi_dense_depth'
INTRINSICS_PATH = os.path.join('..', 'costar_data', 'intrinsics')

TRAIN_IMAGE_OUTPUT_FILEPATH = os.path.join(TRAIN_OUTPUT_REF_DIRPATH, IMAGE_OUTPUT_PATH, 'costar_train_image.txt')
TRAIN_SPARSE_DEPTH_OUTPUT_FILEPATH = os.path.join(TRAIN_OUTPUT_REF_DIRPATH, SPARSE_DEPTH_INPUT_PATH, 'costar_train_sparse_depth.txt')
TRAIN_INTERP_DEPTH_OUTPUT_FILEPATH = os.path.join(TRAIN_OUTPUT_REF_DIRPATH, INTERP_DEPTH_OUTPUT_PATH, 'costar_train_interp_depth.txt')
TRAIN_VALIDITY_MAP_OUTPUT_FILEPATH = os.path.join(TRAIN_OUTPUT_REF_DIRPATH, VALIDITY_MAP_OUTPUT_PATH, 'costar_train_validity_map.txt')
TRAIN_SEMI_DENSE_DEPTH_OUTPUT_FILEPATH = os.path.join(TRAIN_OUTPUT_REF_DIRPATH, SEMI_DENSE_OUTPUT_PATH, 'costar_train_semi_dense_depth.txt')

VAL_IMAGE_OUTPUT_FILEPATH = os.path.join(VAL_OUTPUT_REF_DIRPATH, IMAGE_OUTPUT_PATH, 'costar_val_image.txt')
VAL_SPARSE_DEPTH_OUTPUT_FILEPATH = os.path.join(VAL_OUTPUT_REF_DIRPATH, SPARSE_DEPTH_INPUT_PATH, 'costar_val_sparse_depth.txt')
VAL_INTERP_DEPTH_OUTPUT_FILEPATH = os.path.join(VAL_OUTPUT_REF_DIRPATH, INTERP_DEPTH_OUTPUT_PATH, 'costar_val_interp_depth.txt')
VAL_VALIDITY_MAP_OUTPUT_FILEPATH = os.path.join(VAL_OUTPUT_REF_DIRPATH, VALIDITY_MAP_OUTPUT_PATH, 'costar_val_validity_map.txt')
VAL_SEMI_DENSE_DEPTH_OUTPUT_FILEPATH = os.path.join(VAL_OUTPUT_REF_DIRPATH, SEMI_DENSE_OUTPUT_PATH, 'costar_val_semi_dense_depth.txt')

TRAINED_MODEL_PATH = os.path.join('..', 'costar_data', 'trained_models')

def process_frame(params, output_path, sparse_point_cloud_prev, debug=False, camera=CAMERA):
    image0_idx, image1_idx, image2_idx = params

    with h5py.File(COSTAR_DATA_DIRPATH, 'r') as hf:
        # Read images and concatenate together
        image0 = np.reshape(hf['image']['image'][int(image0_idx), :], (hf['image']['resolution'][0], hf['image']['resolution'][1], 3))
        image1 = np.reshape(hf['image']['image'][int(image1_idx), :], (hf['image']['resolution'][0], hf['image']['resolution'][1], 3))
        image2 = np.reshape(hf['image']['image'][int(image2_idx), :], (hf['image']['resolution'][0], hf['image']['resolution'][1], 3))
        image = np.concatenate([image1, image0, image2], axis=1)
    
        sparse_point_cloud = hf['lidar']['lidar'][image1_idx, :]
        sparse_point_cloud = sparse_point_cloud[~np.isnan(sparse_point_cloud)]
        sparse_point_cloud = np.reshape(sparse_point_cloud, (-1,4))
        sparse_point_cloud = sparse_point_cloud[:, :3]

        sparse_point_cloud1 = hf['lidar']['lidar'][image0_idx, :]
        sparse_point_cloud1 = sparse_point_cloud1[~np.isnan(sparse_point_cloud1)]
        sparse_point_cloud1 = np.reshape(sparse_point_cloud1, (-1,4))
        sparse_point_cloud = np.vstack((sparse_point_cloud, sparse_point_cloud1[:, :3]))

        sparse_point_cloud2 = hf['lidar']['lidar'][image2_idx, :]
        sparse_point_cloud2 = sparse_point_cloud2[~np.isnan(sparse_point_cloud2)]
        sparse_point_cloud2 = np.reshape(sparse_point_cloud2, (-1,4))
        sparse_point_cloud = np.vstack((sparse_point_cloud, sparse_point_cloud2[:, :3]))

        # get the camera intrinsic matrix
        K = np.reshape(hf['image']['intrinsics'][:], (3, 3))
        h, w = hf['image']['resolution'][0], hf['image']['resolution'][1]

        if HAS_DEPTH_CAMERA:
            iz = 255 - np.reshape(hf['image']['depth_image'][int(image0_idx), :], (hf['image']['resolution'][0], hf['image']['resolution'][1]))

        if sparse_point_cloud.shape[0] < 1:
            print("Point cloud empty at line [{}, {}, {}]!".format(image1_idx, image0_idx, image2_idx))
            sparse_point_cloud = sparse_point_cloud_prev.copy()

        else:
            # transform point cloud into camera frame
            if camera == 'rear':
                T_camera_2_img = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
                T_camera_2_img = np.vstack((T_camera_2_img, np.zeros((1,3))))
                T_camera_2_img = np.hstack((T_camera_2_img, np.zeros((4,1))))
                T_camera_2_img[-1, -1] = 1.
                T_camera_2_img = T_camera_2_img.T

                T_velo_2_camera = np.matmul(T_camera_2_img, np.reshape(hf['lidar']['tf_velo2cam'][:], (4, 4)))
                sparse_point_cloud = np.matmul(
                    np.hstack((sparse_point_cloud, np.ones((sparse_point_cloud.shape[0], 1)))),
                    T_velo_2_camera.T
                )
            else:
                sparse_point_cloud = np.matmul(
                    np.hstack((sparse_point_cloud, np.ones((sparse_point_cloud.shape[0], 1)))),
                    np.reshape(hf['lidar']['tf_velo2cam'][:], (4, 4)).T
                )

                # filter out the points which lie outside the camera FOV
                pc_cam_pos_z_idx = np.where((sparse_point_cloud[:, 2] > 0))[0]
                sparse_point_cloud = sparse_point_cloud[pc_cam_pos_z_idx, :] # points in front of the image plane
                sparse_point_cloud_img = np.matmul(sparse_point_cloud[:,:3], K.T) # point cloud given in the image frame
                sparse_point_cloud_img = sparse_point_cloud_img / sparse_point_cloud_img[:, -1, None]
                
                width_mask = ((sparse_point_cloud_img[:,0] >= 0) & (sparse_point_cloud_img[:,0] <= w))
                height_mask = ((sparse_point_cloud_img[:,1] >= 0) & (sparse_point_cloud_img[:,1] <= h))
                pc_img_mask_idx = np.where(width_mask & height_mask)[0]

                sparse_point_cloud = sparse_point_cloud[pc_img_mask_idx, :]

    sz, vm = data_utils.create_depth_with_validity_map(sparse_point_cloud[:, :3], h, w, K)
    if not HAS_DEPTH_CAMERA:
        iz = data_utils.interpolate_depth(sz, vm)

    if debug:
        cv2.imshow('interpolated depth image', iz)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

    # Create output directories
    paths = [os.path.join(output_path, IMAGE_OUTPUT_PATH), os.path.join(output_path, INTERP_DEPTH_OUTPUT_PATH), os.path.join(output_path, VALIDITY_MAP_OUTPUT_PATH), os.path.join(output_path, SPARSE_DEPTH_INPUT_PATH), os.path.join(output_path, SEMI_DENSE_OUTPUT_PATH)]
    for path in paths:
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except FileExistsError:
                pass

    # Write to disk
    fname = "%06i.png" % image0_idx
    cv2.imwrite(os.path.join(paths[0], fname), image)
    data_utils.save_depth(iz, os.path.join(paths[1], fname))
    data_utils.save_validity_map(vm, os.path.join(paths[2], fname))
    data_utils.save_depth(data_utils.create_sparse_depth(sparse_point_cloud[:,:3],  h, w, K), os.path.join(paths[3], fname))

    semi_dense_path = paths[-1]
    paths = [os.path.join(path, fname) for path in paths[:-1]]

    return (paths[0], paths[3], paths[1], paths[2], semi_dense_path, sparse_point_cloud)

def main():

    # parser = argparse.ArgumentParser()

    # parser.add_argument('--n_thread', type=int, default=8)
    # args = parser.parse_args()

    for dirpath in [TRAIN_OUTPUT_REF_DIRPATH, VAL_OUTPUT_REF_DIRPATH, TEST_OUTPUT_REF_DIRPATH, TRAINED_MODEL_PATH, INTRINSICS_PATH]:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    # Build a mapping between the camera intrinsics to the directories
    with h5py.File(COSTAR_DATA_DIRPATH, 'r') as hf:
        # get the camera intrinsic matrix
        intrinsics = np.hstack((
            np.reshape(hf['image']['intrinsics'][:], (3, 3)),
            np.zeros((3, 1))
        ))

        intrinsics = intrinsics[:3, :3]
        # intrinsics["T_velo2cam"] = np.reshape(hf['lidar']['tf_velo2cam'][:], (4, 4))
        # intrinsics["h"], intrinsics["w"] = hf['image']['resolution'][0], hf['image']['resolution'][1]

    '''
        Create validity maps and paths for sparse and semi dense depth for training
    '''
    with h5py.File(COSTAR_DATA_DIRPATH, 'r') as hf:
        image_indexes = hf["image"]["image_idx"]
        lidar_indexes = hf["lidar"]["lidar_idx"]

        train_indexes = image_indexes[:int(image_indexes.shape[0] * TRAIN_SPLIT)]
        validation_indexes = image_indexes[int(image_indexes.shape[0] * TRAIN_SPLIT):]

    intrinsics_path_and_fname = os.path.join(INTRINSICS_PATH, 'intrinsics.npy')
    np.save(intrinsics_path_and_fname, intrinsics)
    with open(os.path.join(INTRINSICS_PATH, 'costar_train_intrinsics.txt'), "w") as o:
        for idx in range(len(train_indexes)-2):
            o.write(os.path.abspath(intrinsics_path_and_fname)+'\n')

    train_image_output_paths = []
    train_sparse_depth_output_paths = []
    train_interp_depth_output_paths = []
    train_validity_map_output_paths = []
    train_semi_dense_depth_output_paths = []
    sparse_point_cloud = None
    for train_idx in train_indexes:
        if train_idx > 0 and train_idx < train_indexes.shape[0]-1:

            params = (train_idx, train_idx-1, train_idx+1)

            results = process_frame(params, TRAIN_OUTPUT_REF_DIRPATH, sparse_point_cloud)

            image_output_path, sparse_depth_path, interp_depth_output_path, \
                validity_map_output_path, semi_dense_depth_path, sparse_point_cloud = results

            train_image_output_paths.append(image_output_path)
            train_sparse_depth_output_paths.append(sparse_depth_path)
            train_interp_depth_output_paths.append(interp_depth_output_path)
            train_validity_map_output_paths.append(validity_map_output_path)
            train_semi_dense_depth_output_paths.append(semi_dense_depth_path)

    print('Training data: Completed processing {} samples'.format(train_indexes.shape[0]))

    print('Storing training image file paths into: %s' % TRAIN_IMAGE_OUTPUT_FILEPATH)
    with open(TRAIN_IMAGE_OUTPUT_FILEPATH, "w") as o:
        for idx in range(len(train_indexes)-2):
            o.write(os.path.abspath(train_image_output_paths[idx])+'\n')

    print('Storing training sparse depth file paths into: %s' % TRAIN_SPARSE_DEPTH_OUTPUT_FILEPATH)
    with open(TRAIN_SPARSE_DEPTH_OUTPUT_FILEPATH, "w") as o:
        for idx in range(len(train_indexes)-2):
            o.write(os.path.abspath(train_sparse_depth_output_paths[idx])+'\n')

    print('Storing training interpolated depth file paths into: %s' % TRAIN_INTERP_DEPTH_OUTPUT_FILEPATH)
    with open(TRAIN_INTERP_DEPTH_OUTPUT_FILEPATH, "w") as o:
        for idx in range(len(train_indexes)-2):
            o.write(os.path.abspath(train_interp_depth_output_paths[idx])+'\n')

    print('Storing training validity map file paths into: %s' % TRAIN_VALIDITY_MAP_OUTPUT_FILEPATH)
    with open(TRAIN_VALIDITY_MAP_OUTPUT_FILEPATH, "w") as o:
        for idx in range(len(train_indexes)-2):
            o.write(os.path.abspath(train_validity_map_output_paths[idx])+'\n')

    print('Storing training semi dense depth file paths into: %s' % TRAIN_SEMI_DENSE_DEPTH_OUTPUT_FILEPATH)
    with open(TRAIN_SEMI_DENSE_DEPTH_OUTPUT_FILEPATH, "w") as o:
        for idx in range(len(train_semi_dense_depth_output_paths)):
            o.write(os.path.abspath(train_semi_dense_depth_output_paths[idx])+'\n')

    '''
        Create validity maps and paths for sparse and semi dense depth for validation
    '''
    validation_image_output_paths = []
    validation_sparse_depth_output_paths = []
    validation_interp_depth_output_paths = []
    validation_validity_map_output_paths = []
    validation_semi_dense_depth_output_paths = []
    for validation_idx in validation_indexes:
        params = (validation_idx, validation_idx, validation_idx)

        results = process_frame(params, VAL_OUTPUT_REF_DIRPATH, sparse_point_cloud)

        image_output_path, sparse_depth_path, interp_depth_output_path, \
                validity_map_output_path, semi_dense_depth_path, sparse_point_cloud = results

        validation_image_output_paths.append(image_output_path)
        validation_sparse_depth_output_paths.append(sparse_depth_path)
        validation_interp_depth_output_paths.append(interp_depth_output_path)
        validation_validity_map_output_paths.append(validity_map_output_path)
        validation_semi_dense_depth_output_paths.append(semi_dense_depth_path)


    print('Validation data: Completed processing {} samples'.format(validation_indexes.shape[0]))

    print('Storing validation image file paths into: %s' % VAL_IMAGE_OUTPUT_FILEPATH)
    with open(VAL_IMAGE_OUTPUT_FILEPATH, "w") as o:
        for idx in range(len(validation_indexes)-2):
            o.write(os.path.abspath(train_image_output_paths[idx])+'\n')

    print('Storing validation sparse depth file paths into: %s' % VAL_SPARSE_DEPTH_OUTPUT_FILEPATH)
    with open(VAL_SPARSE_DEPTH_OUTPUT_FILEPATH, "w") as o:
        for idx in range(len(validation_indexes)-2):
            o.write(os.path.abspath(train_sparse_depth_output_paths[idx])+'\n')

    print('Storing validation interpolated depth file paths into: %s' % VAL_INTERP_DEPTH_OUTPUT_FILEPATH)
    with open(VAL_INTERP_DEPTH_OUTPUT_FILEPATH, "w") as o:
        for idx in range(len(validation_indexes)-2):
            o.write(os.path.abspath(train_interp_depth_output_paths[idx])+'\n')

    print('Storing validation validity map file paths into: %s' % VAL_VALIDITY_MAP_OUTPUT_FILEPATH)
    with open(VAL_VALIDITY_MAP_OUTPUT_FILEPATH, "w") as o:
        for idx in range(len(validation_indexes)-2):
            o.write(os.path.abspath(train_validity_map_output_paths[idx])+'\n')

    print('Storing validation semi dense depth file paths into: %s' % VAL_SEMI_DENSE_DEPTH_OUTPUT_FILEPATH)
    with open(VAL_SEMI_DENSE_DEPTH_OUTPUT_FILEPATH, "w") as o:
        for idx in range(len(validation_indexes)-2):
            o.write(os.path.abspath(train_semi_dense_depth_output_paths[idx])+'\n')

if __name__ == '__main__':
    main()