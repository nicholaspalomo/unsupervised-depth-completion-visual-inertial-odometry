'''
Author: Nicholas Palomo

Create Costar dataset in KITTI format from raw measurements file obtained from ROS bags.
'''

import sys, os, glob, argparse
sys.path.insert(0, 'src')
import numpy as np
import h5py
import multiprocessing as mp
import cv2, data_utils

'''
    Paths for Costar dataset
'''
COSTAR_DATA_DIRPATH = os.path.join('data', 'costar.h5')

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

        print('done!')
        
if __name__ == '__main__':

    params = (0, 1, 2)

    process_frame(params)
