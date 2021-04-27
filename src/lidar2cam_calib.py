import argparse
import cv2
from matplotlib import pyplot as plt
import numpy as np
import h5py
import os
import pathlib

import sys

from numpy.lib.type_check import imag
sys.path.append('.')

# get local gradient magnitudes and directions of the depth map in the area around the lidar scan
# normalize these gradients
# compute the optimal warp to align them with detected Canny edges

def get_intensities(lidar, reflectivity, image, T, K, h, w):

    # transform point cloud into camera frame
    sparse_point_cloud = np.matmul(
        np.hstack((lidar, np.ones((lidar.shape[0], 1)))),
        T.T
    )

    # filter out the points which lie outside the camera FOV
    pc_cam_pos_z_idx = np.where((sparse_point_cloud[:, 2] > 0))[0]
    sparse_point_cloud = sparse_point_cloud[pc_cam_pos_z_idx, :] # points in front of the image plane
    sparse_point_cloud_img = np.matmul(sparse_point_cloud[:,:3], K.T) # point cloud given in the image frame
    sparse_point_cloud_img = sparse_point_cloud_img / sparse_point_cloud_img[:, -1, None]
    
    width_mask = ((sparse_point_cloud_img[:,0] >= 0) & (sparse_point_cloud_img[:,0] <= w))
    height_mask = ((sparse_point_cloud_img[:,1] >= 0) & (sparse_point_cloud_img[:,1] <= h))
    pc_img_mask_idx = np.where(width_mask & height_mask)[0]

    lidar_intensities = reflectivity[pc_cam_pos_z_idx]
    lidar_intensities = lidar_intensities[pc_img_mask_idx]

    sparse_point_cloud = sparse_point_cloud[pc_img_mask_idx, :]
    sparse_point_cloud_img = sparse_point_cloud_img[pc_img_mask_idx, :]

    image_intensities = []
    for point in sparse_point_cloud_img:
        u, v = min(int(point[0]), h-1), min(int(point[1]), w-1)

        image_intensities.append(image[u, v])
    image_intensities = np.array(image_intensities)

    return lidar_intensities, image_intensities, sparse_point_cloud_img

def compute_mutual_information():

    return # mutual_information

def compute_joint_pdf(lidar_intensities, image_intensities):
    
    pXY = []
    for lidar_i, image_i in zip(lidar_intensities, image_intensities):
        kde = 0
        for xi, yi in zip(lidar_intensities, image_intensities):
            x = np.array([[lidar_i - xi], [image_i - yi]])
            cov = np.cov(x[:,0], x[:,0])
            cov = np.sqrt(cov) + np.eye(2) * 1e-3
            kde += np.exp(np.matmul(x.T, np.matmul(np.linalg.inv(cov), x)))
        pXY.append(1 / lidar_intensities.shape[0] * kde)

    return cov

def main(args):
    INDEX = 0

    with h5py.File(args.data_file, 'r') as h5:
        lidar = h5['lidar']['lidar'][INDEX, :]
        image = h5['image']['image'][INDEX, :, :]
        resolution = h5['image']['resolution'][:]
        tf_velo2cam = h5['lidar']['tf_velo2cam'][:]
        K = h5['image']['intrinsics'][:]

    # compute Canny edge detection
    image = np.reshape(image, (resolution[0], -1, 3))
    lidar = np.reshape(lidar[~np.isnan(lidar)], (-1, 4))
    reflectivity = lidar[:, -1]
    lidar = lidar[:, :3]
    tf_velo2cam = np.reshape(tf_velo2cam, (4, 4))
    K = np.reshape(K, (3, 3))

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    lidar_intensities, image_intensities, lidar_image = get_intensities(lidar, reflectivity, gray, tf_velo2cam, K, resolution[0], resolution[1])

    plt.imshow(gray, cmap='gray')
    plt.scatter(lidar_image[:,0], lidar_image[:,1])
    plt.title('Grayscale Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    # X - laser reflectivity value
    # Y - corresponding grayscale value of the image pixel to which this 3D point is projected

    bins = np.array(range(2**8))
    histogram= np.zeros((2**8,))
    for i in range(lidar_intensities.shape[0]):
        j = np.where(bins - lidar_intensities[i] > 0)[0][0]
        histogram[j] += 1

    pdfX = histogram.copy()
    sum = np.sum(pdfX)
    pdfX /= sum

    plt.plot(bins, pdfX)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Lidar Pixel Intensity')
    plt.ylabel('Probability')
    plt.show()

    histogram= np.zeros((2**8,))
    for i in range(image_intensities.shape[0]):
        j = np.where(bins - image_intensities[i] > 0)[0][0]
        histogram[j] += 1

    pdfY = histogram.copy()
    sum = np.sum(pdfY)
    pdfY /= sum

    plt.plot(bins, pdfY)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Image Pixel Intensity')
    plt.ylabel('Probability')
    plt.show()

    pdfXY = compute_joint_pdf(lidar_intensities, image_intensities)

    # mutual_information = compute_mutual_information(pdfX, pdfY)

    # img_canny = cv2.Canny(image, 100, 200)
    # plt.subplot(121), plt.imshow(image,cmap = 'gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    # plt.subplot(122), plt.imshow(img_canny,cmap = 'gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()

    # compute the gradient of the depth image

    print("...end")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", help="Path to data file", required=False, type=str, default=os.path.join(pathlib.Path(__file__).parent.absolute(), "../costar.h5"))
    args = parser.parse_args()

    main(args)
