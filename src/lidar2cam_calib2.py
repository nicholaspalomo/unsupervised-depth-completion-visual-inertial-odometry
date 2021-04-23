import argparse
import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
from scipy.spatial import Delaunay
import h5py
import os
import pathlib
import math
from datetime import datetime

import sys
sys.path.append('.')

# Apply gaussian blurring of depth image (lidar) and actual image
# Compute the local gradient for each pixel (maximum magnitude in neighborhood and corresponding direction)
# Do this for the RGB image as well as for the interpolated depth image
# normalize the depth intensities by the maximum depth present in the camera field of view

def create_depth_with_validity_map(pc, h, w, K, T, debug=False):
    '''
    converts the point cloud to a sparse depth map

    Args:
        pc : numpy
        list of points in point cloud

    Returns:
        numpy : depth map
        numpy : binary validity map for available depth measurement locations
    '''

    # filter out the points which lie outside the camera FOV
    pc_ = pc.copy()

    pc_ = np.matmul(
        np.hstack((pc_, np.ones((pc_.shape[0], 1)))),
        T.T
    )

    pc_cam_pos_z_idx = np.where((pc_[:, 2] > 0))[0]
    pc_ = pc_[pc_cam_pos_z_idx, :] # points in front of the image plane
    pc_img = np.matmul(pc_[:,:3], K.T) # point cloud given in the image frame
    pc_img = pc_img / pc_img[:, -1, None]
    
    width_mask = ((pc_img[:,0] >= 0) & (pc_img[:,0] <= w))
    height_mask = ((pc_img[:,1] >= 0) & (pc_img[:,1] <= h))
    pc_img_mask_idx = np.where(width_mask & height_mask)[0]

    pc_ = pc_[pc_img_mask_idx, :]

    pc_img = np.matmul(pc_[:,:3], K.T)
    pc_img = pc_img / pc_img[:, -1, None]
    pc_intensity_16bit = np.linalg.norm(pc_[:,:3], axis=1)
    pc_intensity_16bit /= np.max(pc_intensity_16bit)

    # apply Gaussian filter to z?
    z = np.zeros((h, w), dtype=np.float32)
    for i, j in enumerate(pc_img):
        u = min(np.int_(j[1]), h-1)
        v = min(np.int_(j[0]), w-1)
        z[u, v] = pc_intensity_16bit[i]
    z = cv2.blur(z, (1,1))
    v = z.astype(np.float32)
    v[z > 0] = 1.0

    if debug:
        cv2.imshow('depth image', z)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
    
    return z, v

def interpolate_depth(depth_map, validity_map, log_space=False):

    assert depth_map.ndim == 2 and validity_map.ndim == 2
    rows, cols = depth_map.shape
    data_row_idx, data_col_idx = np.where(validity_map)
    depth_values = depth_map[data_row_idx, data_col_idx]
    # Perform linear interpolation in log space
    if log_space:
        depth_values = np.log(depth_values)
    interpolator = CloughTocher2DInterpolator(
        points=Delaunay(np.stack([data_row_idx, data_col_idx], axis=1).astype(np.float32)),
        # points=np.stack([data_row_idx, data_col_idx], axis=1),
        values=depth_values,
        fill_value=0 if not log_space else np.log(1e-3))
    query_row_idx, query_col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    query_coord = np.stack([query_row_idx.ravel(), query_col_idx.ravel()], axis=1)
    Z = interpolator(query_coord).reshape([rows, cols])
    if log_space:
        Z = np.exp(Z)
        Z[Z < 1e-1] = 0.0

    return Z

def compute_gradients(validity_map, depth_map, neighborhood=3, thresh=0.2, apply_vm=False):

    ddepth = cv2.CV_16S
    scale = 1
    delta = 0

    grad_x = cv2.Sobel(depth_map * 255, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    grad_y = cv2.Sobel(depth_map * 255, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    # maximum gradient and corresponding direction in neighborhood of pixel under mask

    grad_mag = np.sqrt(grad_x.astype(np.float)**2 + grad_y.astype(np.float)**2)

    grad_mag = grad_mag / np.max(grad_mag)
    # grad_mag[grad_mag < thresh] = 0.

    # strengthen edges
    direction = np.zeros(grad_mag.shape).astype(np.int)
    angles = np.zeros(grad_mag.shape)
    switch_direction = {
        0: [1, 0], # u, v
        1: [1, 1],
        2: [0, 1],
        3: [-1, 1],
        4: [-1, 0],
        5: [-1, -1],
        6: [0, -1],
        7: [1, -1],
        8: [1, 0]
    }
    for u in range(grad_mag.shape[0]):
        for v in range(grad_mag.shape[1]):
            condition = True
            u_next, v_next = u, v
            while condition:
                if u_next == grad_mag.shape[0] or v_next == grad_mag.shape[1]:
                    condition = False
                else:
                    direction[u_next, v_next] = int((math.atan2(grad_y[u_next, v_next], grad_x[u_next, v_next]) + np.pi) // (2 * np.pi / 8))
                    angles[u_next, v_next] = np.float(direction[u_next, v_next] * (2 * np.pi / 8))
                    
                    if grad_mag[u_next, v_next] >= thresh and not grad_mag[u_next, v_next] == 1.:
                        grad_mag[u_next, v_next] = 1.
                        dudv = switch_direction[direction[u_next, v_next]]
                        u_next += dudv[0]
                        v_next += dudv[1]
                    else:
                        grad_mag[u_next, v_next] = 0.
                        condition = False

    if apply_vm:
        grad_mag = np.multiply(grad_mag, validity_map)
        angles = np.multiply(angles, validity_map)

    return grad_x, grad_y, grad_mag / np.max(grad_mag) * 255, angles

def get_descriptors(depth_grad_mag, image_grad_mag, depth_dir, image_dir, depth_map=None, image=None, num_descriptors=10, neighborhood=5):

    descriptor_votes = np.zeros((image_grad_mag.shape[0], image_grad_mag.shape[1], 2)) # votes from image_grad_mag, votes from depth_grad_mag map
    for u in range(neighborhood, image_grad_mag.shape[0] - neighborhood):
        for v in range(neighborhood, image_grad_mag.shape[1] - neighborhood):

            descriptor_votes[u, v, 0] = np.sum(image_grad_mag[u-neighborhood:u+neighborhood, v-neighborhood:v+neighborhood] / 255.)

            descriptor_votes[u, v, 1] = np.sum(depth_grad_mag[u-neighborhood:u+neighborhood, v-neighborhood:v+neighborhood] / 255.)

    # sort the number of descriptor votes first by lidar (since this is sparse), then by votes for image_grad_mag descriptors
    depth_votes = np.dstack(np.unravel_index(np.argsort(descriptor_votes[:,:,1].ravel()), (image_grad_mag.shape[0], image_grad_mag.shape[1])))
    depth_votes = depth_votes[:, -num_descriptors:, :]

    descriptors_depth = []
    descriptors_image = []
    for i in range(depth_votes.shape[1]):
        u = depth_votes[0, i, 0]
        v = depth_votes[0, i, 1]
        descriptors_depth.append(depth_grad_mag[u-neighborhood:u+neighborhood, v-neighborhood:v+neighborhood])
        descriptors_image.append(image_grad_mag[u-neighborhood:u+neighborhood, v-neighborhood:v+neighborhood])

        if depth_map is not None and image is not None:
            ax = plt.subplot2grid((4, 10), (0, i))
            plt.imshow(depth_map[u-neighborhood:u+neighborhood, v-neighborhood:v+neighborhood], cmap='gray')
            plt.xticks([]), plt.yticks([])

            ax = plt.subplot2grid((4, 10), (1, i))
            plt.imshow(depth_grad_mag[u-neighborhood:u+neighborhood, v-neighborhood:v+neighborhood], cmap='gray')
            plt.xticks([]), plt.yticks([])

            ax = plt.subplot2grid((4, 10), (2, i))
            plt.imshow(image[u-neighborhood:u+neighborhood, v-neighborhood:v+neighborhood], cmap='gray')
            plt.xticks([]), plt.yticks([])

            ax = plt.subplot2grid((4, 10), (3, i))
            plt.imshow(image_grad_mag[u-neighborhood:u+neighborhood, v-neighborhood:v+neighborhood], cmap='gray')
            plt.xticks([]), plt.yticks([])

    if depth_map is not None and image is not None:
        plt.gcf().savefig(os.path.dirname(os.path.abspath(__file__)) + datetime.now().strftime("/%d%m%Y%H%M%S"), dpi=300)
        # plt.show(block=False)
        # plt.pause(3)
        plt.close()

    return descriptors_depth, descriptors_image

def rpy2rotmat(gamma, beta, alpha):

    R = np.zeros((3,3))
    R[0, 0] = math.cos(alpha) * math.cos(beta)
    R[0, 1] = math.cos(alpha) * math.sin(beta) * math.sin(gamma) - math.sin(alpha) * math.cos(gamma)
    R[0, 2] = math.cos(alpha) * math.sin(beta) * math.cos(gamma) + math.sin(alpha) * math.sin(gamma)
    R[1, 0] = math.sin(alpha) * math.cos(beta)
    R[1, 1] = math.sin(alpha) * math.sin(beta) * math.sin(gamma) + math.cos(alpha) * math.cos(gamma)
    R[1, 2] = math.sin(alpha) * math.sin(beta) * math.cos(gamma) - math.cos(alpha) * math.sin(gamma)
    R[2, 0] = -math.sin(beta)
    R[2, 1] = math.cos(beta) * math.sin(gamma)
    R[2, 2] = math.cos(beta) * math.cos(gamma)

    return R

def rotmat2rpy(R):

    return math.atan2(R[2,1], R[2,2]), math.atan2(-R[2,0], math.sqrt(R[2,1]**2 + R[2,2]**2)), math.atan2(R[1,0], R[0,0])

def depth_color(val, min_d=0, max_d=120):
    """
    print Color(HSV's H value) corresponding to distance(m)
    close distance = red , far distance = blue
    """
    val = np.clip(val, min_d, max_d)  # max distance is 120m but usually not usual
    return (val - min_d) / (max_d - min_d)

def visualize_lidar_projection(lidar, image, K, T):

    pc_cam = np.matmul(np.hstack((lidar[:,:3], np.ones((lidar.shape[0], 1)))), T.T)
    pc_cam_pos_z_idx = np.where((pc_cam[:, 2] > 0))[0]
    pc_cam = pc_cam[pc_cam_pos_z_idx, :]
    
    pc_img = pc_cam.copy()
    pc_img = np.matmul(pc_img[:, :3], K.T) # (N x 3) x (3 x 3) 
    pc_img = pc_img / pc_img[:, 2, None]
    
    width_mask = ((pc_img[:,0] >= 0) & (pc_img[:,0] <= image.shape[1]))
    height_mask = ((pc_img[:,1] >= 0) & (pc_img[:,1] <= image.shape[0]))
    pc_img_mask_idx = np.where(width_mask & height_mask)[0]
    pc_img = pc_img[pc_img_mask_idx, :]
    
    pc_velo = lidar[pc_cam_pos_z_idx, :]

    intensities = depth_color(np.linalg.norm(pc_velo[:, :3], axis=1), min_d=0, max_d=10.)

    rgb = plt.cm.get_cmap('jet')
    colors = []
    for img_point, velo_point, intensity in zip(pc_img, pc_velo[pc_img_mask_idx, :], intensities[pc_img_mask_idx]):

        color = rgb(intensity, bytes=True)
        color = list(color)
        color = color[:-1]
        colors.append(color[::-1])
        
        cv2.circle(img=image,
            center=(np.int_(img_point[0]), np.int_(img_point[1])),
            radius=1,
            color=np.float_(color),
            thickness=-1)
        
    plt.imshow(image)
    plt.gcf().savefig(os.path.dirname(os.path.abspath(__file__)) + datetime.now().strftime("/%d%m%Y%H%M%S"), dpi=300)
    plt.close()

def main(args):
    debug = True
    INDEX = 0

    with h5py.File(args.data_file, 'r') as h5:
        lidar = h5['lidar']['lidar'][INDEX, :]
        image = h5['image']['image'][INDEX, :, :]
        resolution = h5['image']['resolution'][:]
        tf_velo2cam = h5['lidar']['tf_velo2cam'][:]
        K = h5['image']['intrinsics'][:]

    # compute Canny edge detection
    h = resolution[0]
    w = resolution[1]
    image = np.reshape(image, (h, -1, 3))
    lidar = np.reshape(lidar[~np.isnan(lidar)], (-1, 4))
    reflectivity = lidar[:, -1]
    lidar = lidar[:, :3]
    tf_velo2cam = np.reshape(tf_velo2cam, (4, 4))
    K = np.reshape(K, (3, 3))

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    depth_map, validity_map = create_depth_with_validity_map(lidar, h, w, K, tf_velo2cam, debug=False)
    depth_map /= np.max(depth_map)

    gray = gray.astype(float) / np.max(gray.astype(float))
    gray_blur = cv2.blur(gray, (5,5))
    gray_blur = gray_blur.astype(float) / np.max(gray_blur.astype(float))
    depth_map_blur = cv2.blur(depth_map, (5,5))
    depth_map_blur /= np.max(depth_map_blur)

    if debug:
        plt.subplot(121),plt.imshow(gray, cmap='gray'),plt.title('Grayscale')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(depth_map),plt.title('Depth Map')
        plt.xticks([]), plt.yticks([])
        plt.show()

        plt.subplot(121),plt.imshow(gray_blur, cmap='gray'),plt.title('Grayscale')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(depth_map_blur),plt.title('Depth Map')
        plt.xticks([]), plt.yticks([])
        plt.show()

    depth_map_interp = interpolate_depth(depth_map, validity_map, log_space=False)
    # depth_map_interp = cv2.blur(depth_map_interp, (5,5))
    depth_map_interp /= np.max(depth_map_interp)

    if debug:
        plt.subplot(121),plt.imshow(gray_blur, cmap='gray'),plt.title('Grayscale')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(depth_map_interp),plt.title('Depth Map')
        plt.xticks([]), plt.yticks([])
        plt.show()

    # compute depth map gradients
    grad_x_depth, grad_y_depth, grad_mag_depth, dir_depth = compute_gradients(validity_map, depth_map_interp, neighborhood=3, apply_vm=True, thresh=0.1)

    if debug:
        plt.subplot(131),plt.imshow(grad_x_depth, cmap='gray'),plt.title('grad_x')
        plt.xticks([]), plt.yticks([])
        plt.subplot(132),plt.imshow(grad_y_depth),plt.title('grad_y')
        plt.xticks([]), plt.yticks([])
        plt.subplot(133),plt.imshow(grad_mag_depth),plt.title('grad_mag')
        plt.xticks([]), plt.yticks([])
        plt.show()

    grad_x_image, grad_y_image, grad_mag_image, dir_image = compute_gradients(validity_map, gray, neighborhood=3, thresh=0.4)

    if debug:
        plt.subplot(231),plt.imshow(grad_x_depth, cmap='gray'),plt.title('grad_x')
        plt.xticks([]), plt.yticks([])
        plt.subplot(232),plt.imshow(grad_y_depth),plt.title('grad_y')
        plt.xticks([]), plt.yticks([])
        plt.subplot(233),plt.imshow(grad_mag_depth),plt.title('grad_mag')
        plt.xticks([]), plt.yticks([])
        plt.subplot(234),plt.imshow(grad_x_image, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.subplot(235),plt.imshow(grad_y_image)
        plt.xticks([]), plt.yticks([])
        plt.subplot(236),plt.imshow(grad_mag_image)
        plt.xticks([]), plt.yticks([])
        plt.show()

        plt.subplot(121),plt.imshow(dir_depth),plt.title('depth_dir')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(dir_image),plt.title('image_dir')
        plt.xticks([]), plt.yticks([])
        plt.show()

    # Get descriptors - find the num_descriptors number of regions with the greatest number of activated pixels between the depth image and the RGB image
    descriptors_depth, descriptors_image = get_descriptors(grad_mag_depth, grad_mag_image, dir_depth, dir_image, depth_map=depth_map, image=gray, num_descriptors=10, neighborhood=25)

    # compute the relative warp between the camera and the lidar
    n = 3
    rot = np.concatenate((np.pi * 0.05 * np.linspace(0., 1.0, num=n+1), np.zeros(1)))
    tran = np.concatenate((0.1 * np.linspace(-1.0, 0., num=n+1), np.zeros(1)))

    r0, p0, yaw0 = rotmat2rpy(tf_velo2cam[:3, :3])
    r, yaw = 0, 0
    x, y = 0, 0

    params = []
    # for x in tran:
    #     for y in tran:
    #         for z in tran:
                # for r in rot:
                #     for p in rot:
                #         for yaw in rot:
    for z in tran:
        for p in rot:

            R_cam2velo = rpy2rotmat(0, p, 0.)
            r, p, yaw = rotmat2rpy(R_cam2velo.T)

            rN, pN, yawN = r0 + r, p0 + p, yaw0 + yaw
            xN, yN, zN = tf_velo2cam[0,-1] + x, tf_velo2cam[1,-1] + y, tf_velo2cam[2,-1] + z

            T = np.zeros((4,4))
            T[-1,-1] = 1.

            T[:3, :3] = rpy2rotmat(rN, pN, yawN)
            T[:3, -1] = np.array([xN, yN, zN])

            depth_map, validity_map = create_depth_with_validity_map(lidar, h, w, K, T, debug=False)

            depth_map /= np.max(depth_map)

            depth_map_interp = interpolate_depth(depth_map, validity_map, log_space=False)

            depth_map_interp /= np.max(depth_map_interp)

            grad_x_depth, grad_y_depth, grad_mag_depth, dir_depth = compute_gradients(validity_map, depth_map_interp, neighborhood=3, apply_vm=True, thresh=0.2)

            descriptors_depth, descriptors_image = get_descriptors(grad_mag_depth, grad_mag_image, dir_depth, dir_image, depth_map=depth_map, image=gray, num_descriptors=10, neighborhood=25)

            mse = 0
            for descriptor_depth, descriptor_image in zip(descriptors_depth, descriptors_image):
                mse += np.sum((descriptor_depth - descriptor_image)**2)

            result = np.array([rN, pN, yawN, xN, yN, zN, mse**0.5])
            params.append(result)

            visualize_lidar_projection(lidar.copy(), image.copy(), K, T)

            print("r, p, y: {}, {}, {}. x, y, z: {}, {}, {}. MSE: {}".format(rN, pN, yawN, xN, yN, zN, mse**0.5))

    params = np.array(params)
    min_mse_idx = np.argmin(params[:, -1], axis=0)
    r, p, yaw, x, y, z = list(params[min_mse_idx, :-1])
    T = np.zeros((4,4))
    T[-1,-1] = 1.
    T[:3, :3] = rpy2rotmat(r, p, yaw)
    T[:3, -1] = np.array([x, y, z])

    print("min MSE: {}".format(params[min_mse_idx, -1]))
    visualize_lidar_projection(lidar, image, K, T)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", help="Path to data file", required=False, type=str, default=os.path.join(pathlib.Path(__file__).parent.absolute(), "../costar.h5"))
    args = parser.parse_args()

    main(args)
