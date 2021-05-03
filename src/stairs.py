import argparse
import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator, griddata
from scipy.spatial import Delaunay
import h5py
import os
import pathlib
import math
from datetime import datetime
import open3d as o3d
from scipy.optimize import NonlinearConstraint, minimize
import sklearn
from sklearn import cluster

# Need to keep track of the frame transformations as the robot moves in order to have the point cloud represented in the same frame
COSTAR_DATA_DIRPATH = os.path.join(os.path.dirname(__file__), '..', 'costar.h5')
DATA_INDEX = 23
NUM_DIFF_VECTORS = 10000

def plot_point_cloud(point_clouds):

    # Plot the point cloud
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_clouds)

    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100.0,
        max_nn=50))

    o3d.visualization.draw_geometries([cloud])

def get_normalized_diff_vectors(lidar, d1=0.02, d2=0.07, n_clusters=2, debug=False):

    neighbors = dict()
    for i, point in enumerate(lidar):
        dist = np.linalg.norm(np.tile(point, (lidar.shape[0], 1)) - lidar, axis=1)
        neighbors[i] = lidar[(dist <= d2) * (dist >= d1), :]
        if neighbors[i].shape[0] == 0:
            neighbors.pop(i, None)

    norm_diff_vec = np.zeros((NUM_DIFF_VECTORS, 3))
    lidar_idx = [None] * NUM_DIFF_VECTORS
    subsampled_lidar = np.zeros((NUM_DIFF_VECTORS, 3))
    keys = list(neighbors.keys())
    k = 0
    while k < NUM_DIFF_VECTORS:
        m = np.random.randint(0, len(keys))
        i = keys[m]

        pi = lidar[i, :]
        subsampled_lidar[k, :] = lidar[i, :]
        lidar_idx[k] = i

        j = np.random.randint(0, neighbors[i].shape[0])
        pj = neighbors[i][j, :]
        norm_diff_vec[k, :] = (pi - pj) / np.linalg.norm(pi - pj)
        k += 1

    # cluster the normalized difference vectors
    # kmeans = cluster.KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=500, tol=0.0001, verbose=1).fit(norm_diff_vec)
    kmeans = cluster.SpectralClustering(n_clusters=n_clusters, n_init=10, verbose=1).fit(norm_diff_vec)

    if debug:
        # Plot the clustered normalized difference vectors
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        colors = {0 : 'r', 1 : 'g', 2 : 'b', 3 : 'm'}
        for label in range(np.max(kmeans.labels_)+1):
            ax.scatter(norm_diff_vec[kmeans.labels_==label,0], norm_diff_vec[kmeans.labels_==label,1], norm_diff_vec[kmeans.labels_==label,2], c=colors[label])

        ax.set_xlabel('x')
        ax.set_xlabel('y')
        ax.set_xlabel('z')

        plt.show()

        # Plot the point cloud
        plot_point_cloud(subsampled_lidar)

    return norm_diff_vec, lidar_idx, neighbors, kmeans.labels_ # cluster labels

def cons_f(n):

    return [n[0]*n[0] + n[1]*n[1] + n[2]*n[2]]

def cons_J(n):

    return [2*n[0], 2*n[1], 2*n[2]]

def cons_H(n, v):

    return v[0] * np.array([2, 2, 2])

def objective(n, V):

    return np.sum(np.square(np.matmul(V, n[:, np.newaxis])))

def ransac(lidar, normalized_diff_vectors, lidar_idx, cluster, num_iterations=20, num_points=2, min_inlier_count=100, tol=1e-6, d1=0.02, d2=0.07, debug=True):
    '''
    Inputs:
        cloud - Nx3 point cloud
        num_planes - number of planes to fit
        debug - flag to plot the intermediate results

    Outputs:
        centers - center points of the planes
        normals - normal vectors of the planes
        dims - height and width of the plane
    '''

    # Set up the optimization problem
    constraint = NonlinearConstraint(cons_f, 1., 1., jac=cons_J, hess=cons_H)

    # Apply RANSAC to estimate the main directions of the point cloud
    models = [None] * (np.max(cluster)+1)
    for label, _ in enumerate(models):
        cluster_subset_idx = cluster==label
        normalized_diff_vectors_subset = normalized_diff_vectors[cluster_subset_idx]
        lidar_subset_idx = np.array(lidar_idx)[cluster_subset_idx]
        lidar_subset = lidar[lidar_idx][cluster_subset_idx]
        i = 0
        best_inliers = 0
        while i < num_iterations:
            # Model from num_points samples
            idx = np.random.choice(len(lidar_subset_idx), num_points, replace=False)
            normalized_diff_vectors_sample = normalized_diff_vectors_subset[idx, :]
            subsamp_lidar_idx = lidar_subset_idx[idx]

            # Solve as a linear constrained problem:
            #   minimize_n [v1; v2] [n]
            #   s.t.       [n]^T [n] = 1
            res = minimize(
                objective,
                x0 = np.array([0., -1., 0.]),
                args = (normalized_diff_vectors_sample,),
                constraints = constraint,
                method = 'trust-constr'
            )
            n = res.x

            errors = np.square(np.matmul(normalized_diff_vectors_subset, n[:, np.newaxis]))
            is_inlier = np.squeeze(errors < tol)

            if np.count_nonzero(is_inlier) > 0 and np.where(is_inlier)[0].shape[0] > 1:
                # Reject thin and long plane segments with lamda3 >> lamda2
                cov = np.cov(lidar_subset[is_inlier, :].transpose())
                eigvals, _ = np.linalg.eig(cov)

                # if number of inliers exceeds threshold, add it to the list of candidate models
                if np.count_nonzero(is_inlier) > best_inliers and np.count_nonzero(is_inlier) >= min_inlier_count and abs(eigvals[2]) - abs(eigvals[1]) < 0.5:
                    
                    # push new model to back, pop one off the front
                    # make sure that models are sorted from least to greatest
                    model = dict()
                    model["inliers"] = np.unique(
                        np.hstack((lidar_subset_idx[is_inlier], subsamp_lidar_idx)))
                    model["normal"] = n
                    model["centroid"] = np.sum(lidar[model["inliers"]], axis=0) / model["inliers"].shape[0]

                    models[label] = model

                    best_inliers = np.where(is_inlier)[0].shape[0]

            print("num inliers: {} num iterations {}".format(best_inliers, i))
            i += 1

    # TODO: Merge neighboring planes
    # - Check the angle and the distance between every two sets of planes
    # angles = []
    # distances = []
    # for i in range(num_planes):
    #     plane1 = models[i]
    #     for j in range(num_planes):
    #         plane2 = models[j]

    return models # TODO: Need to double check that the inliers are also neighbors

def plot_pc_over_image(lidar, image, T_velo2cam, K, plane=None):

    # Filter out lidar points outside field of view

    # The rear camera has a non-standard coordinate system:
    #
    #             ^ xc        / z
    #             |          /
    #             |_____>   /_____>
    #            /     zc   |     x
    #           /           |
    #          yc           y
    T_camera_2_img = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    T_camera_2_img = np.vstack((T_camera_2_img, np.zeros((1,3))))
    T_camera_2_img = np.hstack((T_camera_2_img, np.zeros((4,1))))
    T_camera_2_img[-1, -1] = 1.
    T_camera_2_img = T_camera_2_img.T

    T_velo2cam = np.matmul(T_camera_2_img, T_velo2cam)

    pc_cam = np.matmul(np.hstack((lidar[:,:3], np.ones((lidar.shape[0], 1)))), T_velo2cam.T)
    pc_cam_pos_z_idx = np.where((pc_cam[:, 2] > 0))[0]
    pc_cam = pc_cam[pc_cam_pos_z_idx, :]
    
    pc_img = pc_cam.copy()
    pc_img = np.matmul(pc_img[:, :3], K.T) # (N x 3) x (3 x 3) 
    pc_img = pc_img / pc_img[:, 2, None]
    
    width_mask = ((pc_img[:,0] >= 0) & (pc_img[:,0] <= image.shape[1]))
    height_mask = ((pc_img[:,1] >= 0) & (pc_img[:,1] <= image.shape[0]))
    pc_img_mask_idx = np.where(width_mask & height_mask)[0]
    pc_img = pc_img[pc_img_mask_idx, :]
    
    lidar = lidar[pc_cam_pos_z_idx, :]
    lidar = lidar[pc_img_mask_idx, :]

    # Plot the lidar over the image
    rgb = plt.cm.get_cmap('jet')
    max_dist = 3.
    for img_point, velo_point in zip(pc_img, lidar):
        color = rgb(np.clip(np.linalg.norm(velo_point), 0., max_dist) / max_dist, bytes=True)
        color = list(color)
        color = color[:-1]

        cv2.circle(img=image,
        center=(np.int_(img_point[0]), np.int_(img_point[1])),
        radius=1,
        color=np.float_(color),
        thickness=-1)

    if plane is not None:
        center = plane['centroid']
        normal = plane['normal']
        points = np.vstack((center + normal, center))
        points_pc = points.copy()

        points = np.matmul(np.hstack((points, np.ones((2,1)))), T_velo2cam.T)
        points = np.matmul(points[:, :3], K.T) # (N x 3) x (3 x 3) 
        points = points / points[:, 2, None]

        color = rgb(1.0, bytes=True)
        color = list(color)
        color = color[:-1]
        
        for point in points:
            cv2.circle(img=image,
                center=(np.int_(point[0]), np.int_(point[1])),
                radius=5,
                color=np.float_(color),
                thickness=-1)

        plt.plot(points[:, 0], points[:, 1])
        
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.close()

    if plane is not None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(lidar[:,0], lidar[:,1], lidar[:,2])
        ax.plot3D(points_pc[:,0], points_pc[:,1], points_pc[:,2], 'red')

        ax.set_xlabel('x')
        ax.set_xlabel('y')
        ax.set_xlabel('z')

        plt.show()
        plt.close()

    plot_point_cloud(lidar)

    return lidar

def main(debug=True):

    # Load the data
    with h5py.File(COSTAR_DATA_DIRPATH, 'r') as hf:
        h, w = hf['image']['resolution'][0], hf['image']['resolution'][1]
        image = np.reshape(hf['image']['image'][DATA_INDEX, :, :], (h, w, 3))
        lidar = hf['lidar']['lidar'][DATA_INDEX, :]
        lidar = lidar[~np.isnan(lidar)]
        lidar = np.reshape(lidar, (-1, 4))
        lidar = lidar[:, :3]
        T_velo2cam = np.reshape(hf['lidar']['tf_velo2cam'][:], (4,4))
        K = np.reshape(hf['image']['intrinsics'], (3,3))

    lidar = plot_pc_over_image(lidar, image, T_velo2cam, K)

    # Get normalized difference vectors
    normalized_diff_vectors, lidar_idx, neighbors, clusters = get_normalized_diff_vectors(lidar, debug=True)

    # Apply RANSAC to get plane normals
    # Get the plane origins, normals, and dimensions (convex hull)
    planes = ransac(lidar, normalized_diff_vectors, lidar_idx, clusters)
    for plane in planes:
        cloud = lidar[plane['inliers'], :]
        _ = plot_pc_over_image(cloud, image, T_velo2cam, K, plane=plane)

if __name__ == '__main__':

    main()