import argparse
import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import centroid
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
DATA_INDEX = 23 # 21, 23, 27, 86, 127
NUM_DIFF_VECTORS = 10000

# IDEA: Filter out points within X distance of the camera, as these may correspond to the legs

def plot_point_cloud(point_clouds):

    # Plot the point cloud
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_clouds)

    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100.0,
        max_nn=50))

    o3d.visualization.draw_geometries([cloud])

def get_normalized_diff_vectors(lidar, d1=0.02, d2=0.03, n_clusters=2, debug=False):

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

def ransac(lidar, normalized_diff_vectors, lidar_idx, cluster, num_iterations=40, num_points=10, min_inlier_count=10, tol=1e-3, d1=0.02, d2=0.07, debug=True):
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
                # cov = np.cov(lidar_subset[is_inlier, :].transpose())
                # eigvals, _ = np.linalg.eig(cov)

                # if number of inliers exceeds threshold, add it to the list of candidate models
                if np.count_nonzero(is_inlier) > best_inliers and np.count_nonzero(is_inlier) >= min_inlier_count: # and abs(eigvals[2])/(abs(eigvals[1])+0.001) < 10.:
                    
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

    return models # TODO: Need to double check that the inliers are also neighbors

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def rotmat_a2b(a, b):

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)

    R = np.eye(3) + skew(v) * s + skew(v)**2 * (1 - c)

    U, _, V_transpose = np.linalg.svd(R, full_matrices=True)

    return np.matmul(U, V_transpose)

def merge_planes(lidar, planes, ratio=2., min_cluster_size=40, max_dist=0.05, max_angle=30, max_offset=5., debug=True):
    # max_angle given in degrees!

    segmented_cloud = []
    segmented_cloud_centers_normals = []
    for n in planes:
        cloud = lidar[n['inliers'], :]

        # Cut the point cloud into thin slices perpendicular to n, the principal directions of the point cloud
        bins = np.linspace(-max_offset, max_offset, num=250)
        normal = n['normal']
        R_cloud2normal = rotmat_a2b(normal, np.array([0,0,1]))
        T_cloud2normal = np.eye(4)
        T_cloud2normal[:3, :3] = R_cloud2normal

        for i, bin_i in enumerate(bins):
            if i < bins.shape[0]-1:

                # Check if points above plane
                T_cloud2normal[:3, -1] = n['centroid']
                # cloud_frame1 = np.matmul(np.hstack((cloud, np.ones((cloud.shape[0], 1)))), T_cloud2normal.T) # cloud in frame of plane1
                cloud_frame1 = np.matmul(T_cloud2normal[:3,:3], cloud.T).T

                cloud_frame1[:, 2] -= bin_i
                points_above_plane1 = np.dot(cloud_frame1[:, :3], np.array([0, 0, 1])) > 0
                cloud_frame1[:, 2] += bin_i

                # Check if points below plane
                cloud_frame1[:, 2] -= bins[i+1]
                points_below_plane2 = np.dot(cloud_frame1[:, :3], np.array([0, 0, 1])) < 0

                # Get the points in the slice
                points_in_slice = cloud[points_above_plane1 & points_below_plane2, :]

                if points_in_slice.shape[0] >= min_cluster_size:

                    if debug:
                        fig = plt.figure()
                        plt.cla()
                        plt.clf()
                        ax = fig.add_subplot(projection='3d')

                        # plot plane 1
                        # a plane is a*x+b*y+c*z+d=0
                        # [a,b,c] is the normal. Thus, we have to calculate
                        # d and we're set
                        xx, yy = np.meshgrid(np.linspace(-1, 1, num=51), np.linspace(-1, 1, num=51))
                        grid_normal = np.hstack((
                            np.reshape(xx, (-1,1)),
                            np.reshape(yy, (-1,1)),
                            bin_i * np.ones((xx.shape[0]*xx.shape[1], 1))
                        ))
                        grid_cloud = np.matmul(T_cloud2normal[:3,:3].T, grid_normal.T).T

                        xx1 = np.reshape(grid_cloud[:, 0], xx.shape)
                        yy1 = np.reshape(grid_cloud[:, 1], xx.shape)
                        z1 = np.reshape(grid_cloud[:, 2], xx.shape)

                        ax.plot_surface(xx1, yy1, z1, alpha=0.2, color='g')

                        # plot plane 2
                        grid_normal = np.hstack((
                            np.reshape(xx, (-1,1)),
                            np.reshape(yy, (-1,1)),
                            bins[i+1] * np.ones((xx.shape[0]*xx.shape[1], 1))
                        ))
                        grid_cloud = np.matmul(T_cloud2normal[:3,:3].T, grid_normal.T).T

                        xx2 = np.reshape(grid_cloud[:, 0], xx.shape)
                        yy2 = np.reshape(grid_cloud[:, 1], yy.shape)
                        z2 = np.reshape(grid_cloud[:, 2], xx.shape)

                        ax.plot_surface(xx2, yy2, z2, alpha=0.2, color='b')

                        # plot normal vector
                        points_in_slice = cloud[points_above_plane1 & points_below_plane2, :]
                        points_outside_slice = cloud[~(points_above_plane1 & points_below_plane2), :]

                        ax.scatter(points_in_slice[:,0], points_in_slice[:,1], points_in_slice[:,2], color='red')
                        ax.scatter(points_outside_slice[:,0], points_outside_slice[:,1], points_outside_slice[:,2], color='blue')
                        normal_vec = np.vstack((n['centroid'], n['centroid'] + normal))
                        ax.plot3D(normal_vec[:,0], normal_vec[:,1], normal_vec[:,2], 'red')

                        ax.set_xlabel('x')
                        ax.set_xlabel('y')
                        ax.set_xlabel('z')

                        plt.show(block=False)
                        plt.pause(1.5)
                        plt.close(fig)

                    # Get the number of clusters based on the minimum number of points in a plane
                    n_clusters = points_in_slice.shape[0] // min_cluster_size

                    # Cluster the points in the slice
                    kmeans = cluster.KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=500, tol=0.0001, verbose=1).fit(points_in_slice)

                    for j, centroid in enumerate(kmeans.cluster_centers_):
                        # Reject the plane segments that do not satisfy eigenvalue criterion
                        candidates = points_in_slice[kmeans.labels_ == j, :]
                        if candidates.shape[0] > 1:
                            cov = np.cov(candidates.transpose())
                            eigvals, _ = np.linalg.eig(cov)

                            if abs(eigvals[2])/(abs(eigvals[1])+0.001) < ratio:
                                # point_cloud = o3d.geometry.PointCloud()
                                # point_cloud.points = o3d.utility.Vector3dVector(candidates)

                                # point_cloud.estimate_normals(
                                #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100.0,
                                #     max_nn=100))

                                # avg_normal = np.mean(np.asarray(point_cloud.normals), axis=0)
                                # avg_normal /= np.linalg.norm(avg_normal)

                                segmented_cloud.append(candidates)

                                segmented_cloud_centers_normals.append(
                                    np.array([centroid[0], centroid[1], centroid[2], normal[0], normal[1], normal[2]])
                                )

    # Merge neighboring planes based on heuristic
    segmented_cloud_centers_normals = np.array(segmented_cloud_centers_normals)
    clouds = []
    merged = [False] * len(segmented_cloud)
    for i, cloud in enumerate(segmented_cloud):
        dist_check = np.where(abs(np.linalg.norm(segmented_cloud_centers_normals[i, :3]) - np.linalg.norm(segmented_cloud_centers_normals[:, :3], axis=1)) < max_dist)[0]

        angle = (np.arccos(np.dot(segmented_cloud_centers_normals[:, 3:], segmented_cloud_centers_normals[i, 3:])) + np.pi) % (2 * np.pi) - np.pi
        angle_check = np.where(
            abs(angle) * 180 / np.pi < max_angle)[0]
        angle_check = np.hstack((np.array([i]), angle_check))

        check = np.intersect1d(angle_check, dist_check)

        if check.shape[0] > 0:
            clouds.append({'normal' : np.array([0., 0., 0.]), 'centroid': np.array([0., 0., 0.]), 'points' : np.empty((0, 3))})
            for pc_idx in np.unique(check):
                if not merged[pc_idx]:
                    clouds[len(clouds)-1]['points'] = np.vstack((clouds[len(clouds)-1]['points'], segmented_cloud[pc_idx]))
                    clouds[len(clouds)-1]['centroid'] += segmented_cloud_centers_normals[pc_idx, :3]
                    clouds[len(clouds)-1]['normal'] += segmented_cloud_centers_normals[pc_idx, 3:]
                    merged[pc_idx] = True
            clouds[len(clouds)-1]['centroid'] /= np.unique(check).shape[0]
            clouds[len(clouds)-1]['normal'] /= np.unique(check).shape[0]
            clouds[len(clouds)-1]['normal'] /= np.linalg.norm(clouds[len(clouds)-1]['normal'])

    return clouds

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
    normalized_diff_vectors, lidar_idx, neighbors, clusters = get_normalized_diff_vectors(lidar, debug=False)

    # Apply RANSAC to get plane normals
    # Get the plane origins, normals, and dimensions (convex hull)
    planes = ransac(lidar, normalized_diff_vectors, lidar_idx, clusters)
    for plane in planes:
        cloud = lidar[plane['inliers'], :]
        _ = plot_pc_over_image(cloud, image, T_velo2cam, K, plane=plane)

    # Merge neighboring planes
    segmentation = merge_planes(lidar, planes, debug=False)

    # Plot segmentation results
    fig = plt.figure()
    plt.cla()
    plt.clf()
    ax = fig.add_subplot(projection='3d')

    colors = ['red', 'green', 'blue']
    for i, cloud in enumerate(segmentation):
        ax.scatter(cloud['points'][:,0], cloud['points'][:,1], cloud['points'][:,2], color=colors[i % 3])

    ax.set_xlabel('x')
    ax.set_xlabel('y')
    ax.set_xlabel('z')

    plt.show()
    plt.close(fig)

if __name__ == '__main__':

    main()