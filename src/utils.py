#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import math
from matplotlib import pyplot as plt

# import ROS-specific modules
import rospy
import rosbag # bagpy
import sensor_msgs.point_cloud2 as pc2 # point clouds
from image_geometry import PinholeCameraModel
import tf
from tf import transformations as tfm

import os
import pathlib
import glob
import h5py
from datetime import datetime
import vispy
from vispy.scene import visuals, SceneCanvas
import cv2
import open3d as o3d

import sys
sys.path.append('.')

class Camera:
    def __init__(self, params):

        self.bag_path = params['bags_path']
        self.robot_name = params['robot_name']
        self.camera_name = params['camera_name']
        self.camera_info_topic = params['camera_info_topic']
        self.hdf5_fname = params['hdf5_path']

        self.tf_bag_file = None
        for name in glob.glob(self.bag_path + '/*tf*.bag'):
            if name.find('orig') == -1:
                self.tf_bag_file = name
                break

        self.vision_bag_file = None
        for name in glob.glob(self.bag_path + '/*vision*.bag'):
            if name.find('orig') == -1:
                self.vision_bag_file = name
                break

        self.model = self.init_camera_model()
        self.transformer = self.init_tf_tree()

    def init_camera_model(self):

        print("Initializing camera models")
        model = PinholeCameraModel()
        with rosbag.Bag(self.vision_bag_file) as bag:

            # for topic, msg, t in bag.read_messages():
            #     print(topic)

            info_topic = "/" + self.robot_name + "/" + self.camera_name + "/" + self.camera_info_topic
            for topic, msg, t in bag.read_messages([info_topic]):
                model.fromCameraInfo(msg)
                break
            else:
                raise RuntimeError("Camera model not available")
        print("Camera models loaded")

        print("Projection matrix:")
        print(model.projectionMatrix())
        print("TF Frame ID: {}".format(model.tfFrame()))

        return model

    def init_tf_tree(self):
        print("Initializing TF tree")
        transformer = tf.Transformer()
        with rosbag.Bag(self.tf_bag_file) as bag:
            for topic, msg, t in bag.read_messages(["/tf_static"]):
                for tf_msg in msg.transforms:
                    tf_msg.header.stamp = rospy.Time(0)
                    transformer.setTransform(tf_msg)
        print("TF tree constructed")

        return transformer

    def get_tf_lidar2cam(self, frame_id):

        position, orientation = self.transformer.lookupTransform(
            self.robot_name + '/' + self.camera_name + '/' + 'camera_color_optical_frame', # self.model.tfFrame(),
            frame_id,
            rospy.Time(0)
        )

        print("position: {}".format(position))
        print("quaternion: {}".format(orientation))

        camera_T_lidar = tfm.quaternion_matrix(orientation)
        camera_T_lidar[:3, 3] = position

        return camera_T_lidar

    def load_data_and_visualize(self, img_idx=0):

        if not os.path.isfile(self.hdf5_fname):
            raise RuntimeError("Provided directory, {}, does not exist.".format(self.hdf5_fname))

        with h5py.File(self.hdf5_fname, "r") as h5:
        
            # print("Keys: {}".format(h5.keys()))
            # data_key = list(h5.keys())[0]
    
            # extract the point cloud; get rid of the NaNs
            point_clouds = h5['lidar']['lidar'][img_idx, :]
            
            K = np.reshape(h5['image']['intrinsics'][:], (3, 3))
            
            # plot the second point cloud in the sequence and visualize in 3D
            point_clouds = point_clouds[~np.isnan(point_clouds)]
            point_clouds = np.reshape(point_clouds, (-1, 4))
            intensities = point_clouds[:, -1]
            point_clouds = point_clouds[:, :3]
            points = np.hstack((point_clouds, np.zeros((point_clouds.shape[0], 1))))
            
            # project the point cloud to the image
            img = np.reshape(h5['image']['image'][img_idx, :], (h5['image']['resolution'][0], h5['image']['resolution'][1], 3))
            
            T_velo_2_camera = np.reshape(h5['lidar']['tf_velo2cam'][:], (4,4))

            if "rear" in self.camera_name:
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

                T_velo_2_camera = np.matmul(T_camera_2_img, T_velo_2_camera) 
            
            pc_cam = np.matmul(np.hstack((point_clouds[:,:3], np.ones((point_clouds.shape[0], 1)))), T_velo_2_camera.T)
            pc_cam_pos_z_idx = np.where((pc_cam[:, 2] > 0))[0]
            pc_cam = pc_cam[pc_cam_pos_z_idx, :]
            
            w = img.shape[1]
            h = img.shape[0]
            
            pc_img = pc_cam.copy()
            pc_img = np.matmul(pc_img[:, :3], K.T) # (N x 3) x (3 x 3) 
            pc_img = pc_img / pc_img[:, 2, None]
            
            width_mask = ((pc_img[:,0] >= 0) & (pc_img[:,0] <= img.shape[1]))
            height_mask = ((pc_img[:,1] >= 0) & (pc_img[:,1] <= img.shape[0]))
            pc_img_mask_idx = np.where(width_mask & height_mask)[0]
            pc_img = pc_img[pc_img_mask_idx, :]
            
            pc_velo = points[pc_cam_pos_z_idx, :]
            intensities_camera_front = intensities[pc_cam_pos_z_idx]

            rgb = plt.cm.get_cmap('jet')
            colors = []
            for img_point, velo_point, intensity in zip(pc_img, pc_velo[pc_img_mask_idx, :], intensities_camera_front[pc_img_mask_idx]):

                color = rgb(intensity, bytes=True)
                color = list(color)
                color = color[:-1]
                colors.append(color[::-1])
                
                cv2.circle(img=img,
                    center=(np.int_(img_point[0]), np.int_(img_point[1])),
                    radius=1,
                    color=np.float_(color),
                    thickness=-1)
                
            plt.imshow(img)
            plt.gcf().savefig(os.path.dirname(os.path.abspath(__file__)) + datetime.now().strftime("/%d%m%Y%H%M%S"), dpi=300)
            plt.close()
            
            colors = []
            for point, intensity in zip(points, intensities):
                
                color = rgb(intensity, bytes=True)
                color = list(color)
                color = color[:-1]
                colors.append(color[::-1])

            cloud = o3d.geometry.PointCloud()
            point_clouds = point_clouds[pc_cam_pos_z_idx, :]
            point_clouds = point_clouds[pc_img_mask_idx, :]
            cloud.points = o3d.utility.Vector3dVector(point_clouds)

            cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100.0,
                max_nn=50))

            o3d.visualization.draw_geometries([cloud])

class PointCloud:
    def __init__(self, pc=None, lidar=None):
        
        if pc is not None:
            self.pc = pc.copy()
        else:
            self.pc = np.zeros((1,4))
        self.canvas = None
        self.grid = None
        self.view = None
        self.sem_vis = visuals.Markers()
        self.sem_color = None
        self.sem_label = None
        self.axis = None
        self.lidar = lidar
        self.laser_id = np.zeros((self.pc.shape[0],))
        
    def add_visualization(self):

        self.canvas = SceneCanvas(keys='interactive', show=True)
        self.grid = self.canvas.central_widget.add_grid()
        self.view = vispy.scene.widgets.ViewBox(border_color='white',
                        parent=self.canvas.scene)
        self.grid.add_widget(self.view, 0, 0)

        # Point Cloud Visualizer
        self.view.camera = vispy.scene.cameras.TurntableCamera(up='z', azimuth=90)
        self.view.add(self.sem_vis)
        # self.axis = visuals.XYZAxis(parent=self.view.scene)

    def update_view(self, view):

        self.view = view
        self.view.add(self.sem_vis)
        
    def run_visualization(self):

        vispy.app.run()

    def update_pc(self, points, sem_label=None, color_map=None):
        """
        Update the colors in the point cloud visualization according to the semantic label
        inputs:
        - points [N x 4]            x, y, z & intensity point cloud points
        - sem_label [N]             semantic labels for point cloud
        - color_map [dictionary]    color in BGR for each semantic label
        """

        # note: call squeeze() on sem_label
        self.pc = points.copy()

        if color_map is not None and sem_label is not None:
            self.sem_color = []
            self.sem_label = np.squeeze(sem_label.copy())
            for label in self.sem_label:
                self.sem_color.append(color_map[label][::-1]) # colors given in BGR
            self.sem_color = np.array(self.sem_color) / 255
            self.sem_color = np.hstack((self.sem_color, self.pc[:, -1, np.newaxis]))

            self.sem_vis.set_data(self.pc[:, :3], edge_color=None, face_color=self.sem_color, size=3)
        else:
            self.sem_vis.set_data(self.pc[:, :3], edge_color=None, size=3)

    @staticmethod
    def depth_color(val, min_d=0, max_d=120):
        """
        print Color(HSV's H value) corresponding to distance(m)
        close distance = red , far distance = blue
        """
        val = np.clip(val, min_d, max_d)  # max distance is 120m but usually not usual
        return (val - min_d) / (max_d - min_d)

class LiDARCameraRosbagData:
    def __init__(self, params):

        self.bags_path = params["bags_path"]

        self.filenames = dict()
        self.filenames["lidar"] = []
        self.filenames["vision"] = []
        self.filenames["tf"] = []
        
        self.robot_name = params["robot_name"]
        self.camera_name = params["camera_name"]
        self.camera_image_topic = '/' + self.robot_name + '/' + self.camera_name + '/' + params["camera_image_topic"]
        self.camera_depth_image_topic = '/' + self.robot_name + '/' + self.camera_name + '/' + params["depth_camera_image_topic"]
        self.camera_info_topic = '/' + self.robot_name + '/' + self.camera_name + '/' + params["camera_info_topic"]
        self.camera_depth_info_topic = '/' + self.robot_name + '/' + self.camera_name + '/' + params["depth_camera_info_topic"]
        self.lidar_topic = '/' + self.robot_name + '/' + params["point_cloud_topic"]
        self.lidar_out_filename = params["hdf5_path"]

    def get_bag_names(self, path):

        if not os.path.isdir(path):
            raise RuntimeError("Provided directory does not exist.")

        names = ["lidar", "vision", "tf"]
        for name in names:
            self.glob_get_filenames(name, path)

    def glob_get_filenames(self, string, path):

        for name in glob.glob(path + '/*' + string + '*.bag'):
            if name.find('orig') == -1:
                print(name)
                self.filenames[string].append(name)

    @staticmethod
    def print_topic_names(bag):

        for topic, msg, t in bag.read_messages():
            print(topic)

    def read_images_from_bags(self):

        for bag_fn in self.filenames["vision"]:
            with rosbag.Bag(bag_fn, "r") as bag:
                count = 0
                for topic, msg, t in bag.read_messages(topics=self.camera_image_topic):

                    cv_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

                    cv2.imwrite(os.path.join(os.getcwd() + "/img", "frame%06i.png" % count), cv_img)
                    print("Wrote image %i" % count)

                    count += 1

    def create_dataset_from_vision_lidar_tf(self, camera, max_idx=10, max_pc_len=120000):

        for name in ["vision", "lidar", "tf"]:
            self.glob_get_filenames(name, self.bags_path)

        hf = h5py.File(self.lidar_out_filename, 'w')

        image_group = hf.create_group("image")

        camera_resolution_dataset = image_group.create_dataset("resolution", data=np.zeros((2,), dtype=np.int_), compression="gzip", chunks=True, maxshape=(2,))

        camera_intrinsics_dataset = image_group.create_dataset("intrinsics", data=np.zeros((9,), dtype=np.float_), compression="gzip", chunks=True, maxshape=(9,))

        depth_camera_resolution_dataset = image_group.create_dataset("depth_resolution", data=np.zeros((2,), dtype=np.int_), compression="gzip", chunks=True, maxshape=(2,))

        depth_camera_intrinsics_dataset = image_group.create_dataset("depth_intrinsics", data=np.zeros((9,), dtype=np.float_), compression="gzip", chunks=True, maxshape=(9,))

        image_timestamp_dataset = image_group.create_dataset("timestamps", data=np.empty((max_idx,), dtype=np.float_), compression="gzip", chunks=True, maxshape=(max_idx,))

        image_idx_dataset = image_group.create_dataset("image_idx", data=np.empty((max_idx,), dtype=np.int_), compression="gzip", chunks=True, maxshape=(max_idx,))

        count = 0
        flag_rgb = False
        flag_depth = False
        for vision_bag_name in self.filenames["vision"]:
            with rosbag.Bag(vision_bag_name, "r") as bag:
                # LiDARCameraRosbagData.print_topic_names(bag)
                for topic, msg, t in bag.read_messages(topics=self.camera_info_topic):
                    h = msg.height
                    w = msg.width
                    camera_resolution_dataset[:] = np.array([h, w])

                    camera_intrinsics_dataset[:] = np.array(msg.K)
                    
                    if not flag_rgb:
                        rgb_image_dataset = image_group.create_dataset("image", data=np.empty((max_idx, h*w, 3), dtype=np.uint8), compression="gzip", chunks=True, maxshape=(max_idx,h*w, 3))
                        flag_rgb = True

                    break

                for topic, msg, t in bag.read_messages(topics=self.camera_depth_info_topic):
                    depth_h = msg.height
                    depth_w = msg.width
                    depth_camera_resolution_dataset[:] = np.array([depth_h, depth_w])

                    depth_camera_intrinsics_dataset[:] = np.array(msg.K)

                    if not flag_depth:
                        depth_image_dataset = image_group.create_dataset("depth_image", data=np.empty((max_idx, depth_h*depth_w), dtype=np.uint8), compression="gzip", chunks=True, maxshape=(max_idx,depth_h*depth_w))
                        flag_depth = True

                    break
                
                count_prev = count
                for topic, msg, t in bag.read_messages(topics=self.camera_image_topic):
                    if count < max_idx:
                        img = np.frombuffer(msg.data, dtype=np.uint8)
                        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                        rgb_image_dataset[count, :, :] = np.reshape(img, (h*w, 3))

                        image_timestamp_dataset[count] = t.to_sec()

                        cv2.imwrite(os.path.join(os.getcwd() + "/img", "frame%06i.png" % count), np.reshape(img, (h, w, 3)))
                        print("Wrote image {}".format(count))
                        count += 1

                count = count_prev
                for topic, msg, t in bag.read_messages(topics=self.camera_depth_image_topic):
                    if count < max_idx:
                        img = np.frombuffer(msg.data[12:], dtype=np.uint8)
                        img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
                        # img = np.frombuffer(msg.data, dtype=np.uint8)
                        depth_image_dataset[count, :] = np.reshape(img, (-1,)).astype(np.uint8)

                        cv2.imwrite(
                            os.path.join(os.getcwd() + "/img", "depth_frame%06i.png" % count), 
                            np.reshape(img, (depth_h, -1)))
                        print("Wrote depth image {}".format(count))
                        count += 1

        for i, j in zip(np.argsort(image_timestamp_dataset[:]), range(image_timestamp_dataset[:].shape[0])):
            tmp_img = rgb_image_dataset[i, :, :].copy()
            rgb_image_dataset[i, :, :] = rgb_image_dataset[j, :, :]
            rgb_image_dataset[j, :, :] = tmp_img

            tmp_time = image_timestamp_dataset[i].copy()
            image_timestamp_dataset[i] = image_timestamp_dataset[j]
            image_timestamp_dataset[j] = tmp_time

        image_idx_dataset[:] = np.array(range(image_idx_dataset.shape[0]))

        lidar_group = hf.create_group("lidar")
        
        lidar_points_dataset = lidar_group.create_dataset("lidar", data=np.empty((image_idx_dataset.shape[0], max_pc_len), dtype=np.float_), compression="gzip", chunks=True, maxshape=(image_idx_dataset.shape[0], max_pc_len)) # name according to the lidar topic that recorded it
        lidar_points_dataset[:,:] = np.NaN

        lidar_timestamp_dataset = lidar_group.create_dataset("timestamps", data=np.empty((image_idx_dataset.shape[0],), dtype=np.float_), compression="gzip", chunks=True, maxshape=(image_idx_dataset.shape[0],))

        lidar_idx_dataset = lidar_group.create_dataset("lidar_idx", data=np.empty((image_idx_dataset.shape[0],), dtype=np.int_), compression="gzip", chunks=True, maxshape=(image_idx_dataset.shape[0],))

        lidar_pc_len_dataset = lidar_group.create_dataset("lidar_pc_len", data=np.zeros((image_idx_dataset.shape[0],), dtype=np.int_), compression="gzip", chunks=True, maxshape=(image_idx_dataset.shape[0],))

        flag = True
        lidar_idx = -1
        for lidar_bag_name in self.filenames["lidar"]:
            with rosbag.Bag(lidar_bag_name, "r") as bag:
                # LiDARCameraRosbagData.print_topic_names(bag)
                if lidar_idx < max_idx:
                    for topic, msg, t in bag.read_messages(topics=self.lidar_topic):
                        if flag:
                            transforms_dataset = lidar_group.create_dataset("tf_velo2cam", data=np.empty((16,), dtype=np.float_), compression="gzip", chunks=True, maxshape=(16,))

                            transforms_dataset[:] = np.reshape(camera.get_tf_lidar2cam(msg.header.frame_id), (16,))

                            flag = False

                        pts = np.array(
                            list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity")))
                            )

                        lidar_idx = np.where((image_timestamp_dataset[:] - t.to_sec() < 0) * (image_timestamp_dataset[:] - t.to_sec() > -0.2))[0] # 0.2s here is a heuristic

                        if lidar_idx.shape[0] != 0:

                            lidar_idx = lidar_idx[-1]

                            lidar_len = pts.shape[0]*pts.shape[1]

                            lidar_idx_dataset[lidar_idx] = lidar_idx

                            if lidar_pc_len_dataset[lidar_idx] < max_pc_len:
                            
                                if lidar_pc_len_dataset[lidar_idx]+lidar_len >= max_pc_len:
                                    lidar_points_dataset[lidar_idx, lidar_pc_len_dataset[lidar_idx]:max_pc_len] = np.reshape(pts, (pts.shape[0]*pts.shape[1],))[max_pc_len - lidar_pc_len_dataset[lidar_idx]]

                                else:
                                    lidar_points_dataset[lidar_idx, lidar_pc_len_dataset[lidar_idx]:lidar_pc_len_dataset[lidar_idx]+lidar_len] = np.reshape(pts, (pts.shape[0]*pts.shape[1],))

                                lidar_timestamp_dataset[lidar_idx] = t.to_sec()

                                print("The point cloud corresponds to image index {}. Filename: {}. Timestamp: {}. Matching timestamp: {}. Delta: {}.".format(lidar_idx, lidar_bag_name, t.to_sec(), image_timestamp_dataset[lidar_idx], image_timestamp_dataset[lidar_idx]-t.to_sec()))

                            else:
                                print("Point cloud buffer exceeded. Skipping point cloud at t={} and index={}".format(t.to_sec(), lidar_idx))

                            lidar_pc_len_dataset[lidar_idx] += lidar_len

                        else:

                            lidar_idx = -1

        hf.close()

def main(args, params):

    camera = Camera(params)
    util = LiDARCameraRosbagData(params)
    if args.create_dataset == True:
        # creat the training dataset and write it to file
        util.create_dataset_from_vision_lidar_tf(camera, max_idx=int(args.max_idx), max_pc_len=120000)
    else:
        camera.load_data_and_visualize(img_idx=int(args.img_idx))

        util.get_bag_names(args.bags_path)

        # print topics in rosbags
        for bag_id in list(util.filenames.keys()):
            if bag_id != 'tf':
                files = util.filenames[bag_id]
                with rosbag.Bag(files[0], "r") as bag:
                    LiDARCameraRosbagData.print_topic_names(bag)

if __name__ == "__main__":
    '''
    example:
        --create_dataset True
        --hdf5_path ../costar.h5
        --bags_path src/bags/
        --robot_name spot2
        --camera_name camera_front
        --camera_image_topic color/image_raw_throttle/compressed # or e.g. '/mono/image_raw/compressed'
        --depth_camera_image_topic aligned_depth_to_color/image_raw_throttle/compressedDepth
        --camera_info_topic color/camera_info
        --depth_camera_info_topic aligned_depth_to_color/camera_info
        --point_cloud_topic velodyne_points # NOTE: "/spot_driver/local_grid" also valid (for Spot only)
        --max_idx 1200
        --img_idx 20
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument("--create_dataset", help="T/F: Create H5 file", required=False, type=bool, default=False)

    parser.add_argument("--hdf5_path", help="Directory to H5 file with dataset", required=False, type=str, default=os.path.join(pathlib.Path(__file__).parent.absolute(), "data/costar.h5"))

    parser.add_argument("--bags_path", help="Path to bag files", required=False, type=str, default=os.path.join(pathlib.Path(__file__).parent.absolute(), "bags"))

    parser.add_argument("--robot_name", choices=["spot1", "spot2", "husky1", "husky2"], help="Robot name, e.g. husky1, spot2", required=False, type=str, default="spot2")

    parser.add_argument("--camera_name", choices=["camera_front", "camera_left", "camera_right", "builtin_camera_rear"], help="Name of camera, e.g. camera_front, builtin_camera_rear", required=False, type=str, default="camera_front")

    parser.add_argument("--camera_image_topic", help="Name of camera image topic", required=False, type=str, default="color/image_raw_throttle/compressed")

    parser.add_argument("--depth_camera_image_topic", help="Name of depth camera image topic", required=False, type=str, default="aligned_depth_to_color/image_raw_throttle/compressedDepth")

    parser.add_argument("--camera_info_topic", help="Name of the camera info topic containing the camera intrinsics", required=False, type=str, default="color/camera_info")

    parser.add_argument("--depth_camera_info_topic", help="Name of the depth camera info topic containing the depth camera intrinsics", required=False, type=str, default="aligned_depth_to_color/camera_info")

    parser.add_argument("--point_cloud_topic", help="Name of the point cloud topic", required=False, type=str, default="velodyne_points")

    parser.add_argument("--img_idx", help="Index of image in dataset (for plotting/visualization)", required=False, type=int, default=20)

    parser.add_argument("--max_idx", help="Maximum index to extract from rosbags", required=False, type=int, default=50)
    args = parser.parse_args()

    params = {"hdf5_path"                   : args.hdf5_path,
              "bags_path"                   : args.bags_path,
              "robot_name"                  : args.robot_name,
              "camera_name"                 : args.camera_name,
              "camera_image_topic"          : args.camera_image_topic,
              "depth_camera_image_topic"    : args.depth_camera_image_topic,
              "camera_info_topic"           : args.camera_info_topic,
              "depth_camera_info_topic"     : args.depth_camera_info_topic,
              "point_cloud_topic"           : args.point_cloud_topic}

    main(args, params)