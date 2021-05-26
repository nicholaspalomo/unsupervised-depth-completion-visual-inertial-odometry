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
import os, sys, argparse
from re import S
import numpy as np
import tensorflow as tf
import global_constants as settings
import data_utils, eval_utils
from dataloader import DataLoader
from voiced_model import VOICEDModel
from data_utils import log
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import time

def make_mean_abs_err_plot(z, z_gt, K, img):

    # compute SE and graph
    se = np.abs(z_gt - np.squeeze(z)) * settings.MAX_Z
    graph = []
    for i in range(se.shape[0]):
        for j in range(se.shape[1]):
            if ~np.isnan(se[i, j]):
                graph.append([z_gt[i, j], se[i, j], i, j]) # x, y
    graph = np.array(graph)
    idx_sort = np.argsort(graph[:, 0])
    graph = graph[idx_sort, :]

    unique_x, unique_x_idx = np.unique(graph[:, 0], return_index=True)

    unique_y = np.zeros(unique_x.shape)
    max_y = np.zeros(unique_x.shape)
    min_y = np.zeros(unique_x.shape)
    percentage = np.zeros(unique_x.shape)
    for i in range(unique_x_idx.shape[0]-1):
        unique_y[i] = np.mean(graph[unique_x_idx[i]:unique_x_idx[i+1], 1])
        max_y[i] = np.max(graph[unique_x_idx[i]:unique_x_idx[i+1], 1])
        min_y[i] = np.min(graph[unique_x_idx[i]:unique_x_idx[i+1], 1])
        percentage[i] = (unique_x_idx[i+1] - unique_x_idx[i]) / graph.shape[0]
    unique_y[-1] = graph[unique_x_idx[-1], 1]
    max_y[-1] = graph[unique_x_idx[-1], 1]
    min_y[-1] = graph[unique_x_idx[-1], 1]
    percentage[-1] = 1 / graph.shape[0]

    plt.plot(unique_x * settings.MAX_Z, unique_y)
    plt.fill_between(unique_x * settings.MAX_Z, min_y, max_y, alpha=0.3)
    plt.fill_between(unique_x * settings.MAX_Z, np.zeros(percentage.shape), percentage * 100, alpha=0.3)
    plt.xlabel('distance to camera [m]')
    plt.legend(['mean absolute estimation error [m]', 'error bounds [m]', 'depth distribution [%]'])
    plt.show()
    plt.close()

    # make error map
    rgb = plt.cm.get_cmap('jet')
    colors = []
    for point in graph:
        err = point[1]
        i = point[2]
        j = point[3]

        color = rgb(err / np.max(graph[:, 1]), bytes=True)
        color = list(color)
        color = color[:-1]
        colors.append(color[::-1])

        cv2.circle(img=img,
                center=(np.int_(j), np.int_(i)),
                radius=1,
                color=np.float_(color),
                thickness=-1)
                
    plt.imshow(img)
    plt.show()
    plt.close()

def make_cloud_visualization(img, z, K):

    # create the point cloud
    points = np.zeros((img.shape[0]*img.shape[1], 6)) # x, y, z, r, g, b for point cloud
    i = 0
    for u in range(img.shape[1]):
        for v in range(img.shape[0]):
            
            # convert u, v coordinates to x-y-z
            xy = np.squeeze(np.matmul(np.linalg.inv(K), np.array([u, v, 1])[:, np.newaxis]))

            points[i, 0] = xy[0]
            points[i, 1] = xy[1]
            points[i, 2] = z[v, u]
            points[i, 3:] = img[v, u, :]
            i += 1

    points = points[~np.isnan(points[:, 2]), :]

    # create an O3D point cloud object
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    cloud.colors = o3d.utility.Vector3dVector(np.fliplr(points[:, 3:]) / 255.)

    o3d.visualization.draw_geometries([cloud])

    return

def plot_point_cloud(z, gt, vm, img_path, intrinsics_path):

    # load the image
    img = cv2.imread(img_path)
    img = img[:, (img.shape[1] // 3):(2*(img.shape[1] // 3)), :]

    # load the camera intrinsic matrix
    K = np.load(intrinsics_path)

    # densified point cloud
    make_cloud_visualization(img, np.squeeze(z), K)

    # sparse point cloud
    z_gt = gt.copy()
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if vm[i, j] < 1:
                z_gt[i, j] = np.nan
    make_cloud_visualization(img, z_gt, K)

    make_mean_abs_err_plot(z, z_gt, K, img)

parser = argparse.ArgumentParser()

N_HEIGHT = 352
N_WIDTH = 1216

parser.add_argument('--plt_dense_pc', type=bool, default=settings.PLT_DENSE_PC, help='Plot densified point cloud')
# Intrinsics
parser.add_argument('--intrinsics_file',
    type=str, default=settings.INTRINSICS_FILE, help='Path to intrinsics file (npy)')
# Model path
parser.add_argument('--restore_path',
    type=str, default=settings.RESTORE_PATH, help='Path to restore model')
# Input paths
parser.add_argument('--image_path',
    type=str, default=settings.IMAGE_PATH, help='Path to list of image paths')
parser.add_argument('--interp_depth_path',
    type=str, default=settings.INTERP_DEPTH_PATH, help='Path to list of interpolated depth paths')
parser.add_argument('--validity_map_path',
    type=str, default=settings.VALIDITY_MAP_PATH, help='Path to list of validity map paths')
parser.add_argument('--ground_truth_path',
    type=str, default=settings.GROUND_TRUTH_PATH, help='Path to list of ground truth paths')
parser.add_argument('--start_idx',
    type=int, default=settings.START_IDX, help='Start index of the list of paths to evaluate')
parser.add_argument('--end_idx',
    type=int, default=settings.END_IDX, help='Last index of the list of paths to evaluate')
# Batch parameters
parser.add_argument('--n_batch',
    type=int, default=settings.N_BATCH, help='Number of samples per batch')
parser.add_argument('--n_height',
    type=int, default=settings.N_HEIGHT, help='Height of of sample')
parser.add_argument('--n_width',
    type=int, default=settings.N_WIDTH, help='Width of each sample')
parser.add_argument('--n_channel',
    type=int, default=settings.N_CHANNEL, help='Number of channels for each image')
# Network settings
parser.add_argument('--occ_threshold',
    type=float, default=settings.OCC_THRESHOLD, help='Threshold for max change in sparse depth')
parser.add_argument('--occ_ksize',
    type=int, default=settings.OCC_KSIZE, help='Kernel size for checking for possible occlusion')
parser.add_argument('--net_type',
    type=str, default=settings.NET_TYPE, help='Network architecture types: vggnet08, vggnet11')
parser.add_argument('--im_filter_pct',
    type=float, default=settings.IM_FILTER_PCT, help='Percentage filters for the image branch')
parser.add_argument('--sz_filter_pct',
    type=float, default=settings.SZ_FILTER_PCT, help='Percentage filter for the sparse depth branch')
parser.add_argument('--min_predict_z',
    type=float, default=settings.MIN_Z, help='Minimum depth prediction')
parser.add_argument('--max_predict_z',
    type=float, default=settings.MAX_Z, help='Maximum depth prediction')
parser.add_argument('--min_evaluate_z',
    type=float, default=settings.MIN_Z, help='Minimum depth to evaluate')
parser.add_argument('--max_evaluate_z',
    type=float, default=settings.MAX_Z, help='Maximum depth to evaluate')
# Output options
parser.add_argument('--save_depth',
    action='store_true', help='If set, saves depth maps into output_path')
parser.add_argument('--output_path',
    type=str, default='output', help='Directory to store output')
# Hardware settings
parser.add_argument('--n_thread',
    type=int, default=settings.N_THREAD, help='Number of threads for fetching')


args = parser.parse_args()

log_path = os.path.join(args.output_path, 'results.txt')
if not os.path.exists(args.output_path):
  os.makedirs(args.output_path)

# Load image paths from file for evaluation
im_paths = sorted(data_utils.read_paths(args.image_path))[args.start_idx:args.end_idx]
iz_paths = sorted(data_utils.read_paths(args.interp_depth_path))[args.start_idx:args.end_idx]
vm_paths = sorted(data_utils.read_paths(args.validity_map_path))[args.start_idx:args.end_idx]
assert(len(im_paths) == len(iz_paths))
assert(len(im_paths) == len(vm_paths))
n_sample = len(im_paths)

if args.ground_truth_path != '':
  gt_paths = sorted(data_utils.read_paths(args.ground_truth_path))[args.start_idx:args.end_idx]
  assert(len(im_paths) == len(gt_paths))

# Pad all paths based on batch
im_paths = data_utils.pad_batch(im_paths, args.n_batch)
iz_paths = data_utils.pad_batch(iz_paths, args.n_batch)
vm_paths = data_utils.pad_batch(vm_paths, args.n_batch)
n_step = len(im_paths)//args.n_batch

gt_arr = []
if args.ground_truth_path != '':
  # Load ground truth
  for idx in range(n_sample):
    sys.stdout.write(
        'Loading {}/{} groundtruth depth maps \r'.format(idx+1, n_sample))
    sys.stdout.flush()

    gt, vm = data_utils.load_depth_with_validity_map(gt_paths[idx])
    gt = np.concatenate([np.expand_dims(gt, axis=-1), np.expand_dims(vm, axis=-1)], axis=-1)
    gt_arr.append(gt)

  print('Completed loading {} groundtruth depth maps'.format(n_sample))

with tf.Graph().as_default():
  # Initialize dataloader
  dataloader = DataLoader(shape=[args.n_batch, args.n_height, args.n_width, 3],
                          name='dataloader',
                          is_training=False,
                          n_thread=args.n_thread,
                          prefetch_size=2*args.n_thread)
  # Fetch the input from dataloader
  im0 = dataloader.next_element[0]
  sz0 = dataloader.next_element[3]

  # Build computation graph
  model = VOICEDModel(im0, im0, im0, sz0, None,
                      is_training=False,
                      occ_threshold=args.occ_threshold,
                      occ_ksize=args.occ_ksize,
                      net_type=args.net_type,
                      im_filter_pct=args.im_filter_pct,
                      sz_filter_pct=args.sz_filter_pct,
                      min_predict_z=args.min_predict_z,
                      max_predict_z=args.max_predict_z)

  # Initialize Tensorflow session
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  session = tf.Session(config=config)
  # Load from checkpoint
  train_saver = tf.train.Saver()
  session.run(tf.global_variables_initializer())
  session.run(tf.local_variables_initializer())
  train_saver.restore(session, args.restore_path)

  log('Evaluating {}'.format(args.restore_path), log_path)
  # Load image, dense depth, sparse depth, intrinsics, and ground-truth
  dataloader.initialize(session,
                        image_paths=im_paths,
                        interp_depth_paths=iz_paths,
                        validity_map_paths=vm_paths)

  z_arr = np.zeros([n_step*args.n_batch, args.n_height, args.n_width, 1])
  step = 0
  avg_time = 0
  while True:
    try:
      sys.stdout.write(
          'Processed {}/{} examples \r'.format(step*args.n_batch, n_sample))
      sys.stdout.flush()

      batch_start = step*args.n_batch
      batch_end = step*args.n_batch+args.n_batch
      step += 1
      t = time.time()
      z_arr[batch_start:batch_end, ...] = session.run(model.predict)
      print("Elapsed inferencing time: {}".format(time.time() - t))
      avg_time += time.time() - t
    except tf.errors.OutOfRangeError:
      break

  print("Averaged elapsed inferencing time: {}".format(avg_time / step))

  # Remove the padded examples
  z_arr = z_arr[0:n_sample, ...]

  if args.ground_truth_path != '':
    if args.plt_dense_pc:
        for idx in range(n_sample):
            gt, vm = data_utils.load_depth_with_validity_map(gt_paths[idx])
            plot_point_cloud(z_arr[idx, ...], gt, vm, im_paths[idx], args.intrinsics_file)

  # Run evaluation
  if len(gt_arr) > 0:
    mae   = np.zeros(n_sample, np.float32)
    rmse  = np.zeros(n_sample, np.float32)
    imae  = np.zeros(n_sample, np.float32)
    irmse = np.zeros(n_sample, np.float32)

    for idx in range(n_sample):
      z = np.squeeze(z_arr[idx, ...])
      gt = np.squeeze(gt_arr[idx][..., 0])
      vm = np.squeeze(gt_arr[idx][..., 1])

      # Create mask for evaluation
      valid_mask = np.where(vm > 0, 1, 0)
      min_max_mask = np.logical_and(gt > args.min_evaluate_z, gt < args.max_evaluate_z)
      mask = np.where(np.logical_and(valid_mask, min_max_mask) > 0)
      z = z[mask]
      gt = gt[mask]

      # Run evaluations: MAE, RMSE in meters, iMAE, iRMSE in 1/kilometers
      mae[idx] = eval_utils.mean_abs_err(1000.0*z, 1000.0*gt)
      rmse[idx] = eval_utils.root_mean_sq_err(1000.0*z, 1000.0*gt)
      imae[idx] = eval_utils.inv_mean_abs_err(0.001*z, 0.001*gt)
      irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001*z, 0.001*gt)

    # Compute mean error
    mae   = np.mean(mae)
    rmse  = np.mean(rmse)
    imae  = np.mean(imae)
    irmse = np.mean(irmse)
    log('{:>10} {:>10} {:>10} {:>10}'.format('MAE', 'RMSE', 'iMAE', 'iRMSE'), log_path)
    log('{:10.4f} {:10.4f} {:10.4f} {:10.4f}'.format(mae, rmse, imae, irmse), log_path)

  # Store output depth as images
  if args.save_depth:
    output_dirpath = os.path.join(args.output_path, 'saved')
    print('Storing output depth as PNG into {}'.format(output_dirpath))

    if not os.path.exists(output_dirpath):
      os.makedirs(output_dirpath)

    for idx in range(n_sample):
      z = np.squeeze(z_arr[idx, ...])
      _, filename = os.path.split(iz_paths[idx])
      output_path = os.path.join(output_dirpath, filename)
      data_utils.save_depth(z, output_path)
