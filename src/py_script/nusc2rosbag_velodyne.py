from nuscenes import NuScenes
from nuscenes.utils.data_classes import Quaternion
from tqdm import tqdm
import cv2
import json
import numpy as np
import argparse
import rosbag
import os
from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix, TimeReference
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs import point_cloud2
from genpy import Time


def nusc_get_img_cv2(nusc, sample, cam_name):
    cam_data = nusc.get('sample_data', sample['data'][cam_name])
    img_path = f"{nusc.dataroot}/{cam_data['filename']}"
    cv2_img = cv2.imread(img_path)

    return cv2_img, cam_data


def nusc_get_pcd(nusc, sample, lidar_name):
    lidar_data_token = sample['data'][lidar_name]
    lidar_data = nusc.get('sample_data', lidar_data_token)
    pcd_path = f"{nusc.dataroot}/{lidar_data['filename']}"
    
    scan = np.fromfile(pcd_path, dtype=np.float32)
    points_full = scan.reshape((-1, 5))  # (x, y, z, intensity, ring)
    pcd = points_full[:, :4].T  # (4, N) format: [x, y, z, intensity]
    ring_data = points_full[:, 4]
    
    return pcd, lidar_data, ring_data


def cv2_to_ros_image(cv_image, timestamp, frame_id="camera"):
    """Convert OpenCV image to ROS Image message"""
    bridge = CvBridge()
    ros_image = bridge.cv2_to_imgmsg(cv_image, "bgr8")
    ros_image.header.stamp = timestamp
    ros_image.header.frame_id = frame_id
    return ros_image


def pcd_to_ros_pointcloud2_velodyne(points, timestamp, frame_id="lidar", ring_data=None):
    """Convert numpy point cloud to ROS PointCloud2 message in Velodyne format
    Compatible with Velodyne 128 format (requires 'ring' and 'time' fields)
    points: shape (4, N) where rows are [x, y, z, intensity]
    ring_data: shape (N,) containing ring indices from nuScenes
    
    Velodyne format structure (26 bytes per point):
    - x, y, z: float32 (12 bytes)
    - padding: 4 bytes (for 16-byte alignment)
    - intensity: float32 (4 bytes)
    - ring: uint16 (2 bytes) - for Velodyne 128, range is 0-127
    - time: float32 (4 bytes) - relative timestamp within scan
    Total: 26 bytes per point
    """
    xyz = points[:3, :].T  # N x 3
    has_intensity = points.shape[0] > 3
    intensity = points[3, :] if has_intensity else np.zeros(points.shape[1])
    
    # Create point cloud
    header = Header()
    header.stamp = timestamp
    header.frame_id = frame_id
    
    num_points = xyz.shape[0]
    
    if has_intensity and ring_data is not None:
        ring_data = ring_data[:num_points]
        ring = np.clip(ring_data, 0, 65535).astype(np.uint16)
        t_rel = np.zeros(num_points, dtype=np.float32)
        
        dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('padding1', np.uint32),
            ('intensity', np.float32),
            ('ring', np.uint16),
            ('time', np.float32)
        ], align=True)
        
        structured_points = np.empty(num_points, dtype=dtype)
        structured_points['x'] = xyz[:, 0].astype(np.float32)
        structured_points['y'] = xyz[:, 1].astype(np.float32)
        structured_points['z'] = xyz[:, 2].astype(np.float32)
        structured_points['padding1'] = 0
        structured_points['intensity'] = intensity.astype(np.float32)
        structured_points['ring'] = ring
        structured_points['time'] = t_rel
        
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = num_points
        cloud_msg.is_dense = True
        
        cloud_msg.fields = [
            point_cloud2.PointField('x', 0, point_cloud2.PointField.FLOAT32, 1),
            point_cloud2.PointField('y', 4, point_cloud2.PointField.FLOAT32, 1),
            point_cloud2.PointField('z', 8, point_cloud2.PointField.FLOAT32, 1),
            point_cloud2.PointField('intensity', 16, point_cloud2.PointField.FLOAT32, 1),
            point_cloud2.PointField('ring', 20, point_cloud2.PointField.UINT16, 1),
            point_cloud2.PointField('time', 22, point_cloud2.PointField.FLOAT32, 1)
        ]
        
        cloud_msg.point_step = dtype.itemsize
        cloud_msg.row_step = cloud_msg.point_step * num_points
        cloud_msg.data = structured_points.tobytes()
        return cloud_msg
    elif has_intensity:
        points_list = np.column_stack([xyz, intensity]).tolist()
        cloud_msg = point_cloud2.create_cloud(header, 
                                        [point_cloud2.PointField('x', 0, point_cloud2.PointField.FLOAT32, 1),
                                         point_cloud2.PointField('y', 4, point_cloud2.PointField.FLOAT32, 1),
                                         point_cloud2.PointField('z', 8, point_cloud2.PointField.FLOAT32, 1),
                                         point_cloud2.PointField('intensity', 12, point_cloud2.PointField.FLOAT32, 1)],
                                        points_list)
        cloud_msg.is_dense = True
        return cloud_msg
    else:
        cloud_msg = point_cloud2.create_cloud_xyz32(header, xyz)
        cloud_msg.is_dense = True
        return cloud_msg


def imu_data_to_ros_imu(imu_json, timestamp, frame_id="imu", orientation_quaternion=None):
    """Convert NuScenes IMU data to ROS Imu message
    Args:
        imu_json: NuScenes IMU JSON data
        timestamp: ROS timestamp
        frame_id: Frame ID for the IMU
        orientation_quaternion: Optional quaternion [w, x, y, z] from ego_pose (NuScenes format)
    
    Note: Based on the working dataset, GTSAM's MakeSharedU expects gravity in +Z (positive, ~9.8-10.2 m/s^2).
    The working dataset shows: avg=(0.710, -0.926, 10.247) with gravity in +Z.
    We ensure the output has gravity in +Z to match this format.
    """
    imu_msg = Imu()
    imu_msg.header.stamp = timestamp
    imu_msg.header.frame_id = frame_id
    
    imu_msg.linear_acceleration.x = imu_json['acc'][0]
    imu_msg.linear_acceleration.y = imu_json['acc'][1]
    imu_msg.linear_acceleration.z = imu_json['acc'][2]
    
    imu_msg.angular_velocity.x = imu_json['avel'][0]
    imu_msg.angular_velocity.y = imu_json['avel'][1]
    imu_msg.angular_velocity.z = imu_json['avel'][2]
    
    if orientation_quaternion is not None:
        imu_msg.orientation.w = orientation_quaternion[0]
        imu_msg.orientation.x = orientation_quaternion[1]
        imu_msg.orientation.y = orientation_quaternion[2]
        imu_msg.orientation.z = orientation_quaternion[3]
        imu_msg.orientation_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
    else:
        imu_msg.orientation.w = 0.0
        imu_msg.orientation.x = 0.0
        imu_msg.orientation.y = 0.0
        imu_msg.orientation.z = 0.0
        imu_msg.orientation_covariance = [0.0] * 9
    
    imu_msg.linear_acceleration_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
    imu_msg.angular_velocity_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
    
    return imu_msg


def pose_to_ros_odometry(pose_data, timestamp, frame_id="odom", child_frame_id="base_link"):
    """Convert NuScenes pose to ROS Odometry message"""
    odom_msg = Odometry()
    odom_msg.header.stamp = timestamp
    odom_msg.header.frame_id = frame_id
    odom_msg.child_frame_id = child_frame_id
    
    odom_msg.pose.pose.position.x = pose_data['translation'][0]
    odom_msg.pose.pose.position.y = pose_data['translation'][1]
    odom_msg.pose.pose.position.z = pose_data['translation'][2]
    
    odom_msg.pose.pose.orientation.x = pose_data['rotation'][1]
    odom_msg.pose.pose.orientation.y = pose_data['rotation'][2]
    odom_msg.pose.pose.orientation.z = pose_data['rotation'][3]
    odom_msg.pose.pose.orientation.w = pose_data['rotation'][0]
    
    odom_msg.pose.covariance = [0] * 36
    odom_msg.pose.covariance[0] = 0.1  # x
    odom_msg.pose.covariance[7] = 0.1  # y
    odom_msg.pose.covariance[14] = 0.1  # z
    odom_msg.pose.covariance[21] = 0.1  # roll
    odom_msg.pose.covariance[28] = 0.1  # pitch
    odom_msg.pose.covariance[35] = 0.1  # yaw
    
    return odom_msg


def imu_json_to_ros_navsatfix(imu_json, timestamp, frame_id="navsat_link"):
    """Convert NuScenes IMU JSON GPS data to ROS NavSatFix message
    Args:
        imu_json: NuScenes IMU JSON data containing 'lat', 'lon', 'elev'
        timestamp: ROS timestamp
        frame_id: Frame ID for the GPS fix
    """
    navsat_msg = NavSatFix()
    navsat_msg.header.stamp = timestamp
    navsat_msg.header.frame_id = frame_id
    
    if 'lat' in imu_json and 'lon' in imu_json:
        navsat_msg.latitude = imu_json['lat']
        navsat_msg.longitude = imu_json['lon']
    else:
        navsat_msg.latitude = 0.0
        navsat_msg.longitude = 0.0
    
    if 'elev' in imu_json:
        navsat_msg.altitude = imu_json['elev']
    else:
        navsat_msg.altitude = 0.0
    
    navsat_msg.status.status = 2
    navsat_msg.status.service = 1
    
    navsat_msg.position_covariance = [
        0.0084, 0.0, 0.0,
        0.0, 0.0119, 0.0,
        0.0, 0.0, 207.36
    ]
    navsat_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
    
    return navsat_msg


def imu_json_to_ros_twist_stamped(imu_json, timestamp, frame_id="navsat_link"):
    """Convert NuScenes IMU JSON velocity data to ROS TwistStamped message
    Args:
        imu_json: NuScenes IMU JSON data containing 'vel' array [x, y, z]
        timestamp: ROS timestamp
        frame_id: Frame ID for the velocity
    """
    twist_msg = TwistStamped()
    twist_msg.header.stamp = timestamp
    twist_msg.header.frame_id = frame_id
    
    if 'vel' in imu_json and len(imu_json['vel']) >= 3:
        twist_msg.twist.linear.x = imu_json['vel'][0]
        twist_msg.twist.linear.y = imu_json['vel'][1]
        twist_msg.twist.linear.z = imu_json['vel'][2]
    else:
        twist_msg.twist.linear.x = 0.0
        twist_msg.twist.linear.y = 0.0
        twist_msg.twist.linear.z = 0.0
    
    twist_msg.twist.angular.x = 0.0
    twist_msg.twist.angular.y = 0.0
    twist_msg.twist.angular.z = 0.0
    
    return twist_msg


def imu_json_to_ros_time_reference(imu_json, timestamp, frame_id="navsat_link"):
    """Convert NuScenes IMU JSON time data to ROS TimeReference message
    Args:
        imu_json: NuScenes IMU JSON data containing 'utime' (microseconds)
        timestamp: ROS timestamp (from sample_data timestamp)
        frame_id: Frame ID for the time reference
    """
    time_ref_msg = TimeReference()
    time_ref_msg.header.stamp = timestamp
    time_ref_msg.header.frame_id = frame_id
    
    if 'utime' in imu_json:
        utime_us = imu_json['utime']
        time_ref_msg.time_ref.secs = int(utime_us // 1000000)
        time_ref_msg.time_ref.nsecs = int((utime_us % 1000000) * 1000)
    else:
        time_ref_msg.time_ref = timestamp
    
    time_ref_msg.source = frame_id
    
    return time_ref_msg


def timestamp_to_ros_time(timestamp_us):
    """Convert microseconds timestamp to ROS Time"""
    timestamp_sec = timestamp_us / 1e6
    return Time.from_sec(timestamp_sec)


def rotation_matrix_to_rpy(R):
    """Convert rotation matrix to roll, pitch, yaw angles (ZYX convention)
    
    Args:
        R: 3x3 rotation matrix
    
    Returns:
        tuple: (roll, pitch, yaw) in radians
    """
    # Extract RPY using ZYX convention (same as ROS)
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    
    return roll, pitch, yaw


def transform_matrix_to_extrinsics(T):
    """Extract extrinsic parameters from 4x4 transformation matrix
    
    Args:
        T: 4x4 homogeneous transformation matrix
    
    Returns:
        dict: Contains extrinsicTrans (list), extrinsicRot (list), extrinsicRPY (list), rpy (list)
    """
    # Extract translation
    translation = T[:3, 3].tolist()
    
    # Extract rotation matrix
    rotation = T[:3, :3]
    rotation_flat = rotation.flatten().tolist()  # Row-major order
    
    # Convert to RPY angles
    roll, pitch, yaw = rotation_matrix_to_rpy(rotation)
    rpy = [roll, pitch, yaw]
    
    return {
        'extrinsicTrans': translation,
        'extrinsicRot': rotation_flat,
        'extrinsicRPY': rotation_flat,  # Same as extrinsicRot (rotation matrix format)
        'rpy': rpy  # Actual RPY angles in radians
    }


def get_lidar_imu_transform(nusc, lidar_sensor_token, imu_sensor_token):
    """Get transformation matrices between lidar and IMU sensors
    
    Args:
        nusc: NuScenes object
        lidar_sensor_token: Token for lidar sample_data
        imu_sensor_token: Token for IMU sample_data
    
    Returns:
        tuple: (T_lidar_to_imu, T_imu_to_lidar) as 4x4 numpy arrays
    """
    # Get lidar sample_data and calibrated_sensor
    lidar_data = nusc.get('sample_data', lidar_sensor_token)
    lidar_cs = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    
    # Get IMU sample_data and calibrated_sensor
    imu_data = nusc.get('sample_data', imu_sensor_token)
    imu_cs = nusc.get('calibrated_sensor', imu_data['calibrated_sensor_token'])
    
    # Build transformation from lidar to ego frame
    lidar_rotation = Quaternion(lidar_cs['rotation']).rotation_matrix
    lidar_translation = np.array(lidar_cs['translation'])
    T_lidar_to_ego = np.eye(4)
    T_lidar_to_ego[:3, :3] = lidar_rotation
    T_lidar_to_ego[:3, 3] = lidar_translation
    
    # Build transformation from IMU to ego frame
    imu_rotation = Quaternion(imu_cs['rotation']).rotation_matrix
    imu_translation = np.array(imu_cs['translation'])
    T_imu_to_ego = np.eye(4)
    T_imu_to_ego[:3, :3] = imu_rotation
    T_imu_to_ego[:3, 3] = imu_translation
    
    # Compute transformation from ego to IMU (inverse of IMU to ego)
    T_ego_to_imu = np.linalg.inv(T_imu_to_ego)
    
    # Compute transformation from lidar to IMU
    T_lidar_to_imu = T_ego_to_imu @ T_lidar_to_ego
    
    # Compute transformation from IMU to lidar (inverse of lidar to IMU)
    T_imu_to_lidar = np.linalg.inv(T_lidar_to_imu)
    
    return T_lidar_to_imu, T_imu_to_lidar


def save_lidar_imu_extrinsics(nusc, lidar_sensor_token, imu_sensor_token, output_path):
    """Compute and save lidar-IMU extrinsics to JSON file
    
    Args:
        nusc: NuScenes object
        lidar_sensor_token: Token for lidar sample_data
        imu_sensor_token: Token for IMU sample_data
        output_path: Path to save JSON file
    """
    # Get transformation matrices
    T_lidar_to_imu, T_imu_to_lidar = get_lidar_imu_transform(nusc, lidar_sensor_token, imu_sensor_token)
    
    # Extract extrinsics for both transforms
    lidar_to_imu_extrinsics = transform_matrix_to_extrinsics(T_lidar_to_imu)
    imu_to_lidar_extrinsics = transform_matrix_to_extrinsics(T_imu_to_lidar)
    
    # Create output dictionary
    output_data = {
        'lidar_to_imu': lidar_to_imu_extrinsics,
        'imu_to_lidar': imu_to_lidar_extrinsics
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Extrinsics saved to: {output_path}")
    return output_data


def main():
    parser = argparse.ArgumentParser(description='Convert NuScenes dataset to ROS bag with Velodyne format point clouds')
    parser.add_argument('--nusc_dir', type=str, default="/media/zl3466/新加卷/MARS_10Hz_long",
                        help='NuScenes dataset directory')
    parser.add_argument('--nusc_ver', type=str, default="v1.1_z",
                        help='NuScenes version')
    parser.add_argument('--scene_idx', type=int, default=2,
                        help='Target scene index')
    parser.add_argument('--start_tp', type=float, default=None,
                        help='Start timestamp in microseconds (filter samples before this time)')
    parser.add_argument('--end_tp', type=float, default=None,
                        help='End timestamp in microseconds (filter samples after this time)')
    parser.add_argument('--output_bag', type=str, default=None,
                        help='Output bag file path (default: {nusc_dir}/scene_{scene_idx}_velodyne.bag)')
    
    args = parser.parse_args()
    
    nusc = NuScenes(version=args.nusc_ver, dataroot=args.nusc_dir, verbose=True)

    target_scene_idx = args.scene_idx
    if args.output_bag:
        output_bag_path = args.output_bag
    else:
        output_bag_path = f"{args.nusc_dir}/scene_{target_scene_idx}_velodyne.bag"
    
    start_tp_us = None if args.start_tp is None else int(args.start_tp)
    end_tp_us = None if args.end_tp is None else int(args.end_tp)
    
    if start_tp_us is not None:
        print(f"Filtering samples: start_tp = {start_tp_us} us ({start_tp_us / 1e6:.6f} s)")
    if end_tp_us is not None:
        print(f"Filtering samples: end_tp = {end_tp_us} us ({end_tp_us / 1e6:.6f} s)")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_bag_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with rosbag.Bag(output_bag_path, 'w') as bag:
        for scene_idx, scene in enumerate(tqdm(nusc.scene, desc="Processing scenes")):
            if scene_idx != target_scene_idx:
                continue
            
            print(f"Processing scene {scene_idx}: {scene['name']}")
            
            # Get all sample tokens for this scene
            all_sample_tokens = []
            sample = nusc.get("sample", scene["first_sample_token"])
            all_sample_tokens.append(scene["first_sample_token"])
            
            # Collect all sample tokens in the scene
            for j in range(scene['nbr_samples'] - 1):
                if sample['next'] != '':
                    all_sample_tokens.append(sample['next'])
                    sample = nusc.get('sample', sample['next'])
                else:
                    break
            
            print(f"Found {len(all_sample_tokens)} samples in scene {scene_idx}")
            
            # Compute and save extrinsics using the first sample
            if len(all_sample_tokens) > 0:
                first_sample = nusc.get("sample", all_sample_tokens[0])
                lidar_token = first_sample['data'].get('LIDAR_FRONT_CENTER', None)
                imu_token = first_sample['data'].get('IMU_TOP', None)
                
                if lidar_token and imu_token:
                    extrinsics_output_path = f"{args.nusc_dir}/lidar_imu_extrinsics_scene_{scene_idx}.json"
                    try:
                        save_lidar_imu_extrinsics(nusc, lidar_token, imu_token, extrinsics_output_path)
                    except Exception as e:
                        print(f"Warning: Could not save extrinsics: {e}")
            
            filtered_sample_tokens = []
            if start_tp_us is not None or end_tp_us is not None:
                for sample_token in all_sample_tokens:
                    my_sample = nusc.get("sample", sample_token)
                    lidar_token = my_sample['data'].get('LIDAR_FRONT_CENTER', None)
                    if lidar_token:
                        lidar_data = nusc.get('sample_data', lidar_token)
                        sample_timestamp_us = lidar_data['timestamp']
                        
                        if start_tp_us is not None and sample_timestamp_us < start_tp_us:
                            continue
                        if end_tp_us is not None and sample_timestamp_us > end_tp_us:
                            continue
                        
                        filtered_sample_tokens.append(sample_token)
                    else:
                        filtered_sample_tokens.append(sample_token)
                
                print(f"Filtered to {len(filtered_sample_tokens)} samples within time range")
                all_sample_tokens = filtered_sample_tokens
            
            # Process each sample in the scene
            for sample_idx, sample_token in enumerate(tqdm(all_sample_tokens, desc=f"Scene {scene_idx + 1} samples", leave=False)):
                my_sample = nusc.get("sample", sample_token)
                channels = my_sample['data'].keys()
                
                lidar_token = my_sample['data'].get('LIDAR_FRONT_CENTER', None)
                if lidar_token:
                    lidar_data = nusc.get('sample_data', lidar_token)
                    sample_timestamp = timestamp_to_ros_time(lidar_data['timestamp'])
                else:
                    sample_timestamp = Time.now()

                camera_topics = {
                    "CAM_FRONT_CENTER": "/camera/front_center/image_raw",
                    # "CAM_FRONT_LEFT": "/camera/front_left/image_raw",
                    # "CAM_FRONT_RIGHT": "/camera/front_right/image_raw",
                    # "CAM_BACK_CENTER": "/camera/back_center/image_raw",
                    # "CAM_SIDE_LEFT": "/camera/side_left/image_raw",
                    # "CAM_SIDE_RIGHT": "/camera/side_right/image_raw"
                }
                
                for cam_name, topic in camera_topics.items():
                    if cam_name in channels:
                        img, cam_data = nusc_get_img_cv2(nusc, my_sample, cam_name)
                        if img is not None:
                            cam_timestamp = timestamp_to_ros_time(cam_data['timestamp'])
                            ros_img = cv2_to_ros_image(img, cam_timestamp, frame_id=cam_name.lower())
                            bag.write(topic, ros_img, cam_timestamp)

                lidar_data = None
                if "LIDAR_FRONT_CENTER" in channels:
                    lidar_pcd, lidar_data, ring_data = nusc_get_pcd(nusc, my_sample, "LIDAR_FRONT_CENTER")
                    if lidar_pcd is not None and lidar_pcd.shape[1] > 0:
                        lidar_timestamp = timestamp_to_ros_time(lidar_data['timestamp'])
                        ros_pcd = pcd_to_ros_pointcloud2_velodyne(lidar_pcd, lidar_timestamp, frame_id="lidar_front_center", ring_data=ring_data)
                        bag.write("/lidar/points", ros_pcd, lidar_timestamp)
                    
                    if lidar_data and 'ego_pose_token' in lidar_data and lidar_data['ego_pose_token']:
                        try:
                            pose_data = nusc.get('ego_pose', lidar_data['ego_pose_token'])
                            pose_timestamp = timestamp_to_ros_time(lidar_data['timestamp'])
                            ros_odom = pose_to_ros_odometry(pose_data, pose_timestamp, frame_id="odom", child_frame_id="base_link")
                            bag.write("/pose", ros_odom, pose_timestamp)
                        except Exception as e:
                            print(f"Warning: Could not get ego pose for sample {sample_token}: {e}")

                if "IMU_TOP" in channels:
                    imu_sample_data = nusc.get('sample_data', my_sample['data']["IMU_TOP"])
                    imu_file_path = f"{nusc.dataroot}/{imu_sample_data['filename']}"
                    with open(imu_file_path, 'r') as f:
                        imu_json = json.load(f)
                    
                    imu_timestamp = timestamp_to_ros_time(imu_sample_data['timestamp'])
                    
                    orientation_quaternion = None
                    if 'ego_pose_token' in imu_sample_data and imu_sample_data['ego_pose_token']:
                        try:
                            imu_pose_data = nusc.get('ego_pose', imu_sample_data['ego_pose_token'])
                            orientation_quaternion = imu_pose_data['rotation']
                        except Exception as e:
                            print(f"Warning: Could not get ego pose for IMU sample {imu_sample_data['token']}: {e}")
                    
                    ros_imu = imu_data_to_ros_imu(imu_json, imu_timestamp, frame_id="imu_top", 
                                                  orientation_quaternion=orientation_quaternion)
                    bag.write("/imu/data", ros_imu, imu_timestamp)
                    
                    if 'lat' in imu_json and 'lon' in imu_json:
                        ros_gps_fix = imu_json_to_ros_navsatfix(imu_json, imu_timestamp, frame_id="navsat_link")
                        bag.write("/gps/fix", ros_gps_fix, imu_timestamp)
                        
                        if 'vel' in imu_json:
                            ros_gps_vel = imu_json_to_ros_twist_stamped(imu_json, imu_timestamp, frame_id="navsat_link")
                            bag.write("/gps/vel", ros_gps_vel, imu_timestamp)
                        
                        if 'utime' in imu_json:
                            ros_gps_time = imu_json_to_ros_time_reference(imu_json, imu_timestamp, frame_id="navsat_link")
                            bag.write("/gps/time_reference", ros_gps_time, imu_timestamp)
    
    print(f"\nConversion complete! Rosbag saved to: {output_bag_path}")


if __name__ == '__main__':
    main()