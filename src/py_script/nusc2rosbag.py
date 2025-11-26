from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from tqdm import tqdm
import cv2
import json
import numpy as np
import argparse
import rosbag
import os
from sensor_msgs.msg import Image, PointCloud2, Imu
from nav_msgs.msg import Odometry
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
    
    # Load full data including ring index from binary file (single load)
    scan = np.fromfile(pcd_path, dtype=np.float32)
    points_full = scan.reshape((-1, 5))  # (x, y, z, intensity, ring)
    
    # Extract data efficiently
    pcd = points_full[:, :4].T  # (4, N) format: [x, y, z, intensity]
    ring_data = points_full[:, 4]  # Extract ring indices
    
    return pcd, lidar_data, ring_data


def cv2_to_ros_image(cv_image, timestamp, frame_id="camera"):
    """Convert OpenCV image to ROS Image message"""
    bridge = CvBridge()
    ros_image = bridge.cv2_to_imgmsg(cv_image, "bgr8")
    ros_image.header.stamp = timestamp
    ros_image.header.frame_id = frame_id
    return ros_image


def pcd_to_ros_pointcloud2(points, timestamp, frame_id="lidar", ring_data=None):
    """Convert numpy point cloud to ROS PointCloud2 message
    Compatible with OUST64 format (requires 't' and 'ring' fields)
    points: shape (4, N) where rows are [x, y, z, intensity]
    ring_data: shape (N,) containing ring indices from nuScenes
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
        # OUST64 format: x, y, z, intensity, t (uint32), reflectivity (uint16), ring (uint8), ambient (uint16), range (uint32)
        # The 't' field is relative timestamp in nanoseconds from scan start
        # Since all points in a nuScenes frame are from the same scan, we use t=0
        # Vectorized computation for performance
        ring_data = ring_data[:num_points]  # Ensure correct length
        
        # Clamp ring to uint8 range (0-255) - vectorized
        ring = np.clip(ring_data, 0, 255).astype(np.uint8)
        
        # Reflectivity as uint16 (clamp intensity to 0-65535) - vectorized
        # nuScenes intensity is typically 0-1, scale to 0-65535
        reflectivity = np.clip(intensity * 65535, 0, 65535).astype(np.uint16)
        
        # Use relative timestamp: 0 for all points (same scan time)
        t_rel = np.zeros(num_points, dtype=np.uint32)  # Relative timestamp in nanoseconds (uint32)
        
        # Calculate range (distance from origin) - vectorized
        range_val = np.sqrt(np.sum(xyz**2, axis=1)) * 1000  # Convert to mm
        range_val = np.clip(range_val, 0, 4294967295).astype(np.uint32)
        
        # Ambient (typically 0 or low value, can use intensity-related value) - vectorized
        ambient = np.clip(intensity * 1000, 0, 65535).astype(np.uint16)
        
        # Build structured array to match Ouster format exactly with proper padding
        # Ouster structure layout:
        # x(0-3), y(4-7), z(8-11), padding(12-15) = 16 bytes
        # float intensity: 16-19 bytes
        # uint32_t t: 20-23 bytes
        # uint16_t reflectivity: 24-25 bytes
        # uint8_t ring: 26 bytes
        # uint8_t padding: 27 bytes
        # uint16_t ambient: 28-29 bytes
        # uint16_t padding: 30-31 bytes
        # uint32_t range: 32-35 bytes
        # Total: 36 bytes per point
        
        # Create structured numpy array with explicit padding and byte alignment
        dtype = np.dtype([
            ('x', np.float32),      # 0-3
            ('y', np.float32),      # 4-7
            ('z', np.float32),      # 8-11
            ('padding1', np.uint32),  # 12-15 (padding to align to 16 bytes)
            ('intensity', np.float32),  # 16-19
            ('t', np.uint32),       # 20-23
            ('reflectivity', np.uint16),  # 24-25
            ('ring', np.uint8),     # 26
            ('padding2', np.uint8),  # 27 (padding)
            ('ambient', np.uint16),  # 28-29
            ('padding3', np.uint16),  # 30-31 (padding for 4-byte alignment)
            ('range', np.uint32)    # 32-35
        ], align=True)  # Use align=True to ensure proper memory layout
        
        # Create structured array
        structured_points = np.empty(num_points, dtype=dtype)
        structured_points['x'] = xyz[:, 0].astype(np.float32)
        structured_points['y'] = xyz[:, 1].astype(np.float32)
        structured_points['z'] = xyz[:, 2].astype(np.float32)
        structured_points['padding1'] = 0
        structured_points['intensity'] = intensity.astype(np.float32)
        structured_points['t'] = t_rel
        structured_points['reflectivity'] = reflectivity
        structured_points['ring'] = ring
        structured_points['padding2'] = 0
        structured_points['ambient'] = ambient
        structured_points['padding3'] = 0
        structured_points['range'] = range_val
        
        # Manually create PointCloud2 message with correct binary layout
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = num_points
        cloud_msg.is_dense = True
        
        # Set fields with correct offsets
        cloud_msg.fields = [
            point_cloud2.PointField('x', 0, point_cloud2.PointField.FLOAT32, 1),
            point_cloud2.PointField('y', 4, point_cloud2.PointField.FLOAT32, 1),
            point_cloud2.PointField('z', 8, point_cloud2.PointField.FLOAT32, 1),
            point_cloud2.PointField('intensity', 16, point_cloud2.PointField.FLOAT32, 1),
            point_cloud2.PointField('t', 20, point_cloud2.PointField.UINT32, 1),
            point_cloud2.PointField('reflectivity', 24, point_cloud2.PointField.UINT16, 1),
            point_cloud2.PointField('ring', 26, point_cloud2.PointField.UINT8, 1),
            point_cloud2.PointField('ambient', 28, point_cloud2.PointField.UINT16, 1),
            point_cloud2.PointField('range', 32, point_cloud2.PointField.UINT32, 1)
        ]
        
        # Set point_step and row_step
        cloud_msg.point_step = dtype.itemsize  # Should be 36 bytes
        cloud_msg.row_step = cloud_msg.point_step * num_points
        
        # Convert structured array to bytes and assign to data
        cloud_msg.data = structured_points.tobytes()
        cloud_msg.is_dense = True  # Explicitly set to dense (assuming no NaN values)
        return cloud_msg
    elif has_intensity:
        # Fallback: simple format without ring - vectorized
        points_list = np.column_stack([xyz, intensity]).tolist()
        
        cloud_msg = point_cloud2.create_cloud(header, 
                                        [point_cloud2.PointField('x', 0, point_cloud2.PointField.FLOAT32, 1),
                                         point_cloud2.PointField('y', 4, point_cloud2.PointField.FLOAT32, 1),
                                         point_cloud2.PointField('z', 8, point_cloud2.PointField.FLOAT32, 1),
                                         point_cloud2.PointField('intensity', 12, point_cloud2.PointField.FLOAT32, 1)],
                                        points_list)
        cloud_msg.is_dense = True  # Explicitly set to dense (assuming no NaN values)
        return cloud_msg
    else:
        cloud_msg = point_cloud2.create_cloud_xyz32(header, xyz)
        cloud_msg.is_dense = True  # Explicitly set to dense (assuming no NaN values)
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
    
    # Based on working dataset: gravity must be in +Z (positive, ~9.8-10.2 m/s^2)
    # Your original IMU data showed z: 9.774 (positive), so NuScenes has +Z gravity
    # Keep data as-is to match working dataset format (gravity in +Z)
    imu_msg.linear_acceleration.x = imu_json['acc'][0]
    imu_msg.linear_acceleration.y = imu_json['acc'][1]
    imu_msg.linear_acceleration.z = imu_json['acc'][2]  # Keep as-is: NuScenes has +Z, which is correct
    
    imu_msg.angular_velocity.x = imu_json['avel'][0]
    imu_msg.angular_velocity.y = imu_json['avel'][1]
    imu_msg.angular_velocity.z = imu_json['avel'][2]  # Keep as-is
    
    # Orientation (quaternion) - use from ego_pose if provided, otherwise zeros
    if orientation_quaternion is not None:
        # NuScenes format is [w, x, y, z], ROS format is [x, y, z, w]
        imu_msg.orientation.w = orientation_quaternion[0]
        imu_msg.orientation.x = orientation_quaternion[1]
        imu_msg.orientation.y = orientation_quaternion[2]
        imu_msg.orientation.z = orientation_quaternion[3]
        # Set reasonable covariance for orientation
        imu_msg.orientation_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
    else:
        # Default to zeros if no orientation provided
        imu_msg.orientation.w = 0.0
        imu_msg.orientation.x = 0.0
        imu_msg.orientation.y = 0.0
        imu_msg.orientation.z = 0.0
        imu_msg.orientation_covariance = [0.0] * 9
    
    # Set covariance for linear acceleration and angular velocity
    imu_msg.linear_acceleration_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
    imu_msg.angular_velocity_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
    
    return imu_msg


def pose_to_ros_odometry(pose_data, timestamp, frame_id="odom", child_frame_id="base_link"):
    """Convert NuScenes pose to ROS Odometry message"""
    odom_msg = Odometry()
    odom_msg.header.stamp = timestamp
    odom_msg.header.frame_id = frame_id
    odom_msg.child_frame_id = child_frame_id
    
    # Translation
    odom_msg.pose.pose.position.x = pose_data['translation'][0]
    odom_msg.pose.pose.position.y = pose_data['translation'][1]
    odom_msg.pose.pose.position.z = pose_data['translation'][2]
    
    # Rotation (quaternion) - NuScenes format is [w, x, y, z]
    odom_msg.pose.pose.orientation.x = pose_data['rotation'][1]
    odom_msg.pose.pose.orientation.y = pose_data['rotation'][2]
    odom_msg.pose.pose.orientation.z = pose_data['rotation'][3]
    odom_msg.pose.pose.orientation.w = pose_data['rotation'][0]
    
    # Set covariance (identity matrix for now, can be improved)
    odom_msg.pose.covariance = [0] * 36
    odom_msg.pose.covariance[0] = 0.1  # x
    odom_msg.pose.covariance[7] = 0.1  # y
    odom_msg.pose.covariance[14] = 0.1  # z
    odom_msg.pose.covariance[21] = 0.1  # roll
    odom_msg.pose.covariance[28] = 0.1  # pitch
    odom_msg.pose.covariance[35] = 0.1  # yaw
    
    return odom_msg


def timestamp_to_ros_time(timestamp_us):
    """Convert microseconds timestamp to ROS Time"""
    timestamp_sec = timestamp_us / 1e6
    return Time.from_sec(timestamp_sec)


def main():
    parser = argparse.ArgumentParser(description='Convert NuScenes dataset to ROS bag')
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
                        help='Output bag file path (default: {nusc_dir}/scene_{scene_idx}.bag)')
    
    args = parser.parse_args()
    
    nusc = NuScenes(version=args.nusc_ver, dataroot=args.nusc_dir, verbose=True)

    target_scene_idx = args.scene_idx
    if args.output_bag:
        output_bag_path = args.output_bag
    else:
        output_bag_path = f"{args.nusc_dir}/scene_{target_scene_idx}.bag"
    
    # Start/end timestamps are in microseconds (NuScenes format)
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
            
            # Filter samples by timestamp
            filtered_sample_tokens = []
            if start_tp_us is not None or end_tp_us is not None:
                for sample_token in all_sample_tokens:
                    my_sample = nusc.get("sample", sample_token)
                    # Get sample timestamp (use lidar timestamp as reference)
                    lidar_token = my_sample['data'].get('LIDAR_FRONT_CENTER', None)
                    if lidar_token:
                        lidar_data = nusc.get('sample_data', lidar_token)
                        sample_timestamp_us = lidar_data['timestamp']
                        
                        # Check if sample is within time range
                        if start_tp_us is not None and sample_timestamp_us < start_tp_us:
                            continue
                        if end_tp_us is not None and sample_timestamp_us > end_tp_us:
                            continue
                        
                        filtered_sample_tokens.append(sample_token)
                    else:
                        # If no lidar, include it (or skip based on your preference)
                        filtered_sample_tokens.append(sample_token)
                
                print(f"Filtered to {len(filtered_sample_tokens)} samples within time range")
                all_sample_tokens = filtered_sample_tokens
            
            # Process each sample in the scene
            for sample_idx, sample_token in enumerate(tqdm(all_sample_tokens, desc=f"Scene {scene_idx + 1} samples", leave=False)):
                my_sample = nusc.get("sample", sample_token)
                channels = my_sample['data'].keys()
                
                # Get sample timestamp (use lidar timestamp as reference)
                lidar_token = my_sample['data'].get('LIDAR_FRONT_CENTER', None)
                if lidar_token:
                    lidar_data = nusc.get('sample_data', lidar_token)
                    sample_timestamp = timestamp_to_ros_time(lidar_data['timestamp'])
                else:
                    # Fallback: use current time if no lidar available
                    sample_timestamp = Time.now()

                # Process camera images
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

                # Process lidar point cloud
                lidar_data = None
                if "LIDAR_FRONT_CENTER" in channels:
                    lidar_pcd, lidar_data, ring_data = nusc_get_pcd(nusc, my_sample, "LIDAR_FRONT_CENTER")
                    if lidar_pcd is not None and lidar_pcd.shape[1] > 0:
                        lidar_timestamp = timestamp_to_ros_time(lidar_data['timestamp'])
                        ros_pcd = pcd_to_ros_pointcloud2(lidar_pcd, lidar_timestamp, frame_id="lidar_front_center", ring_data=ring_data)
                        bag.write("/lidar/points", ros_pcd, lidar_timestamp)
                    
                    # Get ego pose from lidar_data (NuScenes stores ego_pose_token in sample_data)
                    if lidar_data and 'ego_pose_token' in lidar_data and lidar_data['ego_pose_token']:
                        try:
                            pose_data = nusc.get('ego_pose', lidar_data['ego_pose_token'])
                            pose_timestamp = timestamp_to_ros_time(lidar_data['timestamp'])
                            
                            # Publish Odometry to /pose topic (expected by FAST-LIVO2 as nav_msgs/Odometry)
                            ros_odom = pose_to_ros_odometry(pose_data, pose_timestamp, frame_id="odom", child_frame_id="base_link")
                            bag.write("/pose", ros_odom, pose_timestamp)
                        except Exception as e:
                            print(f"Warning: Could not get ego pose for sample {sample_token}: {e}")

                # Process IMU data
                if "IMU_TOP" in channels:
                    imu_sample_data = nusc.get('sample_data', my_sample['data']["IMU_TOP"])
                    imu_file_path = f"{nusc.dataroot}/{imu_sample_data['filename']}"
                    with open(imu_file_path, 'r') as f:
                        imu_json = json.load(f)
                    
                    imu_timestamp = timestamp_to_ros_time(imu_sample_data['timestamp'])
                    
                    # Get ego pose orientation for IMU (IMU is mounted on vehicle, so use ego_pose orientation)
                    orientation_quaternion = None
                    if 'ego_pose_token' in imu_sample_data and imu_sample_data['ego_pose_token']:
                        try:
                            imu_pose_data = nusc.get('ego_pose', imu_sample_data['ego_pose_token'])
                            orientation_quaternion = imu_pose_data['rotation']  # NuScenes format: [w, x, y, z]
                        except Exception as e:
                            print(f"Warning: Could not get ego pose for IMU sample {imu_sample_data['token']}: {e}")
                    
                    ros_imu = imu_data_to_ros_imu(imu_json, imu_timestamp, frame_id="imu_top", 
                                                  orientation_quaternion=orientation_quaternion)
                    bag.write("/imu/data", ros_imu, imu_timestamp)
    
    print(f"\nConversion complete! Rosbag saved to: {output_bag_path}")


if __name__ == '__main__':
    main()
