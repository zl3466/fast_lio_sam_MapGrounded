#!/usr/bin/env python3
"""
Interpolate optimized poses for intermediate frames from keyframe poses.

This script reads:
1. optimized_pose.txt (KITTI format) - keyframe optimized poses
2. pos_log.txt - all frames with timestamps (to get intermediate frame timestamps)

And outputs:
- trajectory_optimized_all_frames.txt (TUM format) - all frames with interpolated optimized poses
- trajectory_optimized_all_frames_kitti.txt (KITTI format) - all frames with interpolated optimized poses
- pos_log_optimized.txt (pos_log.txt format) - all frames with optimized poses in same format as pos_log.txt

Usage:
    python3 interpolate_optimized_poses.py [optimized_pose.txt] [pos_log.txt] [output_dir]
"""

import numpy as np
import os
import sys
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple

def parse_kitti_pose(line):
    """Parse KITTI format pose: R00 R01 R02 tx R10 R11 R12 ty R20 R21 R22 tz"""
    values = line.strip().split()
    if len(values) != 12:
        return None
    
    # Extract rotation matrix and translation
    R_matrix = np.array([
        [float(values[0]), float(values[1]), float(values[2])],
        [float(values[4]), float(values[5]), float(values[6])],
        [float(values[8]), float(values[9]), float(values[10])]
    ])
    t = np.array([float(values[3]), float(values[7]), float(values[11])])
    
    return R_matrix, t

def parse_pos_log(filename):
    """Parse pos_log.txt to get all data for all frames
    
    Supports two formats:
    - Old format (25 values): time_offset rot_ang[3] position[3] omega[3] velocity[3] acc[3] gyro_bias[3] accel_bias[3] gravity[3]
    - New format (27 values): frame_idx absolute_timestamp time_offset rot_ang[3] position[3] omega[3] velocity[3] acc[3] gyro_bias[3] accel_bias[3] gravity[3]
    """
    data = {
        'time': [],
        'rot_ang': [],
        'position': [],
        'velocity': [],
        'gyro_bias': [],
        'accel_bias': [],
        'gravity': [],
        'frame_idx': [],  # Optional: frame index if available
        'time_offset': []  # Optional: time offset if available
    }
    
    with open(filename, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) == 27:
                # New format with frame index and absolute timestamp
                frame_idx = int(values[0])
                absolute_timestamp = float(values[1])
                time_offset = float(values[2])
                data['frame_idx'].append(frame_idx)
                data['time_offset'].append(time_offset)
                # Use absolute timestamp for interpolation
                data['time'].append(absolute_timestamp)
                data['rot_ang'].append([float(values[3]), float(values[4]), float(values[5])])
                data['position'].append([float(values[6]), float(values[7]), float(values[8])])
                # Skip omega (values[9-11], always 0)
                data['velocity'].append([float(values[12]), float(values[13]), float(values[14])])
                # Skip acc (values[15-17], always 0)
                data['gyro_bias'].append([float(values[18]), float(values[19]), float(values[20])])
                data['accel_bias'].append([float(values[21]), float(values[22]), float(values[23])])
                data['gravity'].append([float(values[24]), float(values[25]), float(values[26])])
            elif len(values) == 25:
                # Old format (backward compatibility)
                time_offset = float(values[0])
                data['time_offset'].append(time_offset)
                data['time'].append(time_offset)  # In old format, time is the offset
                data['rot_ang'].append([float(values[1]), float(values[2]), float(values[3])])
                data['position'].append([float(values[4]), float(values[5]), float(values[6])])
                # Skip omega (values[7-9], always 0)
                data['velocity'].append([float(values[10]), float(values[11]), float(values[12])])
                # Skip acc (values[13-15], always 0)
                data['gyro_bias'].append([float(values[16]), float(values[17]), float(values[18])])
                data['accel_bias'].append([float(values[19]), float(values[20]), float(values[21])])
                data['gravity'].append([float(values[22]), float(values[23]), float(values[24])])
    
    # Convert to numpy arrays
    for key in data:
        if len(data[key]) > 0:  # Only convert if we have data
            data[key] = np.array(data[key])
        else:
            data[key] = np.array([])
    
    return data

def parse_optimized_poses(filename):
    """Parse optimized_pose.txt (KITTI format)"""
    poses = []
    with open(filename, 'r') as f:
        for line in f:
            pose = parse_kitti_pose(line)
            if pose is not None:
                poses.append(pose)
    return poses

def parse_scene_index_map(filename):
    """Parse scene_index_map.txt to find contiguous index ranges."""
    segments: List[Tuple[int, int, str]] = []
    with open(filename, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            range_token = tokens[0]
            if "-" not in range_token:
                continue
            start_str, end_str = range_token.split("-", 1)
            try:
                start_idx = int(start_str)
                end_idx = int(end_str)
            except ValueError:
                continue
            label = tokens[1] if len(tokens) > 1 else f"segment_{len(segments)}"
            segments.append((start_idx, end_idx, label))
    if not segments:
        raise ValueError(f"No valid ranges found in scene index map: {filename}")
    return segments

def extract_last_segment(poses, segments):
    """Slice poses to keep only the last segment defined in scene_index_map."""
    # Choose the segment with the highest end index to ensure we truly get the last coverage.
    last_seg = max(segments, key=lambda seg: seg[1])
    start_idx, end_idx, label = last_seg
    if start_idx < 0 or end_idx >= len(poses):
        raise IndexError(
            f"Scene segment '{label}' range {start_idx}-{end_idx} exceeds available poses (0-{len(poses)-1})."
        )
    print(f"Using scene segment '{label}' covering indices {start_idx}-{end_idx}")
    return poses[start_idx:end_idx + 1], (start_idx, end_idx, label)

def rotation_matrix_to_quaternion(R_matrix):
    """Convert rotation matrix to quaternion (w, x, y, z)"""
    r = R.from_matrix(R_matrix)
    quat = r.as_quat()  # Returns (x, y, z, w)
    return np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to (w, x, y, z)

def quaternion_to_rotation_matrix(quat):
    """Convert quaternion (w, x, y, z) to rotation matrix"""
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy uses (x, y, z, w)
    return r.as_matrix()

def rotation_matrix_to_log(R_matrix):
    """Convert rotation matrix to log (rot_ang) - same as Log() in C++ code"""
    # Compute matrix logarithm
    r = R.from_matrix(R_matrix)
    # Get rotation vector (axis-angle representation)
    rot_vec = r.as_rotvec()
    return rot_vec

def slerp(q1, q2, t):
    """Spherical linear interpolation between two quaternions"""
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Compute dot product
    dot = np.dot(q1, q2)
    
    # If dot < 0, negate one quaternion to take shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    # Calculate angle between quaternions
    theta_0 = np.arccos(np.abs(dot))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    # Spherical interpolation
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0 * q1) + (s1 * q2)

def interpolate_poses(keyframe_poses, keyframe_times, target_times):
    """Interpolate poses for target times using keyframes"""
    interpolated_poses = []
    
    # Convert keyframe poses to quaternions and positions
    keyframe_quats = []
    keyframe_positions = []
    for R_matrix, t in keyframe_poses:
        quat = rotation_matrix_to_quaternion(R_matrix)
        keyframe_quats.append(quat)
        keyframe_positions.append(t)
    
    keyframe_quats = np.array(keyframe_quats)
    keyframe_positions = np.array(keyframe_positions)
    
    # For each target time, find surrounding keyframes and interpolate
    for target_time in target_times:
        # Find the two keyframes that bracket this time
        if target_time <= keyframe_times[0]:
            # Before first keyframe - use first keyframe
            R_interp = keyframe_poses[0][0]
            t_interp = keyframe_positions[0]
        elif target_time >= keyframe_times[-1]:
            # After last keyframe - use last keyframe
            R_interp = keyframe_poses[-1][0]
            t_interp = keyframe_positions[-1]
        else:
            # Find surrounding keyframes
            idx = np.searchsorted(keyframe_times, target_time)
            t1, t2 = keyframe_times[idx-1], keyframe_times[idx]
            
            # Interpolation parameter
            alpha = (target_time - t1) / (t2 - t1) if t2 != t1 else 0.0
            
            # Interpolate position (linear)
            t_interp = (1 - alpha) * keyframe_positions[idx-1] + alpha * keyframe_positions[idx]
            
            # Interpolate rotation (SLERP)
            q1 = keyframe_quats[idx-1]
            q2 = keyframe_quats[idx]
            q_interp = slerp(q1, q2, alpha)
            R_interp = quaternion_to_rotation_matrix(q_interp)
        
        interpolated_poses.append((R_interp, t_interp))
    
    return interpolated_poses

def write_tum_format(poses, timestamps, output_file):
    """Write poses in TUM format: timestamp x y z qx qy qz qw"""
    with open(output_file, 'w') as f:
        for (R_matrix, t), timestamp in zip(poses, timestamps):
            quat = rotation_matrix_to_quaternion(R_matrix)
            f.write(f"{timestamp:.6f} "
                   f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                   f"{quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f} {quat[0]:.6f}\n")

def write_kitti_format(poses, output_file):
    """Write poses in KITTI format: R00 R01 R02 tx R10 R11 R12 ty R20 R21 R22 tz"""
    with open(output_file, 'w') as f:
        for R_matrix, t in poses:
            f.write(f"{R_matrix[0,0]:.6f} {R_matrix[0,1]:.6f} {R_matrix[0,2]:.6f} {t[0]:.6f} "
                   f"{R_matrix[1,0]:.6f} {R_matrix[1,1]:.6f} {R_matrix[1,2]:.6f} {t[1]:.6f} "
                   f"{R_matrix[2,0]:.6f} {R_matrix[2,1]:.6f} {R_matrix[2,2]:.6f} {t[2]:.6f}\n")

def write_pos_log_format(poses, timestamps, pos_log_data, output_file):
    """Write poses in pos_log.txt format
    
    Supports two formats:
    - Old format (25 values): time_offset rot_ang[3] position[3] omega[3] velocity[3] acc[3] gyro_bias[3] accel_bias[3] gravity[3]
    - New format (27 values): frame_idx absolute_timestamp time_offset rot_ang[3] position[3] omega[3] velocity[3] acc[3] gyro_bias[3] accel_bias[3] gravity[3]
    """
    # Check if we have frame indices (new format)
    has_frame_idx = len(pos_log_data.get('frame_idx', [])) > 0
    has_time_offset = len(pos_log_data.get('time_offset', [])) > 0
    
    # Compute time_offset relative to first timestamp if needed
    first_timestamp = timestamps[0] if len(timestamps) > 0 else 0.0
    
    with open(output_file, 'w') as f:
        for i, ((R_matrix, t), timestamp) in enumerate(zip(poses, timestamps)):
            # Compute rot_ang (log of rotation matrix)
            rot_ang = rotation_matrix_to_log(R_matrix)
            
            # Get other values from original pos_log.txt (velocity, biases, gravity)
            # Use original values from pos_log_data
            vel = pos_log_data['velocity'][i] if i < len(pos_log_data['velocity']) else [0.0, 0.0, 0.0]
            gyro_bias = pos_log_data['gyro_bias'][i] if i < len(pos_log_data['gyro_bias']) else [0.0, 0.0, 0.0]
            accel_bias = pos_log_data['accel_bias'][i] if i < len(pos_log_data['accel_bias']) else [0.0, 0.0, 0.0]
            gravity = pos_log_data['gravity'][i] if i < len(pos_log_data['gravity']) else [0.0, 0.0, 0.0]
            
            # Get time_offset: use stored value if available, otherwise compute from first timestamp
            if has_time_offset and i < len(pos_log_data['time_offset']):
                time_offset = pos_log_data['time_offset'][i]
            else:
                time_offset = timestamp - first_timestamp
            
            if has_frame_idx:
                # New format (27 values)
                frame_idx = pos_log_data['frame_idx'][i] if i < len(pos_log_data['frame_idx']) else i
                f.write(f"{frame_idx} "  # frame_idx
                       f"{timestamp:.6f} "  # absolute_timestamp
                       f"{time_offset:.6f} "  # time_offset
                       f"{rot_ang[0]:.6f} {rot_ang[1]:.6f} {rot_ang[2]:.6f} "  # rot_ang[3]
                       f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "  # position[3]
                       f"0.0 0.0 0.0 "  # omega[3] (always 0)
                       f"{vel[0]:.6f} {vel[1]:.6f} {vel[2]:.6f} "  # velocity[3]
                       f"0.0 0.0 0.0 "  # acc[3] (always 0)
                       f"{gyro_bias[0]:.6f} {gyro_bias[1]:.6f} {gyro_bias[2]:.6f} "  # gyro_bias[3]
                       f"{accel_bias[0]:.6f} {accel_bias[1]:.6f} {accel_bias[2]:.6f} "  # accel_bias[3]
                       f"{gravity[0]:.6f} {gravity[1]:.6f} {gravity[2]:.6f}\n")  # gravity[3]
            else:
                # Old format (25 values)
                f.write(f"{time_offset:.6f} "  # time_offset
                       f"{rot_ang[0]:.6f} {rot_ang[1]:.6f} {rot_ang[2]:.6f} "  # rot_ang[3]
                       f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "  # position[3]
                       f"0.0 0.0 0.0 "  # omega[3] (always 0)
                       f"{vel[0]:.6f} {vel[1]:.6f} {vel[2]:.6f} "  # velocity[3]
                       f"0.0 0.0 0.0 "  # acc[3] (always 0)
                       f"{gyro_bias[0]:.6f} {gyro_bias[1]:.6f} {gyro_bias[2]:.6f} "  # gyro_bias[3]
                       f"{accel_bias[0]:.6f} {accel_bias[1]:.6f} {accel_bias[2]:.6f} "  # accel_bias[3]
                       f"{gravity[0]:.6f} {gravity[1]:.6f} {gravity[2]:.6f}\n")  # gravity[3]

def main():
    # Parse arguments
    if len(sys.argv) < 4:
        print("Usage: python3 interpolate_optimized_poses.py <optimized_pose.txt> <pos_log.txt> <output_dir>")
        print("\nExample:")
        print("  python3 interpolate_optimized_poses.py optimized_pose.txt pos_log.txt output/scene_1/")
        sys.exit(1)
    
    optimized_pose_file = sys.argv[1]
    pos_log_file = sys.argv[2]
    output_dir = sys.argv[3]
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("="*60)
    print("Interpolating Optimized Poses for Intermediate Frames")
    print("="*60)
    
    # Parse keyframe optimized poses
    print(f"Reading keyframe poses from: {optimized_pose_file}")
    keyframe_poses = parse_optimized_poses(optimized_pose_file)
    print(f"Found {len(keyframe_poses)} keyframes")
    
    # If a scene_index_map exists alongside optimized_pose.txt, slice to the last segment.
    scene_index_map_file = os.path.join(os.path.dirname(os.path.abspath(optimized_pose_file)), "scene_index_map.txt")
    if os.path.exists(scene_index_map_file):
        segments = parse_scene_index_map(scene_index_map_file)
        keyframe_poses, (seg_start, seg_end, seg_label) = extract_last_segment(keyframe_poses, segments)
        print(f"Restricting interpolation to scene '{seg_label}' ({seg_end - seg_start + 1} keyframes).")
    else:
        raise FileNotFoundError(f"scene_index_map.txt not found near {optimized_pose_file}.")
    
    # Parse all frame data from pos_log.txt
    print(f"Reading frame data from: {pos_log_file}")
    pos_log_data = parse_pos_log(pos_log_file)
    all_timestamps = pos_log_data['time']
    print(f"Found {len(all_timestamps)} frames")
    
    # Check if we have any frames
    if len(all_timestamps) == 0:
        print(f"Error: No frames found in {pos_log_file}")
        print("Cannot interpolate poses without frame timestamps.")
        sys.exit(1)
    
    # Estimate keyframe timestamps (assuming they're evenly distributed)
    n_keyframes = len(keyframe_poses)
    n_frames = len(all_timestamps)
    
    if n_keyframes >= n_frames:
        print("Warning: More keyframes than total frames. Using all frames as keyframes.")
        keyframe_indices = list(range(n_frames))
    else:
        # Distribute keyframes evenly across frames
        keyframe_indices = np.linspace(0, n_frames - 1, n_keyframes, dtype=int)
    
    keyframe_times = all_timestamps[keyframe_indices]
    
    # Check if keyframe_times is empty (shouldn't happen if n_frames > 0, but be safe)
    if len(keyframe_times) == 0:
        print(f"Error: No keyframe times could be determined.")
        print(f"Keyframes: {n_keyframes}, Frames: {n_frames}")
        sys.exit(1)
    
    print(f"Keyframe time range: {keyframe_times[0]:.2f} to {keyframe_times[-1]:.2f} seconds")
    print(f"All frames time range: {all_timestamps[0]:.2f} to {all_timestamps[-1]:.2f} seconds")
    
    # Interpolate poses for all frames
    print("Interpolating poses...")
    interpolated_poses = interpolate_poses(keyframe_poses, keyframe_times, all_timestamps)
    
    # Write output files
    tum_output = os.path.join(output_dir, "trajectory_optimized_all_frames.txt")
    kitti_output = os.path.join(output_dir, "trajectory_optimized_all_frames_kitti.txt")
    pos_log_output = os.path.join(output_dir, "pos_log_optimized.txt")
    
    print(f"Writing TUM format to: {tum_output}")
    write_tum_format(interpolated_poses, all_timestamps, tum_output)
    
    print(f"Writing KITTI format to: {kitti_output}")
    write_kitti_format(interpolated_poses, kitti_output)
    
    print(f"Writing pos_log.txt format to: {pos_log_output}")
    write_pos_log_format(interpolated_poses, all_timestamps, pos_log_data, pos_log_output)
    
    print(f"\nDone! Interpolated {len(interpolated_poses)} poses.")
    print(f"Output files:")
    print(f"  - {tum_output} (TUM format: timestamp x y z qx qy qz qw)")
    print(f"  - {kitti_output} (KITTI format: R00 R01 R02 tx R10 R11 R12 ty R20 R21 R22 tz)")
    print(f"  - {pos_log_output} (pos_log.txt format: 25 values per line)")

if __name__ == '__main__':
    main()

