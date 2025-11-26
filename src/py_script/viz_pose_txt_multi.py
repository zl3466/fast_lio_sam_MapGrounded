#!/usr/bin/env python3
"""
Visualize two pose.txt files together in one interactive 3D plot

Supports two formats:
1. KITTI format (12 values): R00 R01 R02 tx R10 R11 R12 ty R20 R21 R22 tz
2. pos_log.txt format (25 values): time rot_ang[3] position[3] omega[3] velocity[3] acc[3] gyro_bias[3] accel_bias[3] gravity[3]

Usage:
    python3 viz_pose_txt_multi.py <pose_file1.txt> <pose_file2.txt> [output_file.html] [label1] [label2]
    
Example:
    python3 viz_pose_txt_multi.py pos_log.txt pos_log_optimized.txt comparison.html "Original" "Optimized"
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
PLOTLY_AVAILABLE = True


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


def parse_pos_log_format(filename):
    """Parse pos_log.txt format (25 values per line)"""
    data = {
        'time': [],
        'rot_ang': [],
        'position': [],
        'velocity': [],
        'gyro_bias': [],
        'accel_bias': [],
        'gravity': []
    }
    
    with open(filename, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) == 25:
                data['time'].append(float(values[0]))
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
        data[key] = np.array(data[key])
    
    return data


def parse_kitti_format(filename):
    """Parse KITTI format (12 values per line) and convert to same structure as pos_log"""
    positions = []
    rotations = []
    
    with open(filename, 'r') as f:
        for line_idx, line in enumerate(f):
            pose = parse_kitti_pose(line)
            if pose is not None:
                R_matrix, t = pose
                positions.append(t)
                rotations.append(R_matrix)
    
    # Convert to same format as pos_log
    positions = np.array(positions)
    
    # Generate synthetic time (frame index * estimated dt, or use frame index)
    # For KITTI format, we don't have timestamps, so we'll use frame indices
    n_frames = len(positions)
    time = np.arange(n_frames) * 0.1  # Assume ~10Hz (0.1s per frame)
    
    # Calculate velocity from position differences
    if n_frames > 1:
        velocity = np.diff(positions, axis=0) / np.diff(time).reshape(-1, 1)
        # Pad with last velocity
        velocity = np.vstack([velocity, velocity[-1:]])
    else:
        velocity = np.zeros((1, 3))
    
    data = {
        'time': time,
        'position': positions,
        'velocity': velocity,
        'rot_ang': np.zeros((n_frames, 3)),  # Not available in KITTI format
        'gyro_bias': np.zeros((n_frames, 3)),  # Not available
        'accel_bias': np.zeros((n_frames, 3)),  # Not available
        'gravity': np.zeros((n_frames, 3))  # Not available
    }
    
    return data


def detect_format(filename):
    """Detect file format by reading first valid line"""
    with open(filename, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) == 12:
                return 'kitti'
            elif len(values) == 25:
                return 'pos_log'
    return None


def parse_odometry_log(filename):
    """Parse odometry log file - auto-detect format"""
    format_type = detect_format(filename)
    
    if format_type == 'kitti':
        return parse_kitti_format(filename)
    elif format_type == 'pos_log':
        return parse_pos_log_format(filename)
    else:
        raise ValueError(f"Could not detect format for {filename}. Expected 12 (KITTI) or 25 (pos_log) values per line.")


def plot_multi_trajectory_interactive(data1, data2, label1="Trajectory 1", label2="Trajectory 2", 
                                      output_file='odometry_multi_interactive.html', color_by='time'):
    """
    Create an interactive 3D visualization of two odometry trajectories using Plotly.
    
    Parameters:
    -----------
    data1 : dict
        Dictionary containing parsed odometry data for first trajectory
    data2 : dict
        Dictionary containing parsed odometry data for second trajectory
    label1 : str
        Label for first trajectory
    label2 : str
        Label for second trajectory
    output_file : str
        Output HTML filename
    color_by : str
        Color coding option: 'time', 'speed', 'altitude', or 'index'
    """
    if not PLOTLY_AVAILABLE:
        print("Skipping interactive visualization (Plotly not available)")
        return
    
    # Extract coordinates for both trajectories
    x1, y1, z1 = data1['position'][:, 0], data1['position'][:, 1], data1['position'][:, 2]
    x2, y2, z2 = data2['position'][:, 0], data2['position'][:, 1], data2['position'][:, 2]
    
    # Calculate speed for color coding
    speed1 = np.linalg.norm(data1['velocity'], axis=1)
    speed2 = np.linalg.norm(data2['velocity'], axis=1)
    
    # Determine color values based on option
    if color_by == 'time':
        color1_values = data1['time']
        color2_values = data2['time']
        color_title = "Time (s)"
        colorscale1 = 'viridis'
        colorscale2 = 'plasma'
    elif color_by == 'speed':
        color1_values = speed1
        color2_values = speed2
        color_title = "Speed (m/s)"
        colorscale1 = 'plasma'
        colorscale2 = 'cividis'
    elif color_by == 'altitude':
        color1_values = data1['position'][:, 2]
        color2_values = data2['position'][:, 2]
        color_title = "Altitude Z (m)"
        colorscale1 = 'terrain'
        colorscale2 = 'turbo'
    else:  # 'index'
        color1_values = list(range(len(data1['time'])))
        color2_values = list(range(len(data2['time'])))
        color_title = "Frame Index"
        colorscale1 = 'viridis'
        colorscale2 = 'plasma'
    
    # Create hover text for both trajectories
    hover_text1 = []
    for i in range(len(data1['time'])):
        hover_info = (
            f"<b>{label1} - Frame {i}</b><br>"
            f"Time: {data1['time'][i]:.3f} s<br>"
            f"Position: ({data1['position'][i, 0]:.3f}, {data1['position'][i, 1]:.3f}, {data1['position'][i, 2]:.3f}) m<br>"
            f"Velocity: ({data1['velocity'][i, 0]:.3f}, {data1['velocity'][i, 1]:.3f}, {data1['velocity'][i, 2]:.3f}) m/s<br>"
            f"Speed: {speed1[i]:.3f} m/s"
        )
        hover_text1.append(hover_info)
    
    hover_text2 = []
    for i in range(len(data2['time'])):
        hover_info = (
            f"<b>{label2} - Frame {i}</b><br>"
            f"Time: {data2['time'][i]:.3f} s<br>"
            f"Position: ({data2['position'][i, 0]:.3f}, {data2['position'][i, 1]:.3f}, {data2['position'][i, 2]:.3f}) m<br>"
            f"Velocity: ({data2['velocity'][i, 0]:.3f}, {data2['velocity'][i, 1]:.3f}, {data2['velocity'][i, 2]:.3f}) m/s<br>"
            f"Speed: {speed2[i]:.3f} m/s"
        )
        hover_text2.append(hover_info)
    
    # Create 3D scatter plots for both trajectories
    scatter1 = go.Scatter3d(
        x=x1,
        y=y1,
        z=z1,
        mode='markers+lines',
        marker=dict(
            size=3,
            color=color1_values,
            colorscale=colorscale1,
            opacity=0.8,
            colorbar=dict(
                title=dict(text=f"{label1} - {color_title}", font=dict(size=12)),
                tickfont=dict(size=10),
                x=1.02,
                len=0.4,
                y=0.75
            ),
            showscale=True
        ),
        line=dict(
            color=color1_values,
            colorscale=colorscale1,
            width=3,
            showscale=False
        ),
        text=hover_text1,
        hovertemplate='%{text}<extra></extra>',
        name=label1,
        legendgroup=label1
    )
    
    scatter2 = go.Scatter3d(
        x=x2,
        y=y2,
        z=z2,
        mode='markers+lines',
        marker=dict(
            size=3,
            color=color2_values,
            colorscale=colorscale2,
            opacity=0.8,
            colorbar=dict(
                title=dict(text=f"{label2} - {color_title}", font=dict(size=12)),
                tickfont=dict(size=10),
                x=1.02,
                len=0.4,
                y=0.25
            ),
            showscale=True
        ),
        line=dict(
            color=color2_values,
            colorscale=colorscale2,
            width=3,
            showscale=False
        ),
        text=hover_text2,
        hovertemplate='%{text}<extra></extra>',
        name=label2,
        legendgroup=label2
    )
    
    # Add start and end points for trajectory 1
    start1 = go.Scatter3d(
        x=[x1[0]],
        y=[y1[0]],
        z=[z1[0]],
        mode='markers',
        marker=dict(
            size=12,
            color='green',
            symbol='circle',
            line=dict(width=2, color='darkgreen')
        ),
        text=[f"<b>{label1} START</b><br>{hover_text1[0]}"],
        hovertemplate='%{text}<extra></extra>',
        name=f'{label1} Start',
        legendgroup=label1,
        showlegend=True
    )
    
    end1 = go.Scatter3d(
        x=[x1[-1]],
        y=[y1[-1]],
        z=[z1[-1]],
        mode='markers',
        marker=dict(
            size=12,
            color='red',
            symbol='square',
            line=dict(width=2, color='darkred')
        ),
        text=[f"<b>{label1} END</b><br>{hover_text1[-1]}"],
        hovertemplate='%{text}<extra></extra>',
        name=f'{label1} End',
        legendgroup=label1,
        showlegend=True
    )
    
    # Add start and end points for trajectory 2
    start2 = go.Scatter3d(
        x=[x2[0]],
        y=[y2[0]],
        z=[z2[0]],
        mode='markers',
        marker=dict(
            size=12,
            color='lime',
            symbol='diamond',
            line=dict(width=2, color='darkgreen')
        ),
        text=[f"<b>{label2} START</b><br>{hover_text2[0]}"],
        hovertemplate='%{text}<extra></extra>',
        name=f'{label2} Start',
        legendgroup=label2,
        showlegend=True
    )
    
    end2 = go.Scatter3d(
        x=[x2[-1]],
        y=[y2[-1]],
        z=[z2[-1]],
        mode='markers',
        marker=dict(
            size=12,
            color='orange',
            symbol='diamond',
            line=dict(width=2, color='darkred')
        ),
        text=[f"<b>{label2} END</b><br>{hover_text2[-1]}"],
        hovertemplate='%{text}<extra></extra>',
        name=f'{label2} End',
        legendgroup=label2,
        showlegend=True
    )
    
    # Create figure with all trajectories
    fig = go.Figure(data=[scatter1, scatter2, start1, end1, start2, end2])
    
    # Calculate statistics for title
    total_distance1 = np.sum(np.linalg.norm(np.diff(data1['position'], axis=0), axis=1))
    total_distance2 = np.sum(np.linalg.norm(np.diff(data2['position'], axis=0), axis=1))
    avg_speed1 = np.mean(speed1)
    avg_speed2 = np.mean(speed2)
    
    # Calculate position difference at end
    end_diff = np.linalg.norm(data1['position'][-1] - data2['position'][-1])
    
    fig.update_layout(
        title=dict(
            text=f'Interactive 3D Trajectory Comparison: {label1} vs {label2}<br>'
                 f'<sub>{label1}: {total_distance1:.2f} m, {avg_speed1:.2f} m/s avg | '
                 f'{label2}: {total_distance2:.2f} m, {avg_speed2:.2f} m/s avg | '
                 f'End Position Diff: {end_diff:.3f} m</sub>',
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text='X (meters)', font=dict(size=12)),
                backgroundcolor='rgb(230, 230, 230)',
                gridcolor='white',
                showbackground=True
            ),
            yaxis=dict(
                title=dict(text='Y (meters)', font=dict(size=12)),
                backgroundcolor='rgb(230, 230, 230)',
                gridcolor='white',
                showbackground=True
            ),
            zaxis=dict(
                title=dict(text='Z (meters)', font=dict(size=12)),
                backgroundcolor='rgb(230, 230, 230)',
                gridcolor='white',
                showbackground=True
            ),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        ),
        width=1400,
        height=1000,
        margin=dict(l=0, r=150, t=100, b=0),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=11)
        )
    )
    
    # Save as HTML
    fig.write_html(output_file)
    print(f"\nInteractive 3D comparison plot saved to {output_file}")
    print(f"{label1}: {len(data1['time'])} pose points")
    print(f"{label2}: {len(data2['time'])} pose points")
    print(f"X range: {min(np.min(x1), np.min(x2)):.2f} to {max(np.max(x1), np.max(x2)):.2f} m")
    print(f"Y range: {min(np.min(y1), np.min(y2)):.2f} to {max(np.max(y1), np.max(y2)):.2f} m")
    print(f"Z range: {min(np.min(z1), np.min(z2)):.2f} to {max(np.max(z1), np.max(z2)):.2f} m")
    print(f"Color coding: {color_by}")
    print(f"End position difference: {end_diff:.3f} m")
    print(f"Open {output_file} in a web browser to view the interactive plot")


def print_comparison_statistics(data1, data2, label1="Trajectory 1", label2="Trajectory 2"):
    """Print comparison statistics between two trajectories"""
    print("\n" + "="*60)
    print("TRAJECTORY COMPARISON STATISTICS")
    print("="*60)
    
    # Check if data is empty
    if len(data1['time']) == 0:
        print(f"\nERROR: {label1} has no valid data!")
        return
    if len(data2['time']) == 0:
        print(f"\nERROR: {label2} has no valid data!")
        return
    
    # Duration
    print(f"\n--- Duration ---")
    print(f"{label1}: {data1['time'][-1]:.2f} seconds ({len(data1['time'])} frames)")
    print(f"{label2}: {data2['time'][-1]:.2f} seconds ({len(data2['time'])} frames)")
    
    # Position
    print(f"\n--- Start Position (meters) ---")
    print(f"{label1}: [{data1['position'][0, 0]:.3f}, {data1['position'][0, 1]:.3f}, {data1['position'][0, 2]:.3f}]")
    print(f"{label2}: [{data2['position'][0, 0]:.3f}, {data2['position'][0, 1]:.3f}, {data2['position'][0, 2]:.3f}]")
    start_diff = np.linalg.norm(data1['position'][0] - data2['position'][0])
    print(f"Difference: {start_diff:.3f} m")
    
    print(f"\n--- End Position (meters) ---")
    print(f"{label1}: [{data1['position'][-1, 0]:.3f}, {data1['position'][-1, 1]:.3f}, {data1['position'][-1, 2]:.3f}]")
    print(f"{label2}: [{data2['position'][-1, 0]:.3f}, {data2['position'][-1, 1]:.3f}, {data2['position'][-1, 2]:.3f}]")
    end_diff = np.linalg.norm(data1['position'][-1] - data2['position'][-1])
    print(f"Difference: {end_diff:.3f} m")
    
    # Distance traveled
    displacement1 = data1['position'][-1] - data1['position'][0]
    displacement2 = data2['position'][-1] - data2['position'][0]
    total_distance1 = np.sum(np.linalg.norm(np.diff(data1['position'], axis=0), axis=1))
    total_distance2 = np.sum(np.linalg.norm(np.diff(data2['position'], axis=0), axis=1))
    
    print(f"\n--- Distance Traveled ---")
    print(f"{label1}: {total_distance1:.3f} m (displacement: {np.linalg.norm(displacement1):.3f} m)")
    print(f"{label2}: {total_distance2:.3f} m (displacement: {np.linalg.norm(displacement2):.3f} m)")
    print(f"Difference: {abs(total_distance1 - total_distance2):.3f} m")
    
    # Speed
    speed1 = np.linalg.norm(data1['velocity'], axis=1)
    speed2 = np.linalg.norm(data2['velocity'], axis=1)
    print(f"\n--- Speed (m/s) ---")
    print(f"{label1}: Avg={np.mean(speed1):.3f}, Max={np.max(speed1):.3f}")
    print(f"{label2}: Avg={np.mean(speed2):.3f}, Max={np.max(speed2):.3f}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Parse arguments
    if len(sys.argv) < 3:
        print("Usage: python3 viz_pose_txt_multi.py <pose_file1.txt> <pose_file2.txt> [output_file.html] [label1] [label2]")
        print("\nExample:")
        print("  python3 viz_pose_txt_multi.py pos_log.txt pos_log_optimized.txt comparison.html \"Original\" \"Optimized\"")
        print("\nSupports both formats:")
        print("  - KITTI format (12 values): R00 R01 R02 tx R10 R11 R12 ty R20 R21 R22 tz")
        print("  - pos_log.txt format (25 values): time rot_ang[3] position[3] ...")
        sys.exit(1)
    
    filepath1 = sys.argv[1]
    filepath2 = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else 'odometry_multi_interactive.html'
    label1 = sys.argv[4] if len(sys.argv) > 4 else "Trajectory 1"
    label2 = sys.argv[5] if len(sys.argv) > 5 else "Trajectory 2"
    
    # Check if files exist
    if not os.path.exists(filepath1):
        print(f"Error: File not found: {filepath1}")
        sys.exit(1)
    
    if not os.path.exists(filepath2):
        print(f"Error: File not found: {filepath2}")
        sys.exit(1)
    
    print("="*60)
    print("Multi-Trajectory Visualization")
    print("="*60)
    print(f"Trajectory 1: {filepath1} ({label1})")
    print(f"Trajectory 2: {filepath2} ({label2})")
    print(f"Output: {output_file}")
    print("="*60)
    
    # Parse both files
    print(f"\nParsing {filepath1}...")
    try:
        data1 = parse_odometry_log(filepath1)
        format1 = detect_format(filepath1)
        print(f"  Format: {format1.upper()}, Found {len(data1['time'])} frames")
    except Exception as e:
        print(f"ERROR: Failed to parse {filepath1}: {e}")
        sys.exit(1)
    
    if len(data1['time']) == 0:
        print(f"ERROR: No valid data found in {filepath1}")
        sys.exit(1)
    
    print(f"Parsing {filepath2}...")
    try:
        data2 = parse_odometry_log(filepath2)
        format2 = detect_format(filepath2)
        print(f"  Format: {format2.upper()}, Found {len(data2['time'])} frames")
    except Exception as e:
        print(f"ERROR: Failed to parse {filepath2}: {e}")
        sys.exit(1)
    
    if len(data2['time']) == 0:
        print(f"ERROR: No valid data found in {filepath2}")
        sys.exit(1)
    
    # Print comparison statistics
    print_comparison_statistics(data1, data2, label1, label2)
    
    # Create interactive visualization
    plot_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
    if plot_dir and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    plot_multi_trajectory_interactive(data1, data2, label1, label2, output_file, color_by='time')
    
    # Also create a speed-colored version
    if len(output_file.split('.')) > 1:
        base_name = '.'.join(output_file.split('.')[:-1])
        ext = output_file.split('.')[-1]
        speed_output = f"{base_name}_speed.{ext}"
    else:
        speed_output = f"{output_file}_speed.html"
    
    plot_multi_trajectory_interactive(data1, data2, label1, label2, speed_output, color_by='speed')
    
    print("\nDone!")

