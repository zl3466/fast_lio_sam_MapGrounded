#!/usr/bin/env python3
"""
Parser for FAST_LIO odometry_log.txt files

Format: Each line contains 25 space-separated values:
1. time_offset (seconds) - Time relative to first LiDAR frame
2-4. rot_ang[3] (radians) - Rotation angles (log of rotation matrix), typically small values
5-7. position[3] (meters) - Position [x, y, z] in world frame
8-10. omega[3] - Angular velocity (always 0.0, not used)
11-13. velocity[3] (m/s) - Linear velocity [vx, vy, vz]
14-16. acc[3] - Acceleration (always 0.0, not used)
17-19. gyro_bias[3] (rad/s) - Gyroscope bias [bx, by, bz]
20-22. accel_bias[3] (m/s^2) - Accelerometer bias [ax, ay, az]
23-25. gravity[3] (m/s^2) - Gravity vector [gx, gy, gz]
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import plotly.graph_objects as go
PLOTLY_AVAILABLE = True


def parse_odometry_log(filename):
    """Parse odometry log file and return structured data"""
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

def plot_trajectory(data, save_plot=True, output_file='trajectory_plot.png'):
    """Plot 3D trajectory and other useful visualizations"""
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 3D Trajectory
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(data['position'][:, 0], data['position'][:, 1], data['position'][:, 2], 'b-', linewidth=1)
    ax1.scatter(data['position'][0, 0], data['position'][0, 1], data['position'][0, 2], 
                color='green', s=100, label='Start', marker='o')
    ax1.scatter(data['position'][-1, 0], data['position'][-1, 1], data['position'][-1, 2], 
                color='red', s=100, label='End', marker='s')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 2D Trajectory (Top View - XY plane)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(data['position'][:, 0], data['position'][:, 1], 'b-', linewidth=1)
    ax2.scatter(data['position'][0, 0], data['position'][0, 1], 
                color='green', s=100, label='Start', marker='o', zorder=5)
    ax2.scatter(data['position'][-1, 0], data['position'][-1, 1], 
                color='red', s=100, label='End', marker='s', zorder=5)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Trajectory (Top View)')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # 3. Position over time
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(data['time'], data['position'][:, 0], label='X', linewidth=1)
    ax3.plot(data['time'], data['position'][:, 1], label='Y', linewidth=1)
    ax3.plot(data['time'], data['position'][:, 2], label='Z', linewidth=1)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('Position vs Time')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Velocity over time
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(data['time'], data['velocity'][:, 0], label='Vx', linewidth=1)
    ax4.plot(data['time'], data['velocity'][:, 1], label='Vy', linewidth=1)
    ax4.plot(data['time'], data['velocity'][:, 2], label='Vz', linewidth=1)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.set_title('Velocity vs Time')
    ax4.legend()
    ax4.grid(True)
    
    # 5. Speed (magnitude of velocity)
    ax5 = fig.add_subplot(2, 3, 5)
    speed = np.linalg.norm(data['velocity'], axis=1)
    ax5.plot(data['time'], speed, 'b-', linewidth=1)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Speed (m/s)')
    ax5.set_title('Speed vs Time')
    ax5.grid(True)
    
    # 6. Altitude (Z position)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(data['time'], data['position'][:, 2], 'b-', linewidth=1)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Altitude Z (m)')
    ax6.set_title('Altitude vs Time')
    ax6.grid(True)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def print_statistics(data):
    """Print statistics about the trajectory"""
    print("\n" + "="*60)
    print("ODOMETRY LOG STATISTICS")
    print("="*60)
    
    print(f"\nTotal duration: {data['time'][-1]:.2f} seconds")
    print(f"Number of frames: {len(data['time'])}")
    print(f"Average frame rate: {len(data['time'])/data['time'][-1]:.2f} Hz")
    
    print(f"\n--- Position (meters) ---")
    print(f"Start position: [{data['position'][0, 0]:.3f}, {data['position'][0, 1]:.3f}, {data['position'][0, 2]:.3f}]")
    print(f"End position:   [{data['position'][-1, 0]:.3f}, {data['position'][-1, 1]:.3f}, {data['position'][-1, 2]:.3f}]")
    
    displacement = data['position'][-1] - data['position'][0]
    total_distance = np.sum(np.linalg.norm(np.diff(data['position'], axis=0), axis=1))
    print(f"Total displacement: {np.linalg.norm(displacement):.3f} m")
    print(f"Total distance traveled: {total_distance:.3f} m")
    
    print(f"\n--- Velocity (m/s) ---")
    print(f"Average velocity: [{np.mean(data['velocity'][:, 0]):.3f}, {np.mean(data['velocity'][:, 1]):.3f}, {np.mean(data['velocity'][:, 2]):.3f}]")
    print(f"Max speed: {np.max(np.linalg.norm(data['velocity'], axis=1)):.3f} m/s")
    print(f"Average speed: {np.mean(np.linalg.norm(data['velocity'], axis=1)):.3f} m/s")
    
    print(f"\n--- Altitude ---")
    print(f"Min altitude (Z): {np.min(data['position'][:, 2]):.3f} m")
    print(f"Max altitude (Z): {np.max(data['position'][:, 2]):.3f} m")
    print(f"Altitude range: {np.max(data['position'][:, 2]) - np.min(data['position'][:, 2]):.3f} m")
    
    print(f"\n--- Gravity vector (m/s²) ---")
    print(f"Average gravity: [{np.mean(data['gravity'][:, 0]):.3f}, {np.mean(data['gravity'][:, 1]):.3f}, {np.mean(data['gravity'][:, 2]):.3f}]")
    print(f"Gravity magnitude: {np.mean(np.linalg.norm(data['gravity'], axis=1)):.3f} m/s²")
    
    print("\n" + "="*60 + "\n")

def export_to_tum_format(data, output_file='trajectory_tum.txt'):
    """Export trajectory to TUM format (timestamp x y z qx qy qz qw)"""
    # Note: We don't have quaternion, so we'll use identity quaternion
    # rot_ang is the log of rotation matrix, not directly usable as quaternion
    with open(output_file, 'w') as f:
        for i in range(len(data['time'])):
            # Assuming identity rotation (quaternion [0, 0, 0, 1])
            # For proper conversion, you'd need to convert rot_ang back to quaternion
            f.write(f"{data['time'][i]:.6f} "
                   f"{data['position'][i, 0]:.6f} {data['position'][i, 1]:.6f} {data['position'][i, 2]:.6f} "
                   f"0.0 0.0 0.0 1.0\n")
    print(f"TUM format trajectory exported to {output_file} (Note: quaternion is identity)")

def plot_odometry_interactive(data, output_file='odometry_interactive.html', color_by='time'):
    """
    Create an interactive 3D visualization of the odometry trajectory using Plotly.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing parsed odometry data
    output_file : str
        Output HTML filename
    color_by : str
        Color coding option: 'time', 'speed', 'altitude', or 'index'
    """
    if not PLOTLY_AVAILABLE:
        print("Skipping interactive visualization (Plotly not available)")
        return
    
    x_coords = data['position'][:, 0]
    y_coords = data['position'][:, 1]
    z_coords = data['position'][:, 2]
    
    # Calculate speed for color coding
    speed = np.linalg.norm(data['velocity'], axis=1)
    
    # Determine color values based on option
    if color_by == 'time':
        color_values = data['time']
        color_title = "Time (s)"
        colorscale = 'viridis'
    elif color_by == 'speed':
        color_values = speed
        color_title = "Speed (m/s)"
        colorscale = 'plasma'
    elif color_by == 'altitude':
        color_values = data['position'][:, 2]
        color_title = "Altitude Z (m)"
        colorscale = 'terrain'
    else:  # 'index'
        color_values = list(range(len(data['time'])))
        color_title = "Frame Index"
        colorscale = 'viridis'
    
    # Create hover text with detailed information
    hover_text = []
    for i in range(len(data['time'])):
        hover_info = (
            f"<b>Frame {i}</b><br>"
            f"Time: {data['time'][i]:.3f} s<br>"
            f"Position: ({data['position'][i, 0]:.3f}, {data['position'][i, 1]:.3f}, {data['position'][i, 2]:.3f}) m<br>"
            f"Velocity: ({data['velocity'][i, 0]:.3f}, {data['velocity'][i, 1]:.3f}, {data['velocity'][i, 2]:.3f}) m/s<br>"
            f"Speed: {speed[i]:.3f} m/s<br>"
            f"Gyro Bias: ({data['gyro_bias'][i, 0]:.6f}, {data['gyro_bias'][i, 1]:.6f}, {data['gyro_bias'][i, 2]:.6f}) rad/s<br>"
            f"Accel Bias: ({data['accel_bias'][i, 0]:.6f}, {data['accel_bias'][i, 1]:.6f}, {data['accel_bias'][i, 2]:.6f}) m/s²<br>"
            f"Gravity: ({data['gravity'][i, 0]:.3f}, {data['gravity'][i, 1]:.3f}, {data['gravity'][i, 2]:.3f}) m/s²"
        )
        hover_text.append(hover_info)
    
    # Create 3D scatter plot for trajectory points
    scatter = go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='markers+lines',
        marker=dict(
            size=3,
            color=color_values,
            colorscale=colorscale,
            opacity=0.8,
            colorbar=dict(
                title=dict(text=color_title, font=dict(size=14)),
                tickfont=dict(size=12)
            ),
            showscale=True
        ),
        line=dict(
            color=color_values,
            colorscale=colorscale,
            width=2,
            showscale=False
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        name='Trajectory'
    )
    
    # Add start and end points
    start_point = go.Scatter3d(
        x=[x_coords[0]],
        y=[y_coords[0]],
        z=[z_coords[0]],
        mode='markers',
        marker=dict(
            size=10,
            color='green',
            symbol='circle',
            line=dict(width=2, color='darkgreen')
        ),
        text=[f"<b>START</b><br>{hover_text[0]}"],
        hovertemplate='%{text}<extra></extra>',
        name='Start'
    )
    
    end_point = go.Scatter3d(
        x=[x_coords[-1]],
        y=[y_coords[-1]],
        z=[z_coords[-1]],
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            symbol='square',
            line=dict(width=2, color='darkred')
        ),
        text=[f"<b>END</b><br>{hover_text[-1]}"],
        hovertemplate='%{text}<extra></extra>',
        name='End'
    )
    
    # Create figure
    fig = go.Figure(data=[scatter, start_point, end_point])
    
    # Calculate statistics for title
    total_distance = np.sum(np.linalg.norm(np.diff(data['position'], axis=0), axis=1))
    avg_speed = np.mean(speed)
    max_speed = np.max(speed)
    
    fig.update_layout(
        title=dict(
            text=f'Interactive 3D FAST_LIO Odometry Trajectory<br>'
                 f'<sub>Distance: {total_distance:.2f} m | Avg Speed: {avg_speed:.2f} m/s | Max Speed: {max_speed:.2f} m/s | Duration: {data["time"][-1]:.2f} s</sub>',
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
        width=1200,
        height=900,
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    # Save as HTML
    fig.write_html(output_file)
    print(f"\nInteractive 3D plot saved to {output_file}")
    print(f"Plotted {len(data['time'])} pose points")
    print(f"X range: {np.min(x_coords):.2f} to {np.max(x_coords):.2f} m")
    print(f"Y range: {np.min(y_coords):.2f} to {np.max(y_coords):.2f} m")
    print(f"Z range: {np.min(z_coords):.2f} to {np.max(z_coords):.2f} m")
    print(f"Color coding: {color_by}")
    print(f"Open {output_file} in a web browser to view the interactive plot")

if __name__ == "__main__":
    # Default filename
    filepath = '/home/zl3466/Documents/ros1/fast_lio_sam/output/scene_2_short/pos_log_optimized.txt'
    # if len(sys.argv) > 1:
    #     filename = sys.argv[1]
    
    # # Get the script directory
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # filepath = os.path.join(script_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        print(f"Usage: python3 {sys.argv[0]} [path/to/odometry_log.txt]")
        sys.exit(1)
    
    print(f"Parsing {filepath}...")
    data = parse_odometry_log(filepath)
    
    print_statistics(data)
    
    # Plot trajectory (static matplotlib plots)
    plot_dir = os.path.dirname(filepath)
    plot_output = os.path.join(plot_dir, 'trajectory_plot.png')
    plot_trajectory(data, save_plot=True, output_file=plot_output)
    
    # Create interactive 3D visualization (Plotly)
    interactive_output = os.path.join(plot_dir, 'odometry_interactive.html')
    plot_odometry_interactive(data, output_file=interactive_output, color_by='time')
    
    # Also create a speed-colored version
    interactive_speed = os.path.join(plot_dir, 'odometry_interactive_speed.html')
    plot_odometry_interactive(data, output_file=interactive_speed, color_by='speed')
    
    # Export to TUM format (optional)
    tum_output = os.path.join(plot_dir, 'trajectory_tum.txt')
    export_to_tum_format(data, output_file=tum_output)
    
    print("\nDone!")

