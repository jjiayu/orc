import time
import json
import os

# import orc.optimal_control.lipm.biped.romeo_conf as conf
import talos_conf as conf
import matplotlib.pyplot as plt
import numpy as np
import orc.utils.plot_utils as plut
from numpy import nan
from numpy.linalg import norm as norm
from tsid_biped import TsidBiped
from orc.optimal_control.lipm.biped.lipm_to_tsid import compute_5th_order_poly_traj, generate_square_foot_trajectory

import tsid

print("".center(conf.LINE_WIDTH, "#"))
print(" Test Walking ".center(conf.LINE_WIDTH, "#"))
print("".center(conf.LINE_WIDTH, "#"), "\n")

CoM_Height_Offset = 0.6  # Height offset above the stance foot
step_height = 0.1

global q_temp
q_temp = None

USE_EIQUADPROG = 1
USE_PROXQP = 0
USE_OSQP = 0
VERBOSE = 0

PLOT_COM = 0
PLOT_COP = 0
PLOT_FOOT_TRAJ = 0
PLOT_TORQUES = 0
PLOT_JOINT_VEL = 0

# Configuration for loading footsteps from JSON file
USE_JSON_FOOTSTEPS = True
JSON_FOOTSTEP_FILE = "footsteps_output.json"

tsid_biped = TsidBiped(conf, conf.viewer)

# overwrite the default solver
if USE_EIQUADPROG:
    print("Using eiquadprog")
    tsid_biped.solver = tsid.SolverHQuadProgFast("qp solver")

tsid_biped.solver.resize(
    tsid_biped.formulation.nVar, tsid_biped.formulation.nEq, tsid_biped.formulation.nIn
)

#display the robot
q, v = tsid_biped.q, tsid_biped.v
tsid_biped.display(q)
time.sleep(1.0)

# ============================================================================
# FOOTSTEP VISUALIZATION IN MESHCAT
# ============================================================================

def visualize_footsteps_in_meshcat(tsid_biped, footstep_targets, loaded_footsteps=None):
    """Visualize footsteps as boxes in meshcat"""
    import meshcat.geometry as g
    import meshcat.transformations as tf
    
    # Footstep dimensions
    foot_length = 0.22  # x dimension
    foot_width = 0.12   # y dimension
    foot_height = 0.01  # z dimension (thin box)
    
    # Colors for left and right feet
    left_color = 0x0000ff  # Blue
    right_color = 0xff0000  # Red
    
    if hasattr(tsid_biped, 'viz') and tsid_biped.viz is not None:
        print("Visualizing footsteps in meshcat...")
        
        for i, target in enumerate(footstep_targets):
            # Determine foot type
            if loaded_footsteps is not None:
                # For loaded footsteps, we need to map back to the original footstep data
                # The footstep_targets array corresponds to loaded_footsteps[i] (all footsteps including stance)
                original_index = i  # Direct mapping since footstep_targets now includes all positions
                if original_index < len(loaded_footsteps):
                    foot_type = loaded_footsteps[original_index]['foot']
                    yaw = loaded_footsteps[original_index].get('yaw', 0.0)
                else:
                    foot_type = "left"  # fallback
                    yaw = 0.0
            else:
                # For hardcoded footsteps, alternate based on first_swing_foot
                if first_swing_foot == "left":
                    foot_type = "left" if i % 2 == 0 else "right"
                else:
                    foot_type = "right" if i % 2 == 0 else "left"
                yaw = 0.0
            
            # Choose color
            color = left_color if foot_type == "left" else right_color
            
            # Create box geometry
            box = g.Box([foot_length, foot_width, foot_height])
            material = g.MeshLambertMaterial(color=color, opacity=0.7)
            
            # Create transformation matrix
            position = target.copy()
            position[2] += foot_height / 2  # Raise box slightly above ground
            
            # Apply yaw rotation
            transform = tf.translation_matrix(position)
            if yaw != 0.0:
                rotation = tf.rotation_matrix(yaw, [0, 0, 1])
                transform = transform @ rotation
            
            # Add to meshcat
            footstep_name = f"footsteps/step_{i:02d}_{foot_type}"
            tsid_biped.viz.viewer[footstep_name].set_object(box, material)
            tsid_biped.viz.viewer[footstep_name].set_transform(transform)
        
        print(f"Added {len(footstep_targets)} footstep visualizations to meshcat")
        print("Blue boxes = left foot, Red boxes = right foot")
    else:
        print("Meshcat visualizer not available, skipping footstep visualization")

# ============================================================================
# ENVIRONMENT VISUALIZATION FROM ENV FILE
# ============================================================================

def load_and_visualize_environment(tsid_biped, env_file="threepathnas"):
    """Load environment from env file and visualize surfaces in meshcat"""
    import meshcat.geometry as g
    import meshcat.transformations as tf
    import sys
    import os
    
    if hasattr(tsid_biped, 'viz') and tsid_biped.viz is not None:
        try:
            # Add env folder to path
            env_path = os.path.join(os.path.dirname(__file__), '..', 'env')
            sys.path.insert(0, env_path)
            
            # Import the environment module
            env_module = __import__(env_file)
            
            # Get the scene from the environment
            scene = env_module.scene
            
            print(f"Loading environment from {env_file}.py...")
            print(f"Found {len(scene)} surface groups in scene")
            
            # Colors for different surface types
            colors = [0x90EE90, 0x87CEEB, 0xDDA0DD, 0xF0E68C, 0xFFA07A]  # Different colors
            
            surface_count = 0
            for group_idx, surface_group in enumerate(scene):
                color = colors[group_idx % len(colors)]
                material = g.MeshLambertMaterial(color=color, opacity=0.8)
                
                for surface_idx, surface in enumerate(surface_group):
                    # Surface is a 3x4 numpy array: [x_coords, y_coords, z_coords]
                    x_coords = surface[0]
                    y_coords = surface[1] 
                    z_coords = surface[2]
                    
                    # Calculate surface dimensions and center
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    z_min, z_max = min(z_coords), max(z_coords)
                    
                    width = x_max - x_min
                    depth = y_max - y_min
                    height = max(0.05, z_max - z_min)  # Minimum thickness for visibility
                    
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2
                    center_z = (z_min + z_max) / 2 - height / 2  # Shift down so top reaches z=0
                    
                    # Create box geometry
                    box = g.Box([width, depth, height])
                    
                    # Position the box
                    transform = tf.translation_matrix([center_x, center_y, center_z])
                    
                    # Add to meshcat
                    surface_name = f"environment/surface_{surface_count:02d}"
                    tsid_biped.viz.viewer[surface_name].set_object(box, material)
                    tsid_biped.viz.viewer[surface_name].set_transform(transform)
                    
                    print(f"  Surface {surface_count}: [{center_x:.2f}, {center_y:.2f}, {center_z:.2f}] size: [{width:.2f}x{depth:.2f}x{height:.2f}]")
                    surface_count += 1
            
            print(f"Added {surface_count} environment surfaces to meshcat")
            
        except Exception as e:
            print(f"Error loading environment {env_file}: {e}")
            print("Falling back to default visualization")
    else:
        print("Meshcat visualizer not available, skipping environment visualization")

# Load and visualize the narrow passage environment
load_and_visualize_environment(tsid_biped, "narrow_passage")

# ============================================================================
# FOOTSTEP LOADING AND VISUALIZATION
# ============================================================================

def load_footsteps_from_json(filename):
    """Load footsteps from JSON file"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, filename)
    
    print(f"Looking for JSON file at: {full_path}")
    print(f"Current working directory: {os.getcwd()}")
    
    if not os.path.exists(full_path):
        print(f"Warning: JSON file {full_path} not found. Using hardcoded footsteps.")
        return None
    
    try:
        with open(full_path, 'r') as f:
            data = json.load(f)
        
        footsteps = []
        for step in data['footsteps']:
            footsteps.append({
                'position': np.array(step['position']),
                'foot': step['foot'],
                'timing': step.get('timing', 0.0),
                'yaw': step.get('yaw', 0.0)
            })
        
        print(f"Loaded {len(footsteps)} footsteps from {filename}")
        return footsteps
    
    except Exception as e:
        print(f"Error loading JSON file {filename}: {e}")
        print("Using hardcoded footsteps instead.")
        return None

# Load footsteps from JSON
loaded_footsteps = load_footsteps_from_json(JSON_FOOTSTEP_FILE)

# Visualize the footsteps (show all footsteps including initial stance position)
if loaded_footsteps is not None:
    # For visualization, show all footsteps including the initial stance position
    all_footstep_positions = [step['position'] for step in loaded_footsteps]
    visualize_footsteps_in_meshcat(tsid_biped, all_footstep_positions, loaded_footsteps)
    
    print(f"\nFootstep visualization:")
    print(f"  Total footsteps: {len(loaded_footsteps)}")
    print(f"  Initial stance: {loaded_footsteps[0]['foot']} foot at [{loaded_footsteps[0]['position'][0]:.3f}, {loaded_footsteps[0]['position'][1]:.3f}, {loaded_footsteps[0]['position'][2]:.3f}]")
    print("  Blue boxes = left foot, Red boxes = right foot")
else:
    print("No footsteps loaded - skipping footstep visualization")

# ============================================================================
# LOAD AND DISPLAY WHOLE-BODY CONFIGURATION FROM SAVED DATA
# ============================================================================

def load_simulation_data(data_folder="../../../simulation_data"):
    """Load the saved simulation data"""
    try:
        # Get absolute path to data folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, data_folder)
        
        print(f"Loading simulation data from: {data_path}")
        
        # Load the data files
        q_log = np.load(os.path.join(data_path, "q_log.npy"))
        v_log = np.load(os.path.join(data_path, "v_log.npy"))
        tau_log = np.load(os.path.join(data_path, "tau_log.npy"))
        time_log = np.load(os.path.join(data_path, "time_log.npy"))
        
        # Load metadata
        with open(os.path.join(data_path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        print(f"Loaded simulation data:")
        print(f"  Joint positions: {q_log.shape}")
        print(f"  Joint velocities: {v_log.shape}")
        print(f"  Joint torques: {tau_log.shape}")
        print(f"  Time array: {time_log.shape}")
        print(f"  Duration: {metadata['total_duration']:.1f}s")
        print(f"  Time step: {metadata['dt']:.3f}s")
        
        return {
            'q_log': q_log,
            'v_log': v_log,
            'tau_log': tau_log,
            'time_log': time_log,
            'metadata': metadata
        }
        
    except Exception as e:
        print(f"Error loading simulation data: {e}")
        print("Make sure you have run the simulation first to generate the data files.")
        return None

def display_configuration_at_time(tsid_biped, sim_data, target_time):
    """Display the robot configuration at a specific time"""
    if sim_data is None:
        print("No simulation data available")
        return
    
    time_log = sim_data['time_log']
    q_log = sim_data['q_log']
    
    # Find the closest time index
    time_idx = np.argmin(np.abs(time_log - target_time))
    actual_time = time_log[time_idx]
    
    # Get the configuration at that time
    q_at_time = q_log[:, time_idx]
    
    print(f"Displaying configuration at time {actual_time:.3f}s (requested: {target_time:.3f}s)")
    
    # Update the robot visualization
    tsid_biped.display(q_at_time)

    q_temp = q_at_time
    
    return q_at_time, actual_time

def animate_simulation(tsid_biped, sim_data, start_time=0.0, end_time=None, speed_factor=1.0):
    """Animate the simulation data"""
    if sim_data is None:
        print("No simulation data available")
        return
    
    time_log = sim_data['time_log']
    q_log = sim_data['q_log']
    dt = sim_data['metadata']['dt']
    
    if end_time is None:
        end_time = time_log[-1]
    
    # Find start and end indices
    start_idx = np.argmin(np.abs(time_log - start_time))
    end_idx = np.argmin(np.abs(time_log - end_time))
    
    print(f"Animating simulation from {time_log[start_idx]:.3f}s to {time_log[end_idx]:.3f}s")
    print(f"Speed factor: {speed_factor}x")
    print("Press Ctrl+C to stop animation")
    
    try:
        for i in range(start_idx, end_idx, max(1, int(1.0/speed_factor))):
            q_at_time = q_log[:, i]
            current_time = time_log[i]
            
            # Update visualization
            tsid_biped.display(q_at_time)
            
            # Print progress every second
            if i % int(1.0/dt) == 0:
                print(f"Time: {current_time:.1f}s")
            
            # Sleep to control animation speed
            time.sleep(dt / speed_factor)
            
    except KeyboardInterrupt:
        print("\nAnimation stopped by user")

# Load the simulation data
print("\n" + "="*60)
print("LOADING SIMULATION DATA")
print("="*60)
sim_data = load_simulation_data()

if sim_data is not None:
    print("\nSimulation data loaded successfully!")
    print("\nOptions:")
    print("1. Display configuration at specific time:")
    print("   q_at_time, actual_time = display_configuration_at_time(tsid_biped, sim_data, target_time=5.0)")
    print("2. Animate the simulation:")
    print("   animate_simulation(tsid_biped, sim_data, start_time=0.0, end_time=10.0, speed_factor=2.0)")
    print("3. Access data directly:")
    print("   q_log = sim_data['q_log']  # Joint positions")
    print("   time_log = sim_data['time_log']  # Time array")
    
    # Display initial configuration
    print("\nDisplaying initial configuration...")
    display_configuration_at_time(tsid_biped, sim_data, 0.0)
    
    # Example: Display configurations at specific times
    print("\nExample usage - displaying configurations at different times:")
    
    # Uncomment any of these to see the robot at specific times:
    # display_configuration_at_time(tsid_biped, sim_data, target_time=2.0)   # Early walking
    # display_configuration_at_time(tsid_biped, sim_data, target_time=5.0)   # Mid walking  
    # display_configuration_at_time(tsid_biped, sim_data, target_time=10.0)  # Late walking
    # display_configuration_at_time(tsid_biped, sim_data, target_time=15.0)  # End of simulation
    
    # Example: Animate a portion of the simulation
    # animate_simulation(tsid_biped, sim_data, start_time=0.0, end_time=5.0, speed_factor=1.0)
    
    # Interactive time selection
    print(f"\nSimulation duration: 0.0 to {sim_data['time_log'][-1]:.1f} seconds")
    print("To display robot at a specific time, uncomment and modify the lines above, or use:")
    print("display_configuration_at_time(tsid_biped, sim_data, target_time=YOUR_TIME)")
    
    # Uncomment this block for interactive time input:
    """
    try:
        while True:
            user_time = input(f"\nEnter time (0.0 to {sim_data['time_log'][-1]:.1f}s) or 'q' to quit: ")
            if user_time.lower() == 'q':
                break
            try:
                target_time = float(user_time)
                display_configuration_at_time(tsid_biped, sim_data, target_time)
            except ValueError:
                print("Please enter a valid number or 'q' to quit")
    except KeyboardInterrupt:
        print("\nExiting...")
    """
else:
    print("Could not load simulation data. Please run the walking simulation first.")

display_configuration_at_time(tsid_biped, sim_data, target_time=2.0)   # Early walking
display_configuration_at_time(tsid_biped, sim_data, target_time=5.0)   # Mid walking  
display_configuration_at_time(tsid_biped, sim_data, target_time=10.0)  # Late walking
display_configuration_at_time(tsid_biped, sim_data, target_time=15.0)  # End of simulation


display_configuration_at_time(tsid_biped, sim_data, target_time=0.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=12.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=25.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=37.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=46.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=60.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=70.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=85.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=90.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=108.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=118.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=132.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=142.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=156.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=166.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=180.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=192.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=204.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=218.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=228.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=240.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=252.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=262.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=276.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=286.0) ##########TO modify
display_configuration_at_time(tsid_biped, sim_data, target_time=300.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=313.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=325.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=326.0)
display_configuration_at_time(tsid_biped, sim_data, target_time=350.0)














