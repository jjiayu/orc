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

CoM_Height_Offset = 0.7  # Height offset above the stance foot
step_height = 0.1

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
# FOOTSTEP LOADING: Load from JSON file or use hardcoded values
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

# Load footsteps from JSON or use hardcoded values
if USE_JSON_FOOTSTEPS:
    loaded_footsteps = load_footsteps_from_json(JSON_FOOTSTEP_FILE)
else:
    loaded_footsteps = None

if loaded_footsteps is not None:
    # Use loaded footsteps
    # Extract footstep targets and determine gait pattern
    # IMPORTANT: Skip the first footstep as it represents the current stance foot position, not a target
    # The first entry in the JSON file is where the stance foot currently is, not where it should move to
    actual_footsteps = loaded_footsteps[1:]  # Skip first stance foot position
    Num_Steps = len(actual_footsteps)
    # The JSON contains stance foot labels, so we need to reverse to get swing foot
    first_stance_foot = loaded_footsteps[0]['foot'] if loaded_footsteps else "left"
    first_swing_foot = "left" if first_stance_foot == "right" else "right"
    
    # Create footstep targets array (excluding the first stance foot position)
    footstep_targets = [step['position'] for step in actual_footsteps]
    
    print("Using footsteps from JSON file:")
    print(f"  Initial stance: {loaded_footsteps[0]['foot']} foot at [{loaded_footsteps[0]['position'][0]:.3f}, {loaded_footsteps[0]['position'][1]:.3f}, {loaded_footsteps[0]['position'][2]:.3f}] (not used as target)")
    print("  Stepping targets:")
    for i, step in enumerate(actual_footsteps[:4]):  # Show first 4 actual targets
        pos = step['position']
        print(f"    Target {i}: {step['foot']} foot -> [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], yaw: {step['yaw']:.3f}")
    if len(actual_footsteps) > 4:
        print(f"    ... and {len(actual_footsteps) - 4} more targets")
    
else:
    # Use hardcoded footsteps (original implementation)
    print("Using hardcoded footsteps")
    Num_Steps = 3
    first_swing_foot = "left"
    
    if first_swing_foot == "left":
        footstep_targets = [
            np.array([0.11, 0.085, 0.0]),   # Step 0: Left foot target
            np.array([0.21, -0.085, 0.0]),   # Step 1: Right foot target
            np.array([0.31, 0.085, 0.0]),   # Step 2: Left foot target
        ]
    else:
        footstep_targets = [
            np.array([0.11, -0.085, 0.0]),  # Step 0: Right foot target
            np.array([0.21, 0.085, 0.0]),   # Step 1: Left foot target
            np.array([0.31, -0.085, 0.0]),  # Step 2: Right foot target
        ]

# ============================================================================
# GAIT PATTERN GENERATION
# ============================================================================

steps_phases = 3  # Each step has 3 phases: double -> stance -> double

# Create gait pattern: [double, swing_foot, double] for each step
# Alternate between left and right swing foot for each step
gait_pattern = []
for step in range(Num_Steps):
    # Start with first_swing_foot, then alternate
    if first_swing_foot == "left":
        swing_foot = "left" if step % 2 == 0 else "right"
    else:
        swing_foot = "right" if step % 2 == 0 else "left"
    gait_pattern.extend(["double", swing_foot, "double"])

print("Gait pattern:", gait_pattern)
print("Footstep targets:", len(footstep_targets), "steps")

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

def load_and_visualize_environment(tsid_biped, env_file="stairs_up_and_down"):
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
load_and_visualize_environment(tsid_biped, "threepathnas")

# Visualize the footsteps (show all footsteps including initial stance position)
if loaded_footsteps is not None:
    # For visualization, show all footsteps including the initial stance position
    all_footstep_positions = [step['position'] for step in loaded_footsteps]
    visualize_footsteps_in_meshcat(tsid_biped, all_footstep_positions, loaded_footsteps)
else:
    # For hardcoded footsteps, add the current foot positions as initial stance
    current_lf_pos = tsid_biped.get_placement_LF().translation.copy()
    current_rf_pos = tsid_biped.get_placement_RF().translation.copy()
    if first_swing_foot == "left":
        # Right foot is initial stance
        all_footstep_positions = [current_rf_pos] + footstep_targets
    else:
        # Left foot is initial stance  
        all_footstep_positions = [current_lf_pos] + footstep_targets
    visualize_footsteps_in_meshcat(tsid_biped, all_footstep_positions, None)

# ============================================================================
# GOAL VISUALIZATION
# ============================================================================

def visualize_goal(tsid_biped, footstep_targets):
    """Visualize goal as a circle at the last footstep position"""
    import meshcat.geometry as g
    import meshcat.transformations as tf
    
    if hasattr(tsid_biped, 'viz') and tsid_biped.viz is not None and len(footstep_targets) > 0:
        # Get the last footstep position
        last_footstep = footstep_targets[-1]
        
        # Goal ball parameters
        goal_radius = 0.025
        goal_color = 0xFFD700  # Gold color
        
        # Create sphere geometry for the goal ball
        goal_ball = g.Sphere(goal_radius)
        goal_material = g.MeshLambertMaterial(color=goal_color, opacity=0.8)
        
        # Position the goal ball so it's half inside the surface (center at half radius height)
        goal_pos = [last_footstep[0], last_footstep[1], goal_radius / 2]
        goal_transform = tf.translation_matrix(goal_pos)
        
        # Add to meshcat
        tsid_biped.viz.viewer["goal/target_circle"].set_object(goal_ball, goal_material)
        tsid_biped.viz.viewer["goal/target_circle"].set_transform(goal_transform)
        
        print(f"Added goal visualization at [{goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f}]")
    else:
        print("Meshcat visualizer not available or no footsteps, skipping goal visualization")

# Visualize the goal
visualize_goal(tsid_biped, footstep_targets)

# Phase durations: 3 phases per step, repeated for all steps
base_phase_durations = [3.0, 6.0, 3.0]  # [double, stance, double]
phase_durations = base_phase_durations * Num_Steps
print("Phase durations:", phase_durations)
print("Total phases:", len(phase_durations))

# Function to calculate CoM height based on stance foot positions
def calculate_com_height(lf_pos, rf_pos, contact_phase):
    """Calculate CoM height based on stance foot positions and contact phase"""
    if contact_phase == "left":
        # Left foot is stance, CoM height = left foot height + offset
        return lf_pos[2] + CoM_Height_Offset
    elif contact_phase == "right":
        # Right foot is stance, CoM height = right foot height + offset
        return rf_pos[2] + CoM_Height_Offset
    else:  # double support
        # Use the higher of the two feet + offset for stability
        max_foot_height = max(lf_pos[2], rf_pos[2])
        return max_foot_height + CoM_Height_Offset

# Get current robot state for initial positions
current_com = tsid_biped.robot.com(tsid_biped.formulation.data()).copy()
current_lf_pos = tsid_biped.get_placement_LF().translation.copy()
current_rf_pos = tsid_biped.get_placement_RF().translation.copy()

# Calculate initial CoM height based on current foot positions (double support)
initial_com_height = calculate_com_height(current_lf_pos, current_rf_pos, "double")
current_com[2] = initial_com_height

print(f"CoM Height Offset: {CoM_Height_Offset:.3f}m")
print(f"Initial CoM: [{current_com[0]:.3f}, {current_com[1]:.3f}, {current_com[2]:.3f}] (dynamic height)")
print(f"Initial LF:  [{current_lf_pos[0]:.3f}, {current_lf_pos[1]:.3f}, {current_lf_pos[2]:.3f}]")
print(f"Initial RF:  [{current_rf_pos[0]:.3f}, {current_rf_pos[1]:.3f}, {current_rf_pos[2]:.3f}]")

# Create CoM path: [start_pos, end_pos] for each phase
com_path = []
# Create left foot path: [start_pos, end_pos] for each phase  
lf_path = []
# Create right foot path: [start_pos, end_pos] for each phase
rf_path = []

# Initialize positions
com_pos = current_com.copy()
lf_pos = current_lf_pos.copy()
rf_pos = current_rf_pos.copy()

for phase_idx, contact_phase in enumerate(gait_pattern):
    step_number = phase_idx // 3  # Which step we're in (0, 1, 2, ...)
    phase_in_step = phase_idx % 3  # Which phase within the step (0=double, 1=stance, 2=double)
    
    print(f"\nPhase {phase_idx}: {contact_phase} (Step {step_number}, Phase {phase_in_step})")
    
    # Store starting positions for this phase
    com_start = com_pos.copy()
    lf_start = lf_pos.copy()
    rf_start = rf_pos.copy()
    
    if phase_in_step == 0:  # First double support phase of a step
        # CoM moves towards the support foot (opposite of swing foot in next phase)
        if step_number < len(footstep_targets):
            swing_foot = gait_pattern[phase_idx + 1]  # Next phase is the swing phase
            if swing_foot == "right":  # Right foot will swing, so left foot is support
                support_foot_pos = lf_pos.copy()
                stance_foot = "left"
            else:  # Left foot will swing, so right foot is support
                support_foot_pos = rf_pos.copy()
                stance_foot = "right"
            
            # CoM target: above support foot with dynamic height
            com_target = support_foot_pos.copy()
            com_target[2] = calculate_com_height(lf_pos, rf_pos, stance_foot)
        else:
            com_target = com_pos.copy()  # Stay in place
            
        # Feet don't move during double support
        lf_target = lf_pos.copy()
        rf_target = rf_pos.copy()
        
    elif phase_in_step == 1:  # Single support phase (swing phase)
        # Swing foot moves to target position
        if step_number < len(footstep_targets):
            swing_foot = contact_phase  # Current phase tells us which foot swings
            if swing_foot == "right":  # Right foot swings, left foot supports
                rf_target = footstep_targets[step_number].copy()
                lf_target = lf_pos.copy()  # Support foot stays
                stance_foot = "left"
            else:  # Left foot swings, right foot supports
                lf_target = footstep_targets[step_number].copy()
                rf_target = rf_pos.copy()  # Support foot stays
                stance_foot = "right"
            
            # CoM stays above support foot with dynamic height
            com_target = com_pos.copy()
            com_target[2] = calculate_com_height(lf_target, rf_target, stance_foot)
        else:
            lf_target = lf_pos.copy()
            rf_target = rf_pos.copy()
            com_target = com_pos.copy()
            
    else:  # phase_in_step == 2: Second double support phase
        # CoM moves to midpoint between feet for stabilization
        # Calculate midpoint between current feet positions
        midpoint = (lf_pos + rf_pos) / 2.0
        com_target = midpoint.copy()
        com_target[2] = calculate_com_height(lf_pos, rf_pos, "double")
            
        # Feet don't move during double support
        lf_target = lf_pos.copy()
        rf_target = rf_pos.copy()
    
    # Store paths for this phase
    com_path.append([com_start, com_target])
    lf_path.append([lf_start, lf_target])
    rf_path.append([rf_start, rf_target])
    
    # Update positions for next phase
    com_pos = com_target.copy()
    lf_pos = lf_target.copy()
    rf_pos = rf_target.copy()
    
    print(f"  CoM: [{com_start[0]:.3f}, {com_start[1]:.3f}, {com_start[2]:.3f}] -> [{com_target[0]:.3f}, {com_target[1]:.3f}, {com_target[2]:.3f}] (height: {com_target[2]:.3f})")
    print(f"  LF:  [{lf_start[0]:.3f}, {lf_start[1]:.3f}, {lf_start[2]:.3f}] -> [{lf_target[0]:.3f}, {lf_target[1]:.3f}, {lf_target[2]:.3f}]")
    print(f"  RF:  [{rf_start[0]:.3f}, {rf_start[1]:.3f}, {rf_start[2]:.3f}] -> [{rf_target[0]:.3f}, {rf_target[1]:.3f}, {rf_target[2]:.3f}]")

print(f"\nGait pattern setup completed!")
print(f"Total phases: {len(gait_pattern)}")
print(f"CoM path length: {len(com_path)}")
print(f"LF path length: {len(lf_path)}")
print(f"RF path length: {len(rf_path)}")

# Calculate simulation parameters
total_duration = sum(phase_durations)
N = int(total_duration / conf.dt)
N_pre = int(conf.T_pre / conf.dt)
N_post = int(conf.T_post / conf.dt)

# Interpolate trajectories
print(f"Total duration: {total_duration:.1f}s, N: {N}, N_pre: {N_pre}, N_post: {N_post}")

# Initialize trajectory arrays
com_pos_traj = np.zeros((3, N))
com_vel_traj = np.zeros((3, N))
com_acc_traj = np.zeros((3, N))
lf_pos_traj = np.zeros((3, N))
lf_vel_traj = np.zeros((3, N))
lf_acc_traj = np.zeros((3, N))
rf_pos_traj = np.zeros((3, N))
rf_vel_traj = np.zeros((3, N))
rf_acc_traj = np.zeros((3, N))
contact_pattern = []

# Generate trajectories for each phase
time_idx = 0
for phase_idx in range(len(gait_pattern)):
    phase_duration = phase_durations[phase_idx]
    N_phase = int(phase_duration / conf.dt)
    contact_phase = gait_pattern[phase_idx]
    
    print(f"\nInterpolating Phase {phase_idx}: {contact_phase}, Duration: {phase_duration:.1f}s, N_phase: {N_phase}")
    
    # Get start and end positions for this phase
    com_start = com_path[phase_idx][0]
    com_end = com_path[phase_idx][1]
    lf_start = lf_path[phase_idx][0]
    lf_end = lf_path[phase_idx][1]
    rf_start = rf_path[phase_idx][0]
    rf_end = rf_path[phase_idx][1]
    
    # Generate CoM trajectory using 5th order polynomial
    if np.allclose(com_start, com_end):
        # No movement - constant position
        com_pos_phase = np.tile(com_start.reshape(-1, 1), (1, N_phase))
        com_vel_phase = np.zeros((3, N_phase))
        com_acc_phase = np.zeros((3, N_phase))
    else:
        com_pos_phase, com_vel_phase, com_acc_phase = compute_5th_order_poly_traj(
            com_start, com_end, phase_duration, conf.dt)
    
    # Generate foot trajectories based on contact phase
    if contact_phase == "double":
        # Both feet stay constant during double support
        lf_pos_phase = np.tile(lf_start.reshape(-1, 1), (1, N_phase))
        lf_vel_phase = np.zeros((3, N_phase))
        lf_acc_phase = np.zeros((3, N_phase))
        rf_pos_phase = np.tile(rf_start.reshape(-1, 1), (1, N_phase))
        rf_vel_phase = np.zeros((3, N_phase))
        rf_acc_phase = np.zeros((3, N_phase))
        
    elif contact_phase == "left":
        # Left foot swings, right foot is stance (constant)
        rf_pos_phase = np.tile(rf_start.reshape(-1, 1), (1, N_phase))
        rf_vel_phase = np.zeros((3, N_phase))
        rf_acc_phase = np.zeros((3, N_phase))
        
        # Generate square swing trajectory for left foot
        lf_pos_phase = np.zeros((3, N_phase))
        lf_vel_phase = np.zeros((3, N_phase))
        lf_acc_phase = np.zeros((3, N_phase))
        
        for i in range(N_phase):
            local_time = i * conf.dt
            pos, vel, acc = generate_square_foot_trajectory(
                lf_start, lf_end, step_height, phase_duration, local_time)
            lf_pos_phase[:, i] = pos
            lf_vel_phase[:, i] = vel
            lf_acc_phase[:, i] = acc
            
    elif contact_phase == "right":
        # Right foot swings, left foot is stance (constant)
        lf_pos_phase = np.tile(lf_start.reshape(-1, 1), (1, N_phase))
        lf_vel_phase = np.zeros((3, N_phase))
        lf_acc_phase = np.zeros((3, N_phase))
        
        # Generate square swing trajectory for right foot
        rf_pos_phase = np.zeros((3, N_phase))
        rf_vel_phase = np.zeros((3, N_phase))
        rf_acc_phase = np.zeros((3, N_phase))
        
        for i in range(N_phase):
            local_time = i * conf.dt
            pos, vel, acc = generate_square_foot_trajectory(
                rf_start, rf_end, step_height, phase_duration, local_time)
            rf_pos_phase[:, i] = pos
            rf_vel_phase[:, i] = vel
            rf_acc_phase[:, i] = acc
    
    # Store trajectories in main arrays
    end_idx = min(time_idx + N_phase, N)
    actual_N = end_idx - time_idx
    
    com_pos_traj[:, time_idx:end_idx] = com_pos_phase[:, :actual_N]
    com_vel_traj[:, time_idx:end_idx] = com_vel_phase[:, :actual_N]
    com_acc_traj[:, time_idx:end_idx] = com_acc_phase[:, :actual_N]
    lf_pos_traj[:, time_idx:end_idx] = lf_pos_phase[:, :actual_N]
    lf_vel_traj[:, time_idx:end_idx] = lf_vel_phase[:, :actual_N]
    lf_acc_traj[:, time_idx:end_idx] = lf_acc_phase[:, :actual_N]
    rf_pos_traj[:, time_idx:end_idx] = rf_pos_phase[:, :actual_N]
    rf_vel_traj[:, time_idx:end_idx] = rf_vel_phase[:, :actual_N]
    rf_acc_traj[:, time_idx:end_idx] = rf_acc_phase[:, :actual_N]
    
    # Generate contact pattern (stance foot at each timestep)
    for i in range(actual_N):
        if contact_phase == "double":
            contact_pattern.append("double")  # Both feet in contact
        elif contact_phase == "left":
            contact_pattern.append("right")   # Right foot is stance (left swings)
        elif contact_phase == "right":
            contact_pattern.append("left")    # Left foot is stance (right swings)
    
    time_idx = end_idx

print(f"\nTrajectory interpolation completed!")
print(f"Generated {len(contact_pattern)} timesteps")
print(f"Contact pattern types: {set(contact_pattern)}")
print(f"CoM trajectory shape: {com_pos_traj.shape}")
print(f"LF trajectory shape: {lf_pos_traj.shape}")
print(f"RF trajectory shape: {rf_pos_traj.shape}")

# Initialize simulation variables
t = -conf.T_pre
q, v = tsid_biped.q, tsid_biped.v
q_list = []

# Logging variables
com_pos = np.empty((3, N + N_post)) * nan
com_vel = np.empty((3, N + N_post)) * nan
com_acc = np.empty((3, N + N_post)) * nan

# Combined real-time reference generation and simulation loop
input("Press enter to start CoM tracking")
for i in range(-N_pre, N + N_post):
    time_start = time.time()
    
    # Handle contact phase changes
    if i == 0:
        print("Starting to walk")
        # Initial contact state is double support, no changes needed
    elif i > 0 and i < N - 1:
        if contact_pattern[i] != contact_pattern[i - 1]:
            print(f"Time {t:.3f}: Changing contact from {contact_pattern[i-1]} to {contact_pattern[i]}")
            
            if contact_pattern[i] == "left":
                # Left foot is stance, right foot swings
                print("set left support right swing contact configurations")
                if tsid_biped.contact_RF_active:
                    tsid_biped.remove_contact_RF()
                if not tsid_biped.contact_LF_active:
                    tsid_biped.add_contact_LF()
            elif contact_pattern[i] == "right":
                # Right foot is stance, left foot swings  
                print("set right support left swing contact configurations")
                if tsid_biped.contact_LF_active:
                    tsid_biped.remove_contact_LF()
                if not tsid_biped.contact_RF_active:
                    tsid_biped.add_contact_RF()
            elif contact_pattern[i] == "double":
                # Both feet in contact
                print("set double support contact configurations")
                if not tsid_biped.contact_LF_active:
                    tsid_biped.add_contact_LF()
                if not tsid_biped.contact_RF_active:
                    tsid_biped.add_contact_RF()
    
    # Set reference trajectories
    if i < 0:
        # Preparation phase: hold initial position
        tsid_biped.set_com_ref(current_com, np.zeros(3), np.zeros(3))
        tsid_biped.set_LF_3d_ref(current_lf_pos, np.zeros(3), np.zeros(3))
        tsid_biped.set_RF_3d_ref(current_rf_pos, np.zeros(3), np.zeros(3))
    elif i < N:
        # Main trajectory phase
        tsid_biped.set_com_ref(com_pos_traj[:, i], com_vel_traj[:, i], com_acc_traj[:, i])
        tsid_biped.set_LF_3d_ref(lf_pos_traj[:, i], lf_vel_traj[:, i], lf_acc_traj[:, i])
        tsid_biped.set_RF_3d_ref(rf_pos_traj[:, i], rf_vel_traj[:, i], rf_acc_traj[:, i])
    else:
        # Post-trajectory phase: hold final position
        tsid_biped.set_com_ref(com_pos_traj[:, -1], np.zeros(3), np.zeros(3))
        tsid_biped.set_LF_3d_ref(lf_pos_traj[:, -1], np.zeros(3), np.zeros(3))
        tsid_biped.set_RF_3d_ref(rf_pos_traj[:, -1], np.zeros(3), np.zeros(3))
    
    # Solve QP problem
    HQPData = tsid_biped.formulation.computeProblemData(t, q, v)
    sol = tsid_biped.solver.solve(HQPData)
    
    if sol.status != 0:
        print(f"QP problem could not be solved! Error code: {sol.status}")
        break
        
    if norm(v, 2) > 10.0:
        print(f"Time {t:.3f} Velocities are too high! ||v||: {norm(v)}")
        break
    
    # Get solution
    dv = tsid_biped.formulation.getAccelerations(sol)
    
    # Log data
    if i >= 0:
        com_pos[:, i] = tsid_biped.robot.com(tsid_biped.formulation.data())
        com_vel[:, i] = tsid_biped.robot.com_vel(tsid_biped.formulation.data())
        com_acc[:, i] = tsid_biped.comTask.getAcceleration(dv)
    
    # Print status
    if i % conf.PRINT_N == 0:
        print(f"Time {t:.3f}")
        if i >= 0 and i < N:
            # Contact status (check actual QP solution)
            lf_contact = "ACTIVE" if tsid_biped.formulation.checkContact(tsid_biped.contactLF.name, sol) else "INACTIVE"
            rf_contact = "ACTIVE" if tsid_biped.formulation.checkContact(tsid_biped.contactRF.name, sol) else "INACTIVE"
            print(f"  Contact pattern: {contact_pattern[i]} | LF: {lf_contact} | RF: {rf_contact}")
            
            # CoM target vs actual
            com_target = com_pos_traj[:, i]
            com_actual = tsid_biped.robot.com(tsid_biped.formulation.data())
            print(f"  CoM target:  [{com_target[0]:.3f}, {com_target[1]:.3f}, {com_target[2]:.3f}]")
            print(f"  CoM actual:  [{com_actual[0]:.3f}, {com_actual[1]:.3f}, {com_actual[2]:.3f}]")
            print(f"  CoM tracking error: {norm(tsid_biped.comTask.position_error, 2):.3f}")
            
            # Foot targets vs actual
            lf_target = lf_pos_traj[:, i]
            rf_target = rf_pos_traj[:, i]
            lf_actual = tsid_biped.get_placement_LF().translation
            rf_actual = tsid_biped.get_placement_RF().translation
            print(f"  LF target:   [{lf_target[0]:.3f}, {lf_target[1]:.3f}, {lf_target[2]:.3f}]")
            print(f"  LF actual:   [{lf_actual[0]:.3f}, {lf_actual[1]:.3f}, {lf_actual[2]:.3f}]")
            print(f"  RF target:   [{rf_target[0]:.3f}, {rf_target[1]:.3f}, {rf_target[2]:.3f}]")
            print(f"  RF actual:   [{rf_actual[0]:.3f}, {rf_actual[1]:.3f}, {rf_actual[2]:.3f}]")
            
        print(f"  ||v||: {norm(v, 2):.3f}, ||dv||: {norm(dv):.3f}")
    
    # Integrate dynamics
    q, v = tsid_biped.integrate_dv(q, v, dv, conf.dt)
    t += conf.dt
    
    # Update visualization
    if i % conf.DISPLAY_N == 0:
        q_list.append(q.copy())
        tsid_biped.display(q)
    
    # Real-time control
    time_spent = time.time() - time_start
    if time_spent < conf.dt:
        time.sleep(conf.dt - time_spent)

print("CoM tracking completed!")

# input("Press enter to start")
# for i in range(-N_pre, N + N_post):
#     time_start = time.time()

#     if i == 0:
#         print("Starting to walk (remove contact left foot)")
#         tsid_biped.remove_contact_LF()
#     elif i > 0 and i < N - 1:
#         if contact_phase[i] != contact_phase[i - 1]:
#             print(
#                 "Time %.3f Changing contact phase from %s to %s"
#                 % (t, contact_phase[i - 1], contact_phase[i])
#             )
#             if contact_phase[i] == "left":
#                 tsid_biped.add_contact_LF()
#                 tsid_biped.remove_contact_RF()
#             elif contact_phase[i] == "right":
#                 tsid_biped.add_contact_RF()
#                 tsid_biped.remove_contact_LF()
#             elif contact_phase[i] == "double":
#                 # Double support - both feet in contact
#                 tsid_biped.add_contact_LF()
#                 tsid_biped.add_contact_RF()

#     if i < 0:
#         tsid_biped.set_com_ref(
#             com_pos_ref[:, 0], 0 * com_vel_ref[:, 0], 0 * com_acc_ref[:, 0]
#         )
#     elif i < N-1:
#         tsid_biped.set_com_ref(com_pos_ref[:, i], com_vel_ref[:, i], com_acc_ref[:, i])
#         tsid_biped.set_LF_3d_ref(x_LF_ref[:, i], dx_LF_ref[:, i], ddx_LF_ref[:, i])
#         tsid_biped.set_RF_3d_ref(x_RF_ref[:, i], dx_RF_ref[:, i], ddx_RF_ref[:, i])

#     HQPData = tsid_biped.formulation.computeProblemData(t, q, v)
#     sol = tsid_biped.solver.solve(HQPData)

#     if sol.status != 0:
#         print("QP problem could not be solved! Error code:", sol.status)
#         break
#     if norm(v, 2) > 10.0:
#         print("Time %.3f Velocities are too high, stop everything!" % (t), norm(v))
#         break

#     if i > 0:
#         q_log[:, i] = q
#         v_log[:, i] = v
#         tau[:, i] = tsid_biped.formulation.getActuatorForces(sol)
#     dv = tsid_biped.formulation.getAccelerations(sol)

#     if i >= 0:
#         com_pos[:, i] = tsid_biped.robot.com(tsid_biped.formulation.data())
#         com_vel[:, i] = tsid_biped.robot.com_vel(tsid_biped.formulation.data())
#         com_acc[:, i] = tsid_biped.comTask.getAcceleration(dv)
#         com_acc_des[:, i] = tsid_biped.comTask.getDesiredAcceleration
#         x_LF[:, i], dx_LF[:, i], ddx_LF[:, i] = tsid_biped.get_LF_3d_pos_vel_acc(dv)
#         if not tsid_biped.contact_LF_active:
#             ddx_LF_des[:, i] = tsid_biped.leftFootTask.getDesiredAcceleration[:3]
#         x_RF[:, i], dx_RF[:, i], ddx_RF[:, i] = tsid_biped.get_RF_3d_pos_vel_acc(dv)
#         if not tsid_biped.contact_RF_active:
#             ddx_RF_des[:, i] = tsid_biped.rightFootTask.getDesiredAcceleration[:3]

#         if tsid_biped.formulation.checkContact(tsid_biped.contactRF.name, sol):
#             T_RF = tsid_biped.contactRF.getForceGeneratorMatrix
#             f_RF[:, i] = T_RF.dot(
#                 tsid_biped.formulation.getContactForce(tsid_biped.contactRF.name, sol)
#             )
#             if f_RF[2, i] > 1e-3:
#                 cop_RF[0, i] = f_RF[4, i] / f_RF[2, i]
#                 cop_RF[1, i] = -f_RF[3, i] / f_RF[2, i]
#         if tsid_biped.formulation.checkContact(tsid_biped.contactLF.name, sol):
#             T_LF = tsid_biped.contactRF.getForceGeneratorMatrix
#             f_LF[:, i] = T_LF.dot(
#                 tsid_biped.formulation.getContactForce(tsid_biped.contactLF.name, sol)
#             )
#             if f_LF[2, i] > 1e-3:
#                 cop_LF[0, i] = f_LF[4, i] / f_LF[2, i]
#                 cop_LF[1, i] = -f_LF[3, i] / f_LF[2, i]

#     if i % conf.PRINT_N == 0:
#         print("Time %.3f" % (t))
#         if (tsid_biped.formulation.checkContact(tsid_biped.contactRF.name, sol) and i >= 0):
#             print("\tnormal force %s: %.1f"% (tsid_biped.contactRF.name.ljust(20, "."), f_RF[2, i]))
#         if (tsid_biped.formulation.checkContact(tsid_biped.contactLF.name, sol) and i >= 0):
#             print("\tnormal force %s: %.1f"% (tsid_biped.contactLF.name.ljust(20, "."), f_LF[2, i]))
#         print("\ttracking err %s: %.3f"% (tsid_biped.comTask.name.ljust(20, "."),
#                 norm(tsid_biped.comTask.position_error, 2),))
#         print("\t||v||: %.3f\t ||dv||: %.3f" % (norm(v, 2), norm(dv)))

#     q, v = tsid_biped.integrate_dv(q, v, dv, conf.dt)
#     t += conf.dt

#     if i % conf.DISPLAY_N == 0:
#         q_list.append(q)
#         tsid_biped.display(q)

#     time_spent = time.time() - time_start
#     if time_spent < conf.dt:
#         time.sleep(conf.dt - time_spent)

# while True:
#     replay = input("Play video again? [Y]/n: ") or "Y"
#     if replay.lower() == "y":
#         for q in q_list:
#             tsid_biped.display(q)
#             time.sleep(conf.DISPLAY_N*conf.dt)
#     else:
#         break
# # PLOT STUFF
# time = np.arange(0.0, (N + N_post) * conf.dt, conf.dt)

# if PLOT_COM:
#     (f, ax) = plut.create_empty_figure(3, 1)
#     for i in range(3):
#         ax[i].plot(time, com_pos[i, :], label="CoM " + str(i))
#         ax[i].plot(time[:N], com_pos_ref[i, :], "r:", label="CoM Ref " + str(i))
#         ax[i].set_xlabel("Time [s]")
#         ax[i].set_ylabel("CoM [m]")
#         leg = ax[i].legend()
#         leg.get_frame().set_alpha(0.5)

#     (f, ax) = plut.create_empty_figure(3, 1)
#     for i in range(3):
#         ax[i].plot(time, com_vel[i, :], label="CoM Vel " + str(i))
#         ax[i].plot(time[:N], com_vel_ref[i, :], "r:", label="CoM Vel Ref " + str(i))
#         ax[i].set_xlabel("Time [s]")
#         ax[i].set_ylabel("CoM Vel [m/s]")
#         leg = ax[i].legend()
#         leg.get_frame().set_alpha(0.5)

#     (f, ax) = plut.create_empty_figure(3, 1)
#     for i in range(3):
#         ax[i].plot(time, com_acc[i, :], label="CoM Acc " + str(i))
#         ax[i].plot(time[:N], com_acc_ref[i, :], "r:", label="CoM Acc Ref " + str(i))
#         ax[i].plot(time, com_acc_des[i, :], "g--", label="CoM Acc Des " + str(i))
#         ax[i].set_xlabel("Time [s]")
#         ax[i].set_ylabel("CoM Acc [m/s^2]")
#         leg = ax[i].legend()
#         leg.get_frame().set_alpha(0.5)

# if PLOT_COP:
#     (f, ax) = plut.create_empty_figure(2, 1)
#     for i in range(2):
#         ax[i].plot(time, cop_LF[i, :], label="CoP LF " + str(i))
#         ax[i].plot(time, cop_RF[i, :], label="CoP RF " + str(i))
#         #        ax[i].plot(time[:N], cop_ref[i,:], label='CoP ref '+str(i))
#         if i == 0:
#             ax[i].plot(
#                 [time[0], time[-1]],
#                 [-conf.lxn, -conf.lxn],
#                 ":",
#                 label="CoP Lim " + str(i),
#             )
#             ax[i].plot(
#                 [time[0], time[-1]],
#                 [conf.lxp, conf.lxp],
#                 ":",
#                 label="CoP Lim " + str(i),
#             )
#         elif i == 1:
#             ax[i].plot(
#                 [time[0], time[-1]],
#                 [-conf.lyn, -conf.lyn],
#                 ":",
#                 label="CoP Lim " + str(i),
#             )
#             ax[i].plot(
#                 [time[0], time[-1]],
#                 [conf.lyp, conf.lyp],
#                 ":",
#                 label="CoP Lim " + str(i),
#             )
#         ax[i].set_xlabel("Time [s]")
#         ax[i].set_ylabel("CoP [m]")
#         leg = ax[i].legend()
#         leg.get_frame().set_alpha(0.5)


# # (f, ax) = plut.create_empty_figure(3,2)
# # ax = ax.reshape((6))
# # for i in range(6):
# #    ax[i].plot(time, f_LF[i,:], label='Force LF '+str(i))
# #    ax[i].plot(time, f_RF[i,:], label='Force RF '+str(i))
# #    ax[i].set_xlabel('Time [s]')
# #    ax[i].set_ylabel('Force [N/Nm]')
# #    leg = ax[i].legend()
# #    leg.get_frame().set_alpha(0.5)

# if PLOT_FOOT_TRAJ:
#     for i in range(3):
#         plt.figure()
#         plt.plot(time, x_RF[i, :], label="x RF " + str(i))
#         plt.plot(time[:N], x_RF_ref[i, :], ":", label="x RF ref " + str(i))
#         plt.plot(time, x_LF[i, :], label="x LF " + str(i))
#         plt.plot(time[:N], x_LF_ref[i, :], ":", label="x LF ref " + str(i))
#         plt.legend()

#     # for i in range(3):
#     #    plt.figure()
#     #    plt.plot(time, dx_RF[i,:], label='dx RF '+str(i))
#     #    plt.plot(time[:N], dx_RF_ref[i,:], ':', label='dx RF ref '+str(i))
#     #    plt.plot(time, dx_LF[i,:], label='dx LF '+str(i))
#     #    plt.plot(time[:N], dx_LF_ref[i,:], ':', label='dx LF ref '+str(i))
#     #    plt.legend()
#     #
#     # for i in range(3):
#     #    plt.figure()
#     #    plt.plot(time, ddx_RF[i,:], label='ddx RF '+str(i))
#     #    plt.plot(time[:N], ddx_RF_ref[i,:], ':', label='ddx RF ref '+str(i))
#     #    plt.plot(time, ddx_RF_des[i,:], '--', label='ddx RF des '+str(i))
#     #    plt.plot(time, ddx_LF[i,:], label='ddx LF '+str(i))
#     #    plt.plot(time[:N], ddx_LF_ref[i,:], ':', label='ddx LF ref '+str(i))
#     #    plt.plot(time, ddx_LF_des[i,:], '--', label='ddx LF des '+str(i))
#     #    plt.legend()

# if PLOT_TORQUES:
#     plt.figure()
#     for i in range(tsid_biped.robot.na):
#         tau_normalized = (
#             2
#             * (tau[i, :] - tsid_biped.tau_min[i])
#             / (tsid_biped.tau_max[i] - tsid_biped.tau_min[i])
#             - 1
#         )
#         # plot torques only for joints that reached 50% of max torque
#         if np.max(np.abs(tau_normalized)) > 0.5:
#             plt.plot(
#                 time, tau_normalized, alpha=0.5, label=tsid_biped.model.names[i + 2]
#             )
#     plt.plot([time[0], time[-1]], 2 * [-1.0], ":")
#     plt.plot([time[0], time[-1]], 2 * [1.0], ":")
#     plt.gca().set_xlabel("Time [s]")
#     plt.gca().set_ylabel("Normalized Torque")
#     leg = plt.legend()
#     leg.get_frame().set_alpha(0.5)

# if PLOT_JOINT_VEL:
#     plt.figure()
#     for i in range(tsid_biped.robot.na):
#         v_normalized = (
#             2
#             * (v_log[6 + i, :] - tsid_biped.v_min[i])
#             / (tsid_biped.v_max[i] - tsid_biped.v_min[i])
#             - 1
#         )
#         # plot v only for joints that reached 50% of max v
#         if np.max(np.abs(v_normalized)) > 0.5:
#             plt.plot(time, v_normalized, alpha=0.5, label=tsid_biped.model.names[i + 2])
#     plt.plot([time[0], time[-1]], 2 * [-1.0], ":")
#     plt.plot([time[0], time[-1]], 2 * [1.0], ":")
#     plt.gca().set_xlabel("Time [s]")
#     plt.gca().set_ylabel("Normalized Joint Vel")
#     leg = plt.legend()
# #    leg.get_frame().set_alpha(0.5)

# plt.show()
