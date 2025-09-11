import time

import orc.optimal_control.lipm.biped.romeo_conf as conf
# import talos_conf as conf
import matplotlib.pyplot as plt
import numpy as np
import orc.utils.plot_utils as plut
from numpy import nan
from numpy.linalg import norm as norm
from tsid_biped import TsidBiped
from orc.optimal_control.lipm.biped.lipm_to_tsid import compute_3rd_order_poly_traj, generate_swing_foot_trajectory

import tsid

print("".center(conf.LINE_WIDTH, "#"))
print(" Test Walking ".center(conf.LINE_WIDTH, "#"))
print("".center(conf.LINE_WIDTH, "#"), "\n")

CoM_Height = 0.6

USE_EIQUADPROG = 1
USE_PROXQP = 0
USE_OSQP = 0
VERBOSE = 0

PLOT_COM = 0
PLOT_COP = 0
PLOT_FOOT_TRAJ = 0
PLOT_TORQUES = 0
PLOT_JOINT_VEL = 0

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

#define contact pattern
Num_Steps = 2
stride_length = 0.1
first_swing_foot = "right"
initial_lf_position = tsid_biped.get_placement_LF().translation
initial_rf_position = tsid_biped.get_placement_RF().translation
footstep_plan = np.array([[0.11,-0.096,0.0],[0.21,0.096,0.0]])
CoM_Positions = []
for i in range(Num_Steps):
    if i == 0:
        # First step: midpoint between initial right foot (stance) and first footstep (left foot)
        if first_swing_foot == "left":
            stance_foot_pos = initial_rf_position
            next_footstep_pos = footstep_plan[i]
        else:
            stance_foot_pos = initial_lf_position
            next_footstep_pos = footstep_plan[i]
    else:
        # Subsequent steps: midpoint between previous footstep and current footstep
        stance_foot_pos = footstep_plan[i-1]
        next_footstep_pos = footstep_plan[i]
    
    # Calculate midpoint in 3D with constant CoM height
    com_pos = (stance_foot_pos + next_footstep_pos) / 2
    com_pos[2] = com_pos[2] + CoM_Height  # Set constant CoM height
    CoM_Positions.append(com_pos)
    print("stance foot position: ", stance_foot_pos)
    print("next footstep position: ", next_footstep_pos)
    print("CoM position: ", com_pos)

CoM_Positions = np.array(CoM_Positions)

contact_patterns = ["double", "single", "double"] #variables mean contact configuration
phase_durations = np.array([3.0, 6.0, 3.0])

# Multi-step walking: total duration for all steps
total_duration = Num_Steps * np.sum(phase_durations)
N = int(total_duration / conf.dt)
N_pre = int(conf.T_pre / conf.dt)
N_post = int(conf.T_post / conf.dt)

#Logging variables
com_pos = np.empty((3, N + N_post)) * nan
com_vel = np.empty((3, N + N_post)) * nan
com_acc = np.empty((3, N + N_post)) * nan
x_LF = np.empty((3, N + N_post)) * nan
dx_LF = np.empty((3, N + N_post)) * nan
ddx_LF = np.empty((3, N + N_post)) * nan
ddx_LF_des = np.empty((3, N + N_post)) * nan
x_RF = np.empty((3, N + N_post)) * nan
dx_RF = np.empty((3, N + N_post)) * nan
ddx_RF = np.empty((3, N + N_post)) * nan
ddx_RF_des = np.empty((3, N + N_post)) * nan
f_RF = np.zeros((6, N + N_post))
f_LF = np.zeros((6, N + N_post))
cop_RF = np.zeros((2, N + N_post))
cop_LF = np.zeros((2, N + N_post))
tau = np.zeros((tsid_biped.robot.na, N + N_post))
q_log = np.zeros((tsid_biped.robot.nq, N + N_post))
v_log = np.zeros((tsid_biped.robot.nv, N + N_post))

# Simple Planning - all computations in loop

# Initialize arrays
com_pos_ref = np.zeros((3, int(N)))
com_vel_ref = np.zeros((3, int(N)))
com_acc_ref = np.zeros((3, int(N)))
contact_phase = []

# Phase timing setup
phase_start_times = np.cumsum(np.concatenate([[0], phase_durations[:-1]]))
phase_end_times = np.cumsum(phase_durations)

# Get current robot foot positions
lf_current = tsid_biped.get_placement_LF().translation
rf_current = tsid_biped.get_placement_RF().translation

# Define CoM target positions for each phase based on current robot state
com_initial = tsid_biped.robot.com(tsid_biped.formulation.data())  # Current CoM position
com_initial[2] = CoM_Height  # Set desired height
print("com_initial: ", com_initial)
print("lf current: ", lf_current)
print("rf current: ", rf_current)

# Stepping parameters
step_height = 0.05  # Maximum foot lift height (reduced from 0.12 to 0.05)
stride_length = 0.1  # Forward step distance
# swing_foot_initial_pos = rf_current.copy()  # Right foot will be the swing foot
# stance_foot_pos = lf_current.copy()  # Left foot is stance foot

print("CoM trajectory setup completed!")

t = -conf.T_pre
q, v = tsid_biped.q, tsid_biped.v

qp_data_list = []
# c = 0
q_list = []

# Combined real-time reference generation and simulation loop
input("Press enter to start CoM tracking")
for i in range(-N_pre, N + N_post):
    time_start = time.time()
    
    # Generate CoM reference in real-time based on current time
    if i < 0:
        # Preparation phase: hold initial position with zero velocity/acceleration
        com_pos_ref_current = com_initial
        com_vel_ref_current = np.zeros(3)
        com_acc_ref_current = np.zeros(3)
        contact_phase_current = "double"
    elif i < N:
        # Main trajectory phase - compute reference on-the-fly
        t_traj = i * conf.dt  # Trajectory time (starts from 0)
        
        # Calculate which step and phase we're in
        step_duration = np.sum(phase_durations)
        current_step = int(t_traj // step_duration)  # Which step (0, 1, 2, ...)
        t_within_step = t_traj % step_duration  # Time within current step
        
        # Find which phase within the current step
        phase_idx = 0
        cumulative_time = 0
        for j in range(len(phase_durations)):
            cumulative_time += phase_durations[j]
            if t_within_step < cumulative_time:
                phase_idx = j
                break
        
        # Determine which foot is swinging based on step number and first_swing_foot
        if first_swing_foot == "left":
            swing_foot = "left" if current_step % 2 == 0 else "right"
        else:
            swing_foot = "right" if current_step % 2 == 0 else "left"
        
        # Determine contact pattern based on phase and swing foot
        if phase_idx == 1:  # Single support phase
            if swing_foot == "left":
                contact_phase_current = "right"  # Right stance, left swing
            else:
                contact_phase_current = "left"   # Left stance, right swing
        else:
            contact_phase_current = "double"
        
        # Calculate local time within current phase
        phase_start_time = 0
        for j in range(phase_idx):
            phase_start_time += phase_durations[j]
        local_time = t_within_step - phase_start_time
        
        # Get current foot positions from robot state
        current_lf_pos = tsid_biped.get_placement_LF().translation.copy()
        current_rf_pos = tsid_biped.get_placement_RF().translation.copy()
        
        # Determine stance foot position for current step
        if swing_foot == "left":
            stance_foot_pos = current_rf_pos.copy()
            # Keep stance foot at ground level
        else:
            stance_foot_pos = current_lf_pos.copy()
            # Keep stance foot at ground level
        
        # Get target CoM position for current step
        if current_step < Num_Steps:
            target_com_pos = CoM_Positions[current_step].copy()
        else:
            target_com_pos = CoM_Positions[-1].copy()
        
        # Generate CoM reference based on phase
        if phase_idx == 0:  # Phase 1: Double support - move CoM to stance foot
            if local_time >= 0 and local_time <= phase_durations[0]:
                # Start from previous target or initial position
                if current_step == 0:
                    com_start = com_initial.copy()
                else:
                    com_start = CoM_Positions[current_step - 1].copy()
                
                # CoM target should be above stance foot at CoM height
                com_target = stance_foot_pos.copy()
                com_target[2] = CoM_Height
                
                com_phase_pos, com_phase_vel, com_phase_acc = compute_3rd_order_poly_traj(
                    com_start, com_target, phase_durations[0], conf.dt
                )
                local_idx = int(local_time / conf.dt)
                if local_idx < com_phase_pos.shape[1]:
                    com_pos_ref_current = com_phase_pos[:, local_idx]
                    com_vel_ref_current = com_phase_vel[:, local_idx]
                    com_acc_ref_current = com_phase_acc[:, local_idx]
                else:
                    com_pos_ref_current = com_target
                    com_vel_ref_current = np.zeros(3)
                    com_acc_ref_current = np.zeros(3)
            else:
                com_target = stance_foot_pos.copy()
                com_target[2] = CoM_Height
                com_pos_ref_current = com_target
                com_vel_ref_current = np.zeros(3)
                com_acc_ref_current = np.zeros(3)
                
        elif phase_idx == 1:  # Phase 2: Single support - swing foot
            # CoM stays over stance foot during swing
            com_target = stance_foot_pos.copy()
            com_target[2] = CoM_Height
            com_pos_ref_current = com_target
            com_vel_ref_current = np.zeros(3)
            com_acc_ref_current = np.zeros(3)
            
        elif phase_idx == 2:  # Phase 3: Double support - move CoM to target position
            if local_time >= 0 and local_time <= phase_durations[2]:
                # Start from CoM above stance foot
                com_start = stance_foot_pos.copy()
                com_start[2] = CoM_Height
                
                com_phase_pos, com_phase_vel, com_phase_acc = compute_3rd_order_poly_traj(
                    com_start, target_com_pos, phase_durations[2], conf.dt
                )
                local_idx = int(local_time / conf.dt)
                if local_idx < com_phase_pos.shape[1]:
                    com_pos_ref_current = com_phase_pos[:, local_idx]
                    com_vel_ref_current = com_phase_vel[:, local_idx]
                    com_acc_ref_current = com_phase_acc[:, local_idx]
                else:
                    com_pos_ref_current = target_com_pos
                    com_vel_ref_current = np.zeros(3)
                    com_acc_ref_current = np.zeros(3)
            else:
                com_pos_ref_current = target_com_pos
                com_vel_ref_current = np.zeros(3)
                com_acc_ref_current = np.zeros(3)
        else:
            com_pos_ref_current = target_com_pos
            com_vel_ref_current = np.zeros(3)
            com_acc_ref_current = np.zeros(3)
    else:
        # Stabilization phase: hold final position with zero velocity/acceleration
        com_pos_ref_current = CoM_Positions[-1].copy()  # Final target CoM position
        com_vel_ref_current = np.zeros(3)
        com_acc_ref_current = np.zeros(3)
        contact_phase_current = "double"
    
    # Handle contact phase changes (only during main trajectory)
    if i > 0 and i < N - 1:
        # Get previous phase for comparison
        t_prev = (i-1) * conf.dt
        prev_step_duration = np.sum(phase_durations)
        prev_current_step = int(t_prev // prev_step_duration)
        prev_t_within_step = t_prev % prev_step_duration
        
        prev_phase_idx = 0
        prev_cumulative_time = 0
        for j in range(len(phase_durations)):
            prev_cumulative_time += phase_durations[j]
            if prev_t_within_step < prev_cumulative_time:
                prev_phase_idx = j
                break
        
        # Determine previous swing foot
        if first_swing_foot == "left":
            prev_swing_foot = "left" if prev_current_step % 2 == 0 else "right"
        else:
            prev_swing_foot = "right" if prev_current_step % 2 == 0 else "left"
        
        # Determine previous contact pattern
        if prev_phase_idx == 1:  # Single support phase
            if prev_swing_foot == "left":
                contact_phase_prev = "right"  # Right stance, left swing
            else:
                contact_phase_prev = "left"   # Left stance, right swing
        else:
            contact_phase_prev = "double"
        
        if contact_phase_current != contact_phase_prev:
            print(f"Time {t:.3f} Step {current_step} Phase {phase_idx} Changing contact from {contact_phase_prev} to {contact_phase_current}, swing foot: {swing_foot}")
            if contact_phase_current == "left":
                # Entering single support - left foot stance, right foot swing
                print(f"  -> Adding LF contact, removing RF contact")
                tsid_biped.add_contact_LF()
                tsid_biped.remove_contact_RF()
            elif contact_phase_current == "right":
                # Entering single support - right foot stance, left foot swing
                print(f"  -> Adding RF contact, removing LF contact")
                tsid_biped.add_contact_RF()
                tsid_biped.remove_contact_LF()
            elif contact_phase_current == "double":
                # Entering double support - both feet in contact
                print(f"  -> Adding both LF and RF contacts")
                tsid_biped.add_contact_LF()
                tsid_biped.add_contact_RF()
    
    # Set CoM reference
    tsid_biped.set_com_ref(com_pos_ref_current, com_vel_ref_current, com_acc_ref_current)
    
    # Set foot references for swing foot during single support
    if i >= 0 and i < N and 'swing_foot' in locals() and 'phase_idx' in locals():
        if phase_idx == 1:  # Single support phase
            swing_duration = phase_durations[1]
            
            # Get swing foot initial and target positions
            if swing_foot == "left":
                swing_initial_pos = current_lf_pos.copy()
                if current_step < Num_Steps:
                    swing_target_pos = footstep_plan[current_step].copy()
                else:
                    swing_target_pos = swing_initial_pos.copy()
                    swing_target_pos[0] += stride_length
            else:  # right swing
                swing_initial_pos = current_rf_pos.copy()
                if current_step < Num_Steps:
                    swing_target_pos = footstep_plan[current_step].copy()
                else:
                    swing_target_pos = swing_initial_pos.copy()
                    swing_target_pos[0] += stride_length
            
            if local_time >= 0 and local_time <= swing_duration:
                # Calculate stride length for this step
                step_stride = np.linalg.norm(swing_target_pos[:2] - swing_initial_pos[:2])
                
                # Generate swing foot trajectory with proper landing at target position
                swing_foot_pos, swing_foot_vel, swing_foot_acc = generate_swing_foot_trajectory(
                    swing_initial_pos, step_stride, step_height, swing_duration, local_time
                )
                
                # Adjust trajectory to reach target position in all dimensions
                progress = local_time / swing_duration
                
                # X and Y coordinates: linear interpolation to target
                swing_foot_pos[:2] = swing_initial_pos[:2] + progress * (swing_target_pos[:2] - swing_initial_pos[:2])
                
                # Z coordinate: modify the trajectory to land at target z
                # The generate_swing_foot_trajectory assumes landing at initial_pos[2]
                # We need to adjust it to land at swing_target_pos[2]
                z_offset = swing_target_pos[2] - swing_initial_pos[2]
                
                # Apply the z offset with proper scaling based on trajectory phase
                if local_time <= swing_duration / 3.0:
                    # Lift phase: add offset proportionally
                    swing_foot_pos[2] += z_offset * progress * 3.0  # Scale by 3 since we're in first third
                elif local_time <= 2.0 * swing_duration / 3.0:
                    # Forward phase: maintain offset
                    swing_foot_pos[2] += z_offset
                else:
                    # Landing phase: the trajectory should naturally land at target
                    swing_foot_pos[2] += z_offset
                
                if swing_foot == "left":
                    tsid_biped.set_LF_3d_ref(swing_foot_pos, swing_foot_vel, swing_foot_acc)
                    # Debug print for left foot trajectory
                    if i % 50 == 0:  # More frequent debug prints
                        print(f"  Setting LF ref: [{swing_foot_pos[0]:.3f}, {swing_foot_pos[1]:.3f}, {swing_foot_pos[2]:.3f}], progress={progress:.3f}")
                        print(f"    Initial: [{swing_initial_pos[0]:.3f}, {swing_initial_pos[1]:.3f}, {swing_initial_pos[2]:.3f}]")
                        print(f"    Target:  [{swing_target_pos[0]:.3f}, {swing_target_pos[1]:.3f}, {swing_target_pos[2]:.3f}]")
                        print(f"    Z offset: {z_offset:.3f}")
                else:
                    tsid_biped.set_RF_3d_ref(swing_foot_pos, swing_foot_vel, swing_foot_acc)
                    # Debug print for right foot trajectory
                    if i % 50 == 0:  # More frequent debug prints
                        print(f"  Setting RF ref: [{swing_foot_pos[0]:.3f}, {swing_foot_pos[1]:.3f}, {swing_foot_pos[2]:.3f}], progress={progress:.3f}")
                        print(f"    Initial: [{swing_initial_pos[0]:.3f}, {swing_initial_pos[1]:.3f}, {swing_initial_pos[2]:.3f}]")
                        print(f"    Target:  [{swing_target_pos[0]:.3f}, {swing_target_pos[1]:.3f}, {swing_target_pos[2]:.3f}]")
                        print(f"    Z offset: {z_offset:.3f}")
            else:
                # End of swing - foot at target position
                if swing_foot == "left":
                    tsid_biped.set_LF_3d_ref(swing_target_pos, np.zeros(3), np.zeros(3))
                else:
                    tsid_biped.set_RF_3d_ref(swing_target_pos, np.zeros(3), np.zeros(3))
    
    
    # Solve the QP problem
    HQPData = tsid_biped.formulation.computeProblemData(t, q, v)
    sol = tsid_biped.solver.solve(HQPData)
    
    if sol.status != 0:
        print("QP problem could not be solved! Error code:", sol.status)
        break
    if norm(v, 2) > 10.0:
        print("Time %.3f Velocities are too high, stop everything!" % (t), norm(v))
        break
    
    # Get accelerations and integrate
    dv = tsid_biped.formulation.getAccelerations(sol)
    
    # Log data (only for non-negative indices)
    if i >= 0:
        com_pos[:, i] = tsid_biped.robot.com(tsid_biped.formulation.data())
        com_vel[:, i] = tsid_biped.robot.com_vel(tsid_biped.formulation.data())
        com_acc[:, i] = tsid_biped.comTask.getAcceleration(dv)
    
    # Print progress
    if i % 100 == 0:
        current_com = tsid_biped.robot.com(tsid_biped.formulation.data())
        current_com_vel = tsid_biped.robot.com_vel(tsid_biped.formulation.data())
        actual_lf_pos = tsid_biped.get_placement_LF().translation
        actual_rf_pos = tsid_biped.get_placement_RF().translation
        
        print("Time %.3f" % (t))
        if i >= 0 and i < N:
            print(f"  CoM ref: [{com_pos_ref_current[0]:.3f}, {com_pos_ref_current[1]:.3f}, {com_pos_ref_current[2]:.3f}]")
            print(f"  CoM vel ref: [{com_vel_ref_current[0]:.3f}, {com_vel_ref_current[1]:.3f}, {com_vel_ref_current[2]:.3f}]")
            
            # Print step and phase info
            if 'current_step' in locals() and 'phase_idx' in locals() and 'swing_foot' in locals():
                print(f"  Step {current_step}, Phase {phase_idx}, Swing foot: {swing_foot}")
                
                # Print foot reference targets if they were set
                if phase_idx == 1 and 'swing_target_pos' in locals():
                    if swing_foot == "left":
                        print(f"  LF target: [{swing_target_pos[0]:.3f}, {swing_target_pos[1]:.3f}, {swing_target_pos[2]:.3f}]")
                    else:
                        print(f"  RF target: [{swing_target_pos[0]:.3f}, {swing_target_pos[1]:.3f}, {swing_target_pos[2]:.3f}]")
                        
        elif i < 0:
            print(f"  CoM ref: [{com_pos_ref_current[0]:.3f}, {com_pos_ref_current[1]:.3f}, {com_pos_ref_current[2]:.3f}] (prep)")
            print(f"  CoM vel ref: [0.000, 0.000, 0.000] (prep)")
        else:
            print(f"  CoM ref: [{com_pos_ref_current[0]:.3f}, {com_pos_ref_current[1]:.3f}, {com_pos_ref_current[2]:.3f}] (stab)")
            print(f"  CoM vel ref: [0.000, 0.000, 0.000] (stab)")
            
        print(f"  CoM actual: [{current_com[0]:.3f}, {current_com[1]:.3f}, {current_com[2]:.3f}]")
        print(f"  CoM vel actual: [{current_com_vel[0]:.3f}, {current_com_vel[1]:.3f}, {current_com_vel[2]:.3f}]")
        print(f"  LF actual: [{actual_lf_pos[0]:.3f}, {actual_lf_pos[1]:.3f}, {actual_lf_pos[2]:.3f}]")
        print(f"  RF actual: [{actual_rf_pos[0]:.3f}, {actual_rf_pos[1]:.3f}, {actual_rf_pos[2]:.3f}]")
        
        # Print contact status
        lf_contact_active = tsid_biped.contact_LF_active
        rf_contact_active = tsid_biped.contact_RF_active
        print(f"  LF contact: {'ACTIVE' if lf_contact_active else 'BROKEN'}")
        print(f"  RF contact: {'ACTIVE' if rf_contact_active else 'BROKEN'}")
        
        print(f"  Tracking error: {norm(tsid_biped.comTask.position_error, 2):.3f}")
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
