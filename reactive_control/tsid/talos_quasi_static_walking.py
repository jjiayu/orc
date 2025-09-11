import time

import orc.optimal_control.lipm.biped.romeo_conf as conf
# import talos_conf as conf
import matplotlib.pyplot as plt
import numpy as np
import orc.utils.plot_utils as plut
from numpy import nan
from numpy.linalg import norm as norm
from tsid_biped import TsidBiped
from orc.optimal_control.lipm.biped.lipm_to_tsid import compute_5th_order_poly_traj, generate_swing_foot_trajectory

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


# ============================================================================
# NEW IMPLEMENTATION: Gait Pattern with Start/End Position Interpolation
# ============================================================================

# Define number of steps and gait pattern
Num_Steps = 2
steps_phases = 3  # Each step has 3 phases: double -> stance -> double

# Create gait pattern: [double, swing_foot, double] for each step
# Alternate between left and right swing foot for each step
first_swing_foot = "left"
gait_pattern = []
for step in range(Num_Steps):
    # Start with first_swing_foot, then alternate
    if first_swing_foot == "left":
        swing_foot = "left" if step % 2 == 0 else "right"
    else:
        swing_foot = "right" if step % 2 == 0 else "left"
    gait_pattern.extend(["double", swing_foot, "double"])

print("Gait pattern:", gait_pattern)

# Define target footstep positions for each step
# These correspond to the swing foot positions for each step
if first_swing_foot == "left":
    footstep_targets = [
        np.array([0.11, 0.096, 0.07]),   # Step 0: Left foot target
        np.array([0.21, -0.096, 0.07])   # Step 1: Right foot target
    ]
else:
    footstep_targets = [
        np.array([0.11, -0.096, 0.07]),  # Step 0: Right foot target
        np.array([0.21, 0.096, 0.07])    # Step 1: Left foot target
    ]

print("Footstep targets:", footstep_targets)

# Phase durations: 3 phases per step, repeated for all steps
base_phase_durations = [3.0, 6.0, 3.0]  # [double, stance, double]
phase_durations = base_phase_durations * Num_Steps
print("Phase durations:", phase_durations)
print("Total phases:", len(phase_durations))

# Get current robot state for initial positions
current_com = tsid_biped.robot.com(tsid_biped.formulation.data()).copy()
current_com[2] = CoM_Height  # Set desired CoM height
current_lf_pos = tsid_biped.get_placement_LF().translation.copy()
current_rf_pos = tsid_biped.get_placement_RF().translation.copy()

print(f"Initial CoM: [{current_com[0]:.3f}, {current_com[1]:.3f}, {current_com[2]:.3f}]")
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
            else:  # Left foot will swing, so right foot is support
                support_foot_pos = rf_pos.copy()
            
            # CoM target: above support foot
            com_target = support_foot_pos.copy()
            com_target[2] = CoM_Height
        else:
            com_target = com_pos.copy()  # Stay in place
            
        # Feet don't move during double support
        lf_target = lf_pos.copy()
        rf_target = rf_pos.copy()
        
    elif phase_in_step == 1:  # Single support phase (swing phase)
        # CoM stays above support foot
        com_target = com_pos.copy()
        
        # Swing foot moves to target position
        if step_number < len(footstep_targets):
            swing_foot = contact_phase  # Current phase tells us which foot swings
            if swing_foot == "right":  # Right foot swings, left foot supports
                rf_target = footstep_targets[step_number].copy()
                lf_target = lf_pos.copy()  # Support foot stays
            else:  # Left foot swings, right foot supports
                lf_target = footstep_targets[step_number].copy()
                rf_target = rf_pos.copy()  # Support foot stays
        else:
            lf_target = lf_pos.copy()
            rf_target = rf_pos.copy()
            
    else:  # phase_in_step == 2: Second double support phase
        # CoM moves to midpoint between feet for next step preparation
        if step_number + 1 < len(footstep_targets):
            # Calculate midpoint between current feet positions for next step
            midpoint = (lf_pos + rf_pos) / 2.0
            com_target = midpoint.copy()
            com_target[2] = CoM_Height
        else:
            # Final phase: stay at current position
            com_target = com_pos.copy()
            
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
    
    print(f"  CoM: [{com_start[0]:.3f}, {com_start[1]:.3f}, {com_start[2]:.3f}] -> [{com_target[0]:.3f}, {com_target[1]:.3f}, {com_target[2]:.3f}]")
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

print(f"Total duration: {total_duration:.1f}s, N: {N}, N_pre: {N_pre}, N_post: {N_post}")

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
    
    # NEW INTERPOLATION-BASED REFERENCE GENERATION
    if i < 0:
        # Preparation phase: hold initial position with zero velocity/acceleration
        com_pos_ref_current = current_com.copy()
        com_vel_ref_current = np.zeros(3)
        com_acc_ref_current = np.zeros(3)
        lf_pos_ref_current = current_lf_pos.copy()
        rf_pos_ref_current = current_rf_pos.copy()
        lf_vel_ref_current = np.zeros(3)
        rf_vel_ref_current = np.zeros(3)
        lf_acc_ref_current = np.zeros(3)
        rf_acc_ref_current = np.zeros(3)
        contact_phase_current = "double"
    elif i < N:
        # Main trajectory phase - use interpolation between start/end positions
        t_traj = i * conf.dt  # Trajectory time (starts from 0)
        
        # Find which phase we're currently in
        cumulative_time = 0
        current_phase_idx = 0
        phase_start_time = 0
        
        for phase_idx in range(len(phase_durations)):
            phase_end_time = cumulative_time + phase_durations[phase_idx]
            if t_traj < phase_end_time:
                current_phase_idx = phase_idx
                phase_start_time = cumulative_time
                break
            cumulative_time = phase_end_time
        
        # Ensure we don't exceed the available phases
        if current_phase_idx >= len(gait_pattern):
            current_phase_idx = len(gait_pattern) - 1
            phase_start_time = cumulative_time - phase_durations[current_phase_idx]
        
        # Calculate local time within current phase
        local_time = t_traj - phase_start_time
        phase_duration = phase_durations[current_phase_idx]
        progress = min(1.0, max(0.0, local_time / phase_duration))  # Clamp between 0 and 1
        
        # Get contact pattern for current phase
        contact_phase_current = gait_pattern[current_phase_idx]
        
        # Interpolate CoM position
        com_start = com_path[current_phase_idx][0]
        com_end = com_path[current_phase_idx][1]
        com_pos_ref_current = com_start + progress * (com_end - com_start)
        com_vel_ref_current = (com_end - com_start) / phase_duration if phase_duration > 0 else np.zeros(3)
        com_acc_ref_current = np.zeros(3)  # Simple interpolation, no acceleration
        
        # Interpolate left foot position
        lf_start = lf_path[current_phase_idx][0]
        lf_end = lf_path[current_phase_idx][1]
        lf_pos_ref_current = lf_start + progress * (lf_end - lf_start)
        lf_vel_ref_current = (lf_end - lf_start) / phase_duration if phase_duration > 0 else np.zeros(3)
        lf_acc_ref_current = np.zeros(3)
        
        # Interpolate right foot position  
        rf_start = rf_path[current_phase_idx][0]
        rf_end = rf_path[current_phase_idx][1]
        rf_pos_ref_current = rf_start + progress * (rf_end - rf_start)
        rf_vel_ref_current = (rf_end - rf_start) / phase_duration if phase_duration > 0 else np.zeros(3)
        rf_acc_ref_current = np.zeros(3)
    else:
        # Stabilization phase: hold final position with zero velocity/acceleration
        final_com = com_path[-1][1] if com_path else current_com.copy()
        final_lf = lf_path[-1][1] if lf_path else current_lf_pos.copy()
        final_rf = rf_path[-1][1] if rf_path else current_rf_pos.copy()
        
        com_pos_ref_current = final_com
        com_vel_ref_current = np.zeros(3)
        com_acc_ref_current = np.zeros(3)
        lf_pos_ref_current = final_lf
        rf_pos_ref_current = final_rf
        lf_vel_ref_current = np.zeros(3)
        rf_vel_ref_current = np.zeros(3)
        lf_acc_ref_current = np.zeros(3)
        rf_acc_ref_current = np.zeros(3)
        contact_phase_current = "double"
    
    # Handle contact phase changes
    if i > 0 and i < N:
        # Get previous contact phase for comparison
        if i == 1:  # First iteration of main trajectory
            contact_phase_prev = "double"
        else:
            # Calculate previous phase
            t_prev = (i-1) * conf.dt
            cumulative_time_prev = 0
            prev_phase_idx = 0
            for phase_idx in range(len(phase_durations)):
                if t_prev < cumulative_time_prev + phase_durations[phase_idx]:
                    prev_phase_idx = phase_idx
                    break
                cumulative_time_prev += phase_durations[phase_idx]
            
            if prev_phase_idx >= len(gait_pattern):
                prev_phase_idx = len(gait_pattern) - 1
            contact_phase_prev = gait_pattern[prev_phase_idx]
        
        # Handle contact transitions
        if contact_phase_current != contact_phase_prev:
            print(f"Time {t:.3f} Phase {current_phase_idx}: Changing contact from {contact_phase_prev} to {contact_phase_current}")
            
            if contact_phase_current == "left":
                # Left foot stance, right foot swing
                print(f"  -> Adding LF contact, removing RF contact")
                tsid_biped.add_contact_LF()
                tsid_biped.remove_contact_RF()
            elif contact_phase_current == "right":
                # Right foot stance, left foot swing
                print(f"  -> Adding RF contact, removing LF contact")
                tsid_biped.add_contact_RF()
                tsid_biped.remove_contact_LF()
            elif contact_phase_current == "double":
                # Both feet in contact
                print(f"  -> Adding both LF and RF contacts")
                tsid_biped.add_contact_LF()
                tsid_biped.add_contact_RF()
    
    # Set references to the robot
    tsid_biped.set_com_ref(com_pos_ref_current, com_vel_ref_current, com_acc_ref_current)
    
    # Set foot references (only for swing foot during single support)
    if i >= 0 and i < N and contact_phase_current in ["left", "right"]:
        # Single support phase - set swing foot reference
        if contact_phase_current == "left":
            # Left foot stance, right foot swings
            tsid_biped.set_RF_3d_ref(rf_pos_ref_current, rf_vel_ref_current, rf_acc_ref_current)
        else:
            # Right foot stance, left foot swings  
            tsid_biped.set_LF_3d_ref(lf_pos_ref_current, lf_vel_ref_current, lf_acc_ref_current)
    
    
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
            
            # Print phase info for new implementation
            if 'current_phase_idx' in locals() and 'contact_phase_current' in locals():
                print(f"  Phase {current_phase_idx}: {contact_phase_current}, Progress: {progress:.3f}")
                
                # Print foot reference targets
                if contact_phase_current in ["left", "right"]:
                    if contact_phase_current == "left":
                        print(f"  RF target: [{rf_pos_ref_current[0]:.3f}, {rf_pos_ref_current[1]:.3f}, {rf_pos_ref_current[2]:.3f}]")
                    else:
                        print(f"  LF target: [{lf_pos_ref_current[0]:.3f}, {lf_pos_ref_current[1]:.3f}, {lf_pos_ref_current[2]:.3f}]")
                        
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
