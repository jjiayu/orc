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
contact_patterns = ["double", "left", "double"] #variables mean contact configuration
phase_durations = np.array([3.0, 6.0, 3.0])
stride_length = 0.1

#Let us just make one step
N = int(np.sum(phase_durations)/conf.dt) #1500 #data["com"].shape[1] #Number of Time steps, means 3s duration
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
com_above_left = com_initial.copy()
com_above_left[0] = lf_current[0]  # X position of left foot
com_above_left[1] = lf_current[1]  # Y position of left foot

com_middle = com_initial.copy()  # Return to middle between feet
com_x_offset = 0.021
com_middle[0] = (lf_current[0] + rf_current[0]) / 2 + com_x_offset
com_middle[1] = (lf_current[1] + rf_current[1]) / 2

# Stepping parameters
step_height = 0.12  # Maximum foot lift height
stride_length = 0.1  # Forward step distance
swing_foot_initial_pos = rf_current.copy()  # Right foot will be the swing foot
stance_foot_pos = lf_current.copy()  # Left foot is stance foot

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
        
        # Find which phase we're in
        phase_idx = 0
        for j in range(len(phase_durations)):
            if t_traj < phase_end_times[j]:
                phase_idx = j
                break
        
        contact_phase_current = contact_patterns[phase_idx]
        
        # Generate CoM reference based on phase
        if phase_idx == 0:  # Phase 1: Double support - move to left foot
            local_time = t_traj
            if local_time <= phase_durations[0]:
                # Compute 3rd order polynomial trajectory for this phase
                com_phase1_pos, com_phase1_vel, com_phase1_acc = compute_3rd_order_poly_traj(
                    com_initial, com_above_left, phase_durations[0], conf.dt
                )
                local_idx = int(local_time / conf.dt)
                if local_idx < com_phase1_pos.shape[1]:
                    com_pos_ref_current = com_phase1_pos[:, local_idx]
                    com_vel_ref_current = com_phase1_vel[:, local_idx]
                    com_acc_ref_current = com_phase1_acc[:, local_idx]
                else:
                    com_pos_ref_current = com_above_left
                    com_vel_ref_current = np.zeros(3)
                    com_acc_ref_current = np.zeros(3)
            else:
                com_pos_ref_current = com_above_left
                com_vel_ref_current = np.zeros(3)
                com_acc_ref_current = np.zeros(3)
                
        elif phase_idx == 1:  # Phase 2: Left support - swing right foot
            com_pos_ref_current = com_above_left
            com_vel_ref_current = np.zeros(3)
            com_acc_ref_current = np.zeros(3)
            
            # Generate swing foot trajectory
            phase_start_time = phase_durations[0]
            local_time = t_traj - phase_start_time
            swing_duration = phase_durations[1]
            
            if local_time >= 0 and local_time <= swing_duration:
                # Calculate swing foot position
                swing_foot_pos, swing_foot_vel, swing_foot_acc = generate_swing_foot_trajectory(
                    swing_foot_initial_pos, stride_length, step_height, swing_duration, local_time
                )
            else:
                # End of swing - foot should be on ground at new position
                final_pos = swing_foot_initial_pos.copy()
                final_pos[0] += stride_length  # Move forward by stride length
                swing_foot_pos = final_pos
                swing_foot_vel = np.zeros(3)
                swing_foot_acc = np.zeros(3)
            
        elif phase_idx == 2:  # Phase 3: Double support - move to middle
            phase_start_time = phase_durations[0] + phase_durations[1]
            local_time = t_traj - phase_start_time
            
            if local_time >= 0 and local_time <= phase_durations[2]:
                # Compute 3rd order polynomial trajectory for this phase
                com_phase3_pos, com_phase3_vel, com_phase3_acc = compute_3rd_order_poly_traj(
                    com_above_left, com_middle, phase_durations[2], conf.dt
                )
                local_idx = int(local_time / conf.dt)
                if local_idx < com_phase3_pos.shape[1]:
                    com_pos_ref_current = com_phase3_pos[:, local_idx]
                    com_vel_ref_current = com_phase3_vel[:, local_idx]
                    com_acc_ref_current = com_phase3_acc[:, local_idx]
                else:
                    com_pos_ref_current = com_middle
                    com_vel_ref_current = np.zeros(3)
                    com_acc_ref_current = np.zeros(3)
            else:
                com_pos_ref_current = com_middle
                com_vel_ref_current = np.zeros(3)
                com_acc_ref_current = np.zeros(3)
        else:
            com_pos_ref_current = com_middle
            com_vel_ref_current = np.zeros(3)
            com_acc_ref_current = np.zeros(3)
    else:
        # Stabilization phase: hold final position with zero velocity/acceleration
        com_pos_ref_current = com_middle
        com_vel_ref_current = np.zeros(3)
        com_acc_ref_current = np.zeros(3)
        contact_phase_current = "double"
    
    # Handle contact phase changes (only during main trajectory)
    if i > 0 and i < N - 1:
        # Get previous phase for comparison
        t_prev = (i-1) * conf.dt
        phase_idx_prev = 0
        for j in range(len(phase_durations)):
            if t_prev < phase_end_times[j]:
                phase_idx_prev = j
                break
        contact_phase_prev = contact_patterns[phase_idx_prev]
        
        if contact_phase_current != contact_phase_prev:
            print(f"Time {t:.3f} Changing contact phase from {contact_phase_prev} to {contact_phase_current}")
            if contact_phase_current == "left":
                # Entering single support - left foot stance, right foot swing
                tsid_biped.add_contact_LF()
                tsid_biped.remove_contact_RF()
            elif contact_phase_current == "double":
                # Entering double support - both feet in contact
                tsid_biped.add_contact_LF()
                tsid_biped.add_contact_RF()
    
    # Set CoM reference
    tsid_biped.set_com_ref(com_pos_ref_current, com_vel_ref_current, com_acc_ref_current)
    
    # Set foot references
    if i >= 0 and i < N:
        t_traj = i * conf.dt
        phase_idx = 0
        for j in range(len(phase_durations)):
            if t_traj < phase_end_times[j]:
                phase_idx = j
                break
        
        if phase_idx == 1:  # Single support phase - swing right foot
            phase_start_time = phase_durations[0]
            local_time = t_traj - phase_start_time
            swing_duration = phase_durations[1]
            
            if local_time >= 0 and local_time <= swing_duration:
                swing_foot_pos, swing_foot_vel, swing_foot_acc = generate_swing_foot_trajectory(
                    swing_foot_initial_pos, stride_length, step_height, swing_duration, local_time
                )
                tsid_biped.set_RF_3d_ref(swing_foot_pos, swing_foot_vel, swing_foot_acc)
            else:
                # End of swing - foot at final position
                final_pos = swing_foot_initial_pos.copy()
                final_pos[0] += stride_length
                tsid_biped.set_RF_3d_ref(final_pos, np.zeros(3), np.zeros(3))
    
    
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
        print("Time %.3f" % (t))
        if i >= 0 and i < N:
            print(f"  CoM ref: [{com_pos_ref_current[0]:.3f}, {com_pos_ref_current[1]:.3f}, {com_pos_ref_current[2]:.3f}]")
            print(f"  CoM vel ref: [{com_vel_ref_current[0]:.3f}, {com_vel_ref_current[1]:.3f}, {com_vel_ref_current[2]:.3f}]")
        elif i < 0:
            print(f"  CoM ref: [{com_pos_ref_current[0]:.3f}, {com_pos_ref_current[1]:.3f}, {com_pos_ref_current[2]:.3f}] (prep)")
            print(f"  CoM vel ref: [0.000, 0.000, 0.000] (prep)")
        else:
            print(f"  CoM ref: [{com_pos_ref_current[0]:.3f}, {com_pos_ref_current[1]:.3f}, {com_pos_ref_current[2]:.3f}] (stab)")
            print(f"  CoM vel ref: [0.000, 0.000, 0.000] (stab)")
        print(f"  CoM actual: [{current_com[0]:.3f}, {current_com[1]:.3f}, {current_com[2]:.3f}]")
        print(f"  CoM vel actual: [{current_com_vel[0]:.3f}, {current_com_vel[1]:.3f}, {current_com_vel[2]:.3f}]")
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
