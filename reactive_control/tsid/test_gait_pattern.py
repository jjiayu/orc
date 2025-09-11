#!/usr/bin/env python3
"""
Test script to verify the new gait pattern and path generation
"""
import numpy as np

# Simulate the gait pattern generation logic
Num_Steps = 2
steps_phases = 3

# Create gait pattern: [double, swing_foot, double] for each step
gait_pattern = []
for step in range(Num_Steps):
    gait_pattern.extend(["double", "right" if step % 2 == 0 else "left", "double"])

print("Gait pattern:", gait_pattern)

# Phase durations: 3 phases per step, repeated for all steps
base_phase_durations = [3.0, 6.0, 3.0]  # [double, stance, double]
phase_durations = base_phase_durations * Num_Steps
print("Phase durations:", phase_durations)
print("Total phases:", len(phase_durations))

# Mock initial positions
CoM_Height = 0.6
current_com = np.array([0.0, 0.0, CoM_Height])
current_lf_pos = np.array([0.0, 0.096, 0.07])
current_rf_pos = np.array([0.0, -0.096, 0.07])

print(f"Initial CoM: [{current_com[0]:.3f}, {current_com[1]:.3f}, {current_com[2]:.3f}]")
print(f"Initial LF:  [{current_lf_pos[0]:.3f}, {current_lf_pos[1]:.3f}, {current_lf_pos[2]:.3f}]")
print(f"Initial RF:  [{current_rf_pos[0]:.3f}, {current_rf_pos[1]:.3f}, {current_rf_pos[2]:.3f}]")

# Define target footstep positions for each step
footstep_targets = [
    np.array([0.11, -0.096, 0.07]),  # Step 0: Right foot target
    np.array([0.21, 0.096, 0.07])    # Step 1: Left foot target
]

# Create paths
com_path = []
lf_path = []
rf_path = []

# Initialize positions
com_pos = current_com.copy()
lf_pos = current_lf_pos.copy()
rf_pos = current_rf_pos.copy()

for phase_idx, contact_phase in enumerate(gait_pattern):
    step_number = phase_idx // 3
    phase_in_step = phase_idx % 3
    
    print(f"\nPhase {phase_idx}: {contact_phase} (Step {step_number}, Phase {phase_in_step})")
    
    # Store starting positions for this phase
    com_start = com_pos.copy()
    lf_start = lf_pos.copy()
    rf_start = rf_pos.copy()
    
    if phase_in_step == 0:  # First double support phase of a step
        # CoM moves towards the stance foot (which will be active in next phase)
        if step_number < len(footstep_targets):
            if step_number % 2 == 0:  # Right foot will be stance
                stance_foot_pos = rf_pos.copy()
            else:  # Left foot will be stance  
                stance_foot_pos = lf_pos.copy()
            
            # CoM target: above stance foot
            com_target = stance_foot_pos.copy()
            com_target[2] = CoM_Height
        else:
            com_target = com_pos.copy()  # Stay in place
            
        # Feet don't move during double support
        lf_target = lf_pos.copy()
        rf_target = rf_pos.copy()
        
    elif phase_in_step == 1:  # Single support phase (stance phase)
        # CoM stays above stance foot
        com_target = com_pos.copy()
        
        # Swing foot moves to target position
        if step_number < len(footstep_targets):
            if step_number % 2 == 0:  # Right foot swings
                rf_target = footstep_targets[step_number].copy()
                lf_target = lf_pos.copy()  # Stance foot stays
            else:  # Left foot swings
                lf_target = footstep_targets[step_number].copy()
                rf_target = rf_pos.copy()  # Stance foot stays
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

# Test interpolation for a few time points
print(f"\n=== Testing Interpolation ===")
total_time = sum(phase_durations)
dt = 0.1

for test_time in [0.0, 1.5, 4.5, 7.5, 10.5, 13.5]:
    if test_time > total_time:
        continue
        
    # Find which phase we're currently in
    cumulative_time = 0
    current_phase_idx = 0
    for phase_idx in range(len(phase_durations)):
        if test_time < cumulative_time + phase_durations[phase_idx]:
            current_phase_idx = phase_idx
            break
        cumulative_time += phase_durations[phase_idx]
    
    # Calculate local time within current phase
    local_time = test_time - cumulative_time
    phase_duration = phase_durations[current_phase_idx]
    progress = min(1.0, max(0.0, local_time / phase_duration))
    
    # Get contact pattern for current phase
    contact_phase_current = gait_pattern[current_phase_idx]
    
    # Interpolate CoM position
    com_start = com_path[current_phase_idx][0]
    com_end = com_path[current_phase_idx][1]
    com_pos_ref = com_start + progress * (com_end - com_start)
    
    print(f"Time {test_time:.1f}s: Phase {current_phase_idx} ({contact_phase_current}), Progress {progress:.2f}")
    print(f"  CoM ref: [{com_pos_ref[0]:.3f}, {com_pos_ref[1]:.3f}, {com_pos_ref[2]:.3f}]")
