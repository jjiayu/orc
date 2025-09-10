import time

import numpy as np
import pinocchio as pin

import talos_conf as conf
import vizutils
from tsid_biped import TsidBiped

### Main Part ###

print("#" * conf.LINE_WIDTH)
print(" Test Quasi Static Walking ".center(conf.LINE_WIDTH, '#'))
print("#" * conf.LINE_WIDTH)

tsid = TsidBiped(conf, conf.viewer)

tsid.q0[2] = 1.02127

com_0 = tsid.robot.com(tsid.formulation.data())
H_rf_0 = tsid.robot.framePosition(tsid.formulation.data(), tsid.model.getFrameId(conf.rf_frame_name))
H_lf_0 = tsid.robot.framePosition(tsid.formulation.data(), tsid.model.getFrameId(conf.lf_frame_name))

vizutils.addViewerSphere(tsid.viz, 'world/com', conf.SPHERE_RADIUS, conf.COM_SPHERE_COLOR)
vizutils.addViewerSphere(tsid.viz, 'world/com_ref', conf.REF_SPHERE_RADIUS, conf.COM_REF_SPHERE_COLOR)
vizutils.addViewerSphere(tsid.viz, 'world/rf', conf.SPHERE_RADIUS, conf.RF_SPHERE_COLOR)
vizutils.addViewerSphere(tsid.viz, 'world/rf_ref', conf.REF_SPHERE_RADIUS, conf.RF_REF_SPHERE_COLOR)
vizutils.addViewerSphere(tsid.viz, 'world/lf', conf.SPHERE_RADIUS, conf.LF_SPHERE_COLOR)
vizutils.addViewerSphere(tsid.viz, 'world/lf_ref', conf.REF_SPHERE_RADIUS, conf.LF_REF_SPHERE_COLOR)

# Display the robot initially
q, v = tsid.q, tsid.v
tsid.display(q)

# Update visualization markers
x_com = tsid.robot.com(tsid.formulation.data())
H_lf = tsid.robot.framePosition(tsid.formulation.data(), tsid.LF)
H_rf = tsid.robot.framePosition(tsid.formulation.data(), tsid.RF)

vizutils.applyViewerConfiguration(tsid.viz, 'world/com', x_com.tolist() + [0, 0, 0, 1.])
vizutils.applyViewerConfiguration(tsid.viz, 'world/com_ref', com_0.tolist() + [0, 0, 0, 1.])
vizutils.applyViewerConfiguration(tsid.viz, 'world/rf', pin.SE3ToXYZQUATtuple(H_rf))
vizutils.applyViewerConfiguration(tsid.viz, 'world/lf', pin.SE3ToXYZQUATtuple(H_lf))
vizutils.applyViewerConfiguration(tsid.viz, 'world/rf_ref', pin.SE3ToXYZQUATtuple(H_rf_0))
vizutils.applyViewerConfiguration(tsid.viz, 'world/lf_ref', pin.SE3ToXYZQUATtuple(H_lf_0))

print("Robot displayed! Check the visualizer at the URL above.")
print("Press Ctrl+C to exit.")

# Keep the program running so you can see the robot
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nExiting...")