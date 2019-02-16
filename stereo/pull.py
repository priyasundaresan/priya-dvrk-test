import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import robot
import pprint
import numpy as np
import PyKDL
import transform
import read_camera
import rigid_transform
import read_needle
import time
import math

"""
Usage:
Run 'python needle3d.py' to find the needle centers in view and get their 3d positions.
Then run 'python move_psm_test.py' to move the PSM to those centers and pick up the needles.
"""

def home(psm, pos, rot):
	""" Move to arbitrary start position (near upper left corner) & release anything gripper is
	holding. """
	print("Moving to start position...")
	start = PyKDL.Frame(rot, pos)
	psm.open_jaw()
	time.sleep(.25)
	psm.move(start)
	time.sleep(.25)
	psm.close_jaw()
	time.sleep(.25)

def pull(psm, points, z_upper, z_final):
	time.sleep(3)
	points = [points[i:i + 2] for i in range(0, len(points), 2)]
	for pair in points:
		start, end = pair[0], pair[1]
		x1, y1, z1 = start[0], start[1], start[2]
		x2, y2, z2 = end[0], end[1], end[2]
		# print("Moving to:")
		# print(start)
		# psm.move(PyKDL.Vector(x1, y1, z_upper))
		# time.sleep(.25)
		psm.open_jaw()
		print("Lowering...")
		time.sleep(.25)
		psm.move(PyKDL.Vector(x1, y1, z1))
		time.sleep(.25)
		print("Grasping...")
		psm.close_jaw()
		time.sleep(.25)
		print("Grasped...")
		psm.move(PyKDL.Vector(x2, y2, z2))
		time.sleep(.25)
		print("Pulling...")
		psm.open_jaw()
		time.sleep(.25)
		psm.move(PyKDL.Vector(x1, y1, z_upper))
		time.sleep(.25)


if __name__ == '__main__':

	psm2 = robot.robot('PSM2')
	kdl_pose = psm2.get_current_position()
	print("Current Position:")
	pprint.pprint(kdl_pose)

	""" An arbitrary z starting point that is not too close/far from the platform.
	When the gripper picks up a needle, it moves up to this point and then releases the needle.
	Change if it is too high/low above the platform. """
	z_upper = -0.08
	# z_final = -0.06

	""" Where PSM2 touches the platform """
	# For white background:
	z_lower = -0.1233
	# For phantom background:
	# z_lower = -0.128

	""" POSE PSM2 WAS CALIBRATED IN """
	pos = PyKDL.Vector(-0.118749, 0.0203151, -0.081688)
	rot = PyKDL.Rotation(-0.988883, -0.00205771,   -0.148682,
						-0.00509171,    0.999786,   0.0200282,
						 0.148609,   0.0205626,   -0.988682)

	pos2 = PyKDL.Vector(-0.0972128,  -0.0170138,   -0.106974)
	sideways = PyKDL.Rotation(  -0.453413,    0.428549,   -0.781513,
     							-0.17203,    0.818259,    0.548505,
     							0.874541,    0.383143,   -0.297286)


	""" Move to arbitrary start position (near upper left corner) & release anything gripper is
	holding. """
	# home(psm2, pos, rot)
	home(psm2, pos2, sideways)
	
	""" Get PSM and endoscope calibration data (25 corresponding chess points) """
	psm2_calibration_data = list(transform.load_all('../utils/psm2_recordings.txt'))
	psm2_calibration_matrix = transform.fit_to_plane(transform.psm_data_to_matrix(psm2_calibration_data))
	endoscope_calibration_matrix = transform.fit_to_plane(np.matrix(list(read_camera.load_all('../camera_data/endoscope_chesspts.p'))[0]))

	""" Get the coordinates of most recently found needle centers (in endoscope frame) """
	needle_points = np.matrix(list(read_needle.load_all('needle_data/needle_points.p'))[0])

	""" Solve for the transform between endoscope to PSM2 """
	TE_2 = transform.get_transform("Endoscope", "PSM2", endoscope_calibration_matrix, psm2_calibration_matrix)
	needle_to_psm2 = transform.transform_data("Endoscope", "PSM2", needle_points, TE_2)
	pprint.pprint(needle_to_psm2)

	""" Move to needle centers, pcik them up, and release them """
	pull(psm2, needle_to_psm2.tolist(), z_upper, z_lower)

	home(psm2, pos2, rot)
