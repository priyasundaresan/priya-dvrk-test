import robot
import pprint
import numpy as np
import PyKDL
import transform
import read_chessboard
import rigid_transform
import read_needle
import time
import math

"""
Usage:
Run 'python needle3d.py' to find the needle centers in view and get their 3d positions.
Then run 'python move_psm_test.py' to move the PSM to those centers and pick up the needles.
"""

USE_WORLD_TRANSFORM = False

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

def pickup(psm, points, z_upper, z_final):
	start, end = points[0], points[1]
	x1, y1, z1 = start[0], start[1], start[2]
	x2, y2, z2 = end[0], end[1], z_upper
	print("Moving to:")
	print(start)
	psm.move(PyKDL.Vector(x1, y1, z_upper))
	time.sleep(.25)
	psm.open_jaw()
	print("Lowering...")
	time.sleep(.25)
	psm.move(PyKDL.Vector(x1, y1, z1+0.002))
	time.sleep(.25)
	print("Grasping...")
	psm.close_jaw()
	time.sleep(.25)
	print("Grasped...")
	psm.move(PyKDL.Vector(x2, y2, z_upper))
	time.sleep(.25)
	print("Pulling...")
	psm.open_jaw()
	time.sleep(.25)


if __name__ == '__main__':

	psm2 = robot.robot('PSM2')
	kdl_pose = psm2.get_current_position().p
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

	""" Move to arbitrary start position (near upper left corner) & release anything gripper is
	holding. """
	home(psm2, pos, rot)
	
	""" Get PSM and endoscope calibration data (25 corresponding chess points) """
	psm2_calibration_data = list(transform.load_all('world/psm2_recordings.txt'))
	psm2_calibration_matrix = transform.psm_data_to_matrix(psm2_calibration_data)
	endoscope_calibration_matrix = np.matrix(list(read_chessboard.load_all('world/endoscope_chesspts.p'))[0])

	""" Get the coordinates of most recently found needle centers (in endoscope frame) """
	needle_points = np.matrix(list(read_needle.load_all('needle_data/needle_points.p'))[0])

	if USE_WORLD_TRANSFORM:

		world = transform.generate_world()

		TE_W = rigid_transform.solve_for_rigid_transformation(endoscope_calibration_matrix, world)
		needle_to_world = transform.transform_data("Endoscope", "World", needle_points, TE_W)
		needle_to_world[:,2] = 0.
		pprint.pprint(needle_to_world)

		TW_2 = rigid_transform.solve_for_rigid_transformation(world, psm2_calibration_matrix)
		world_to_psm2 = transform.transform_data("World", "PSM2", needle_to_world, TW_2)
		pprint.pprint(world_to_psm2)

		""" Move to needle centers, pcik them up, and release them """
		pickup(psm2, world_to_psm2.tolist(), z_upper, z_lower)

	else:
		""" Solve for the transform between endoscope to PSM2 """
		TE_2 = rigid_transform.solve_for_rigid_transformation(endoscope_calibration_matrix, psm2_calibration_matrix)
		needle_to_psm2 = transform.transform_data("Endoscope", "PSM2",needle_points, TE_2)

		""" Move to needle centers, pcik them up, and release them """
		pickup(psm2, needle_to_psm2.tolist(), z_upper, z_lower)

	home(psm2, pos, rot)
