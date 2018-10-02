import robot
import pprint
import numpy as np
import PyKDL
import read_psm_data
import read_chessboard_data
import rigid_transform_test
import read_needle_points
import time
import math

"""
Usage:
Run 'python needle3d.py' to find the needle centers in view and get their 3d positions.
Then run 'python move_psm_test.py' to move the PSM to those centers and pick up the needles.
"""

if __name__ == '__main__':

	psm2 = robot.robot('PSM2')
	kdl_pose = psm2.get_current_position().p
	print("Current Position:")
	pprint.pprint(kdl_pose)

	""" An arbitrary z starting point that is not too close/far from the platform.
	When the gripper picks up a needle, it moves up to this point and then releases the needle.
	Change if it is too high/low above the platform. """
	z_start = -0.111688

	""" Where PSM2 touches the platform """
	z_limit = -0.1233

	""" POSE PSM2 WAS CALIBRATED IN """
	pos = PyKDL.Vector(-0.118749, 0.0203151, -0.111688)
	rot = PyKDL.Rotation(-0.988883, -0.00205771,   -0.148682,
						-0.00509171,    0.999786,   0.0200282,
						 0.148609,   0.0205626,   -0.988682)

	start = PyKDL.Frame(rot, pos)

	""" Move to arbitrary start position (near upper left corner) & release anything gripper is
	holding. """
	print("Moving to start position")
	psm2.open_jaw()
	time.sleep(.5)
	psm2.move(start)
	time.sleep(.5)
	psm2.close_jaw()
	time.sleep(.5)

	""" Get PSM and endoscope calibration data (25 corresponding chess points) """
	psm2_calibration_data = list(read_psm_data.load_all('calibration/psm2_recordings.txt'))
	psm2_calibration_matrix = read_psm_data.psm_data_to_matrix(psm2_calibration_data)
	endoscope_calibration_matrix = np.matrix(list(read_chessboard_data.load_all('calibration/endoscope_chesspts.p'))[0])

	""" Get the coordinates of most recently found needle centers (in endoscope frame) """
	needle_points = np.matrix(list(read_needle_points.load_all('needle_data/needle_points.p'))[0])

	""" Solve for the transform between endoscope to PSM2 """
	TE_2 = rigid_transform_test.solve_for_rigid_transformation(endoscope_calibration_matrix, psm2_calibration_matrix)
	print('\nTransforming Needle Points Endoscope Frame --> PSM2 Frame')
	print('            x           y           z')
	needle_to_psm2 = read_psm_data.transform_matrix(needle_points, TE_2)
	print(needle_to_psm2)

	""" Verbose test for moving the PSM to needle centers, picking them up, and releasing them """
	for point in needle_to_psm2.tolist():
		print("Moving to", point)
		psm2.move(PyKDL.Vector(point[0], point[1], z_start))
		time.sleep(.5)
		psm2.open_jaw()
		print("Lowering...")
		time.sleep(.25)
		psm2.move(PyKDL.Vector(point[0], point[1], z_limit))
		time.sleep(.25)
		print("Grasping...")
		psm2.close_jaw()
		time.sleep(.25)
		print("Grasped...")
		psm2.move(PyKDL.Vector(point[0], point[1], z_start))
		time.sleep(.25)
		print("Releasing...")
		psm2.open_jaw()
		time.sleep(.25)
