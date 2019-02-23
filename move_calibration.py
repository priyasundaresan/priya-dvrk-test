import os,sys
sys.path.insert(1, os.path.join(sys.path[0], './utils'))
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
	return

def move_to(psm, points, z_upper):
	for point in points:
		x, y, z = point[0], point[1], point[2]
		print("Moving to:")
		print(point)
		psm.move(PyKDL.Vector(x, y, z_upper))
		time.sleep(.25)
		psm.move(PyKDL.Vector(x, y, z))
		time.sleep(.25)
		psm.move(PyKDL.Vector(x, y, z_upper))
		time.sleep(.25)


if __name__ == '__main__':

	psm2 = robot.robot('PSM2')
	kdl_pose = psm2.get_current_position().p
	print("Current Position:")
	pprint.pprint(kdl_pose)
	pprint.pprint(psm2.get_current_position().M)

	""" An arbitrary z starting point that is not too close/far from the platform.
	When the gripper picks up a needle, it moves up to this point and then releases the needle.
	Change if it is too high/low above the platform. """
	z_upper = -0.112688

	""" POSE PSM2 WAS CALIBRATED IN """
	pos = PyKDL.Vector(-0.118749, 0.0203151, -0.111688)
	rot = PyKDL.Rotation(-0.940475,    0.313929,   -0.130214,
						0.330906,    0.933198,   -0.140157,
						 0.0775163,   -0.174902,    -0.98153)

	pos2 = PyKDL.Vector(-0.0972128,  -0.0170138,   -0.106974)
	sideways = PyKDL.Rotation(  -0.453413,    0.428549,   -0.781513,
     							-0.17203,    0.818259,    0.548505,
     							0.874541,    0.383143,   -0.297286)

	""" Move to arbitrary start position (near upper left corner) & release anything gripper is
	holding. """
	home(psm2, pos2, sideways)
	
	""" Get PSM and endoscope calibration data (25 corresponding chess points) """
	psm2_calibration_data = list(transform.load_all('utils/psm2_recordings.txt'))
	psm2_calibration_matrix = transform.fit_to_plane(transform.psm_data_to_matrix(psm2_calibration_data))
	endoscope_calibration_matrix = transform.fit_to_plane(np.matrix(list(read_camera.load_all('camera_data/endoscope_chesspts.p'))[0]))

	world = transform.generate_world()

	TE_2 = transform.get_transform("Endoscope", "PSM2", endoscope_calibration_matrix, psm2_calibration_matrix)
	psme_2 = transform.transform_data("Endoscope", "PSM2", endoscope_calibration_matrix, TE_2, psm2_calibration_matrix)
	pprint.pprint(psme_2)

	""" Move to chessboard corner, descend, come up,and go to next. """
	move_to(psm2, psme_2.tolist(), z_upper)

	home(psm2, pos, rot)
