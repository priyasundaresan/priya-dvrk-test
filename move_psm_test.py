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



if __name__ == '__main__':

	psm2 = robot.robot('PSM2')
	kdl_pose = psm2.get_current_position()
	print("Current Pose:")
	pprint.pprint(kdl_pose)

	z_limit = -0.124247
	
	pos = PyKDL.Vector(-0.118749, 0.0203151, -0.121688)

	#ROTATION PSM2 WAS CALIBRATED IN
	rot = PyKDL.Rotation(-0.988883, -0.00205771,   -0.148682,
						-0.00509171,    0.999786,   0.0200282,
						 0.148609,   0.0205626,   -0.988682)

	# start = PyKDL.Frame(rot, pos)
	# psm2.move(start)
	# time.sleep(2)
	# psm2.close_jaw()
	

	z_offset = PyKDL.Vector(0, 0, 0.005)

	psm2_calibration_data = list(read_psm_data.load_all('calibration/psm2_recordings.txt'))
	psm2_calibration_matrix = read_psm_data.psm_data_to_matrix(psm2_calibration_data)
	endoscope_calibration_matrix = np.matrix(list(read_chessboard_data.load_all('calibration/endoscope_chesspts.p'))[0])

	needle_points = np.matrix(list(read_needle_points.load_all('needle_data/needle_points.p'))[0])


	TE_2 = rigid_transform_test.solve_for_rigid_transformation(endoscope_calibration_matrix, psm2_calibration_matrix)
	print('\nTransforming Needle Points Endoscope Frame --> PSM2 Frame')
	print('            x           y           z')
	needle_to_psm2 = read_psm_data.transform_matrix(needle_points, TE_2)
	print(needle_to_psm2)

	# Test moving the PSM to needle centers
	# for point in needle_to_psm2.tolist()[:1]:
	# 	print(point)
	# 	psm2.move(PyKDL.Vector(point[0], point[1], max(z_limit, point[2])))
	# 	time.sleep(.25)
	# 	psm2.dmove(z_offset)
	# 	time.sleep(.25)


