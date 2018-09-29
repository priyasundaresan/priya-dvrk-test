import robot
import pprint
import numpy as np
import PyKDL
import read_psm_data
import read_chessboard_data
import rigid_transform_test
import time
import math 




if __name__ == '__main__':
	psm2 = robot.robot('PSM2')
	kdl_pose = psm2.get_current_position()
	pprint.pprint(kdl_pose)

	psm2_calibration_data = list(read_psm_data.load_all('calibration/psm2_recordings.txt'))
	psm2_calibration_matrix = read_psm_data.psm_data_to_matrix(psm2_calibration_data)
	endoscope_calibration_matrix = np.matrix(list(read_chessboard_data.load_all('calibration/endoscope_chesspts.p'))[0])

	endoscope_points = np.matrix(list(read_chessboard_data.load_all('camera_data/endoscope_chesspts.p'))[0])

	TE_2 = rigid_transform_test.solve_for_rigid_transformation(endoscope_calibration_matrix, psm2_calibration_matrix)
	print('\nTransforming Endoscope --> PSM2')
	print('            x           y           z')
	psme_2 = read_psm_data.transform_matrix(endoscope_points, TE_2)
	print(psme_2)
	print('Actual PSM2 Data')
	print('            x           y           z')
	print(psm2_calibration_matrix)
	print('Associated Error:', read_psm_data.error(psme_2, psm2_calibration_matrix))

	# Test moving the PSM to unseen points (the first row of the chessboard, in this case)
	for point in psme_2.tolist()[:10]:
		print(point)
		psm2.move(PyKDL.Vector(point[0], point[1], point[2]))
		time.sleep(.5)

