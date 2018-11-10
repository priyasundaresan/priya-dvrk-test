import pickle
import pprint
import numpy as np
import rigid_transform
import transform

def load_all(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                return

def print_cache(lst, heading):
    print(heading)
    print('---')
    pprint.pprint(lst)


if __name__ == '__main__':

	endoscope_chesspts = list(load_all('calibration/endoscope_chesspts.p'))
	camera_info = list(load_all('camera_data/camera_info.p'))
	left_chesspts = np.matrix(list(load_all('camera_data/left_chesspts'))[0])
	right_chesspts = np.matrix(list(load_all('camera_data/right_chesspts'))[0])

	z = np.zeros((25, 1))
	left_chesspts = np.hstack((left_chesspts, z))
	right_chesspts = np.hstack((right_chesspts, z))
	
	print_cache(endoscope_chesspts, "ENDOSCOPE CHESSPOINTS")
	# print_cache(camera_info, "CAMERA INFO")
	print_cache(left_chesspts, "LEFT CHESSPOINTS")
	print_cache(right_chesspts, "RIGHT CHESSPOINTS")

	TL_R = transform.get_transform("Left Camera", "Right Camera", left_chesspts, right_chesspts)
	L_R = transform.transform_data("Left Camera", "Right Camera", left_chesspts, TL_R, right_chesspts)
