import pickle
import pprint
import numpy as np
import rigid_transform

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
	endoscope_chesspts = list(load_all('camera_data/endoscope_chesspts.p'))
	# camera_info = list(load_all('camera_data/camera_info.p'))
	# left_chesspts = list(load_all('camera_data/left_chesspts'))
	# right_chesspts = list(load_all('camera_data/right_chesspts'))
	print_cache(endoscope_chesspts, "ENDOSCOPE CHESSPOINTS")
	# print_cache(camera_info, "CAMERA INFO")
	# print(type(left_chesspts))
	# print_cache(left_chesspts, "LEFT CHESSPOINTS")
	# print_cache(right_chesspts, "RIGHT CHESSPOINTS")
