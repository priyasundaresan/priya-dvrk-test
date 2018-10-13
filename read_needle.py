import pickle
import pprint
import numpy as np

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
	needle_points = list(load_all('needle_data/needle_points.p'))
	print_cache(needle_points, "NEEDLE POINTS")
