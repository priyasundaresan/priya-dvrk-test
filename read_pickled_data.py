import pickle
import pprint
import numpy as np
import rigid_transform_test

def loadall(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                return

def printdata(lst, heading):
    print(heading)
    print('---')
    pprint.pprint(lst)
    print('\n')

def data_to_matrix(cache):
    i = 0
    matrix = np.matrix(np.zeros(shape=(len(cache), 3)))
    for pair in cache:
        params = pair.values()[0]
        a = np.array([params['x'], params['y'], params['z']])
        matrix[i] = a
        i += 1
    return matrix


if __name__ == '__main__':
    psm1_data = list(loadall('psm1_recordings.txt'))
    psm2_data = list(loadall('psm2_recordings.txt'))

    printdata(psm1_data, 'PSM1 DATA')
    printdata(psm2_data, 'PSM2 DATA')

    psm1_matrix = data_to_matrix(psm1_data)
    psm2_matrix = data_to_matrix(psm2_data)

    pprint.pprint(data_to_matrix(psm1_data))
    pprint.pprint(data_to_matrix(psm2_data))

    print("PSM1 to PSM2 Transform")
    T1_2 = rigid_transform_test.solve_for_rigid_transformation(psm1_matrix, psm2_matrix)
    print(T1_2)

    print("PSM2 to PSM1 Transform")
    T2_1 = rigid_transform_test.solve_for_rigid_transformation(psm1_matrix, psm2_matrix)
    print(T2_1)

