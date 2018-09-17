import pickle
import pprint
import numpy as np
import rigid_transform_test

def load_all(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                return

def print_psm_cache(lst, heading):
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

def transform_matrix(inpt, T):
    outpt = np.matrix(np.zeros(shape=inpt.shape))
    for i in range(np.size(inpt, 0)):
        outpt[i] = rigid_transform_test.transform_point(np.array(inpt[i]), T)
    return outpt

def error(m1, m2):
    assert m1.shape == m2.shape
    errors = []
    for i in range(np.size(m1, 0)):
        errors.append(np.linalg.norm(m1[i] - m2[i]))
    return np.mean(errors)


if __name__ == '__main__':

    psm1_data = list(load_all('psm1_recordings.txt'))
    psm2_data = list(load_all('psm2_recordings.txt'))

    # print_psm_cache(psm1_data, 'PSM1 DATA')
    # print_psm_cache(psm2_data, 'PSM2 DATA')

    psm1_matrix = data_to_matrix(psm1_data)
    psm2_matrix = data_to_matrix(psm2_data)

    print("\nPSM1 --> PSM2 Transform Matrix")
    T1_2 = rigid_transform_test.solve_for_rigid_transformation(psm1_matrix, psm2_matrix)
    print(T1_2)

    print("\nPSM2 --> PSM1 Transform Matrix")
    T2_1 = rigid_transform_test.solve_for_rigid_transformation(psm2_matrix, psm1_matrix)
    print(T2_1)

    print('\nTransforming PSM1 --> PSM2')
    print('            x           y           z')
    psm1_2 = transform_matrix(psm1_matrix, T1_2)
    print(psm1_2)
    print('Actual PSM2 Data')
    print('            x           y           z')
    print(data_to_matrix(psm2_data))
    print('Associated Error:', error(psm1_2, psm2_matrix))

    print('\nTransforming PSM2 --> PSM1')
    print('            x           y           z')
    psm2_1 = transform_matrix(psm2_matrix, T2_1)
    print(psm2_1)
    print('Actual PSM1 Data')
    print('            x           y           z')
    print(data_to_matrix(psm1_data))
    print('Associated Error:', error(psm2_1, psm1_matrix))

    