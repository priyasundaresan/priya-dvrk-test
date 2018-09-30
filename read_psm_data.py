import pickle
import pprint
import numpy as np
import rigid_transform_test
import read_chessboard_data

"""NOTE: PSM2 RECORDED IN THIS POSE:
[[   -0.988883, -0.00205771,   -0.148682;
  -0.00509171,    0.999786,   0.0200282;
     0.148609,   0.0205626,   -0.988682]
[   -0.128112,    0.021001,   -0.124348]]
"""

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

def psm_data_to_matrix(cache):
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

    # psm1_data = list(load_all('calibration/psm1_recordings.txt'))
    psm2_data = list(load_all('calibration/psm2_recordings.txt'))

    # print_psm_cache(psm1_data, 'PSM1 DATA')
    # print_psm_cache(psm2_data, 'PSM2 DATA')

    # psm1_matrix = psm_data_to_matrix(psm1_data)
    psm2_matrix = psm_data_to_matrix(psm2_data)
    endoscope_matrix = np.matrix(list(read_chessboard_data.load_all('calibration/endoscope_chesspts.p'))[0])


    # print("\nPSM1 --> PSM2 Transform Matrix")
    # T1_2 = rigid_transform_test.solve_for_rigid_transformation(psm1_matrix, psm2_matrix)
    # print(T1_2)

    # print("\nPSM2 --> PSM1 Transform Matrix")
    # T2_1 = rigid_transform_test.solve_for_rigid_transformation(psm2_matrix, psm1_matrix)
    # print(T2_1)

    # print("\nPSM1 --> Endoscope Transform Matrix")
    # T1_E = rigid_transform_test.solve_for_rigid_transformation(psm1_matrix, endoscope_matrix)
    # print(T1_E)

    print("\nPSM2 --> Endoscope Transform Matrix")
    T2_E = rigid_transform_test.solve_for_rigid_transformation(psm2_matrix, endoscope_matrix)
    print(T2_E)

    print("\nEndoscope --> PSM2 Transform Matrix")
    TE_2 = rigid_transform_test.solve_for_rigid_transformation(endoscope_matrix, psm2_matrix)
    print(TE_2)

    # print('\nTransforming PSM1 --> PSM2')
    # print('            x           y           z')
    # psm1_2 = transform_matrix(psm1_matrix, T1_2)
    # print(psm1_2)
    # print('Actual PSM2 Data')
    # print('            x           y           z')
    # print(psm2_matrix)
    # print('Associated Error:', error(psm1_2, psm2_matrix))

    # print('\nTransforming PSM2 --> PSM1')
    # print('            x           y           z')
    # psm2_1 = transform_matrix(psm2_matrix, T2_1)
    # print(psm2_1)
    # print('Actual PSM1 Data')
    # print('            x           y           z')
    # print(psm1_matrix)
    # print('Associated Error:', error(psm2_1, psm1_matrix))

    # print('\nTransforming PSM1 --> Endoscope')
    # print('            x           y           z')
    # psm1_e = transform_matrix(psm1_matrix, T1_E)
    # print(psm1_e)
    # print('Actual Endoscope Data')
    # print('            x           y           z')
    # print(endoscope_matrix)
    # print('Associated Error:', error(psm1_e, endoscope_matrix))

    print('\nTransforming PSM2 --> Endoscope')
    print('            x           y           z')
    psm2_e = transform_matrix(psm2_matrix, T2_E)
    print(psm2_e)
    print('Actual Endoscope Data')
    print('            x           y           z')
    print(endoscope_matrix)
    print('Associated Error:', error(psm2_e, endoscope_matrix))

    print('\nTransforming Endoscope --> PSM2')
    print('            x           y           z')
    psme_2 = transform_matrix(endoscope_matrix, TE_2)
    print(psme_2)
    print('Actual PSM2 Data')
    print('            x           y           z')
    print(psm2_matrix)
    print('Associated Error:', error(psme_2, psm2_matrix))

    
