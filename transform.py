import pickle
import pprint
import numpy as np
import rigid_transform
import read_chessboard

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
        outpt[i] = rigid_transform.transform_point(np.array(inpt[i]), T)
    return outpt

def error(m1, m2):
    assert m1.shape == m2.shape
    errors = []
    for i in range(np.size(m1, 0)):
        errors.append(np.linalg.norm(m1[i] - m2[i]))
    return np.mean(errors)

if __name__ == '__main__':

    w = []
    for i in range(5):
        for j in range(5):
            w.append([float(j)/80, float(i)/80, 0.])
    world = np.matrix(w)

    psm2_data = list(load_all('psm2_recordings.txt'))

    # print_psm_cache(psm1_data, 'PSM1 DATA')
    # print_psm_cache(psm2_data, 'PSM2 DATA')

    psm2_matrix = psm_data_to_matrix(psm2_data)
    endoscope_matrix = np.matrix(list(read_chessboard.load_all('camera_data/endoscope_chesspts.p'))[0])

    print("\nPSM2 --> Endoscope Transform Matrix")
    T2_E = rigid_transform.solve_for_rigid_transformation(psm2_matrix, endoscope_matrix)
    print(T2_E)

    print("\nEndoscope --> PSM2 Transform Matrix")
    TE_2 = rigid_transform.solve_for_rigid_transformation(endoscope_matrix, psm2_matrix)
    print(TE_2)

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

    print("\nPSM2 --> World Transform Matrix")
    T2_W = rigid_transform.solve_for_rigid_transformation(psm2_matrix, world)
    print(T2_W)

    print('\nTransforming PSM2 --> World')
    print('            x           y           z')
    psm2_w = transform_matrix(psm2_matrix, T2_W)
    print(psm2_w)
    print('Actual World Data')
    print('            x           y           z')
    print(world)
    print('Associated Error:', error(psm2_w, world))

    print("\n Endoscope --> World Transform Matrix")
    TE_W = rigid_transform.solve_for_rigid_transformation(endoscope_matrix, world)
    print(TE_W)

    print('\nTransforming Endoscope --> World')
    print('            x           y           z')
    e_w = transform_matrix(endoscope_matrix, TE_W)
    print(e_w)
    print('Actual World Data')
    print('            x           y           z')
    print(world)
    print('Associated Error:', error(e_w, world))


    
