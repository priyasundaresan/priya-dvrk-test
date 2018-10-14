import pickle
import pprint
import numpy as np
import rigid_transform
import read_chessboard

def load_all(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                return
""" Usage:
psm2_data = list(load_all('psm2_recordings.txt'))
print_psm_cache(psm2_data, 'PSM2 DATA')
// Prints out the images of PSM2 and corresponding poses.
"""
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

def transform(inpt, T):
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

def get_transform(inpt, outpt, data_in, data_out):
    print("\n{0} --> {1} Transformation Matrix".format(inpt, outpt))
    T = rigid_transform.solve_for_rigid_transformation(data_in, data_out)
    print(T)
    return T

def transform_data(inpt, outpt, data_in, T, data_out=None):
    print("\nTransforming {0} --> {1} Transform Matrix".format(inpt, outpt))
    expected = transform(data_in, T)
    print(expected)
    if data_out:
        print("Actual {0} Data".format(outpt))
        print(data_out)
        print("Associated Error: " + str(error(expected, data_out)))
    return expected

if __name__ == '__main__':
    w = []
    for i in range(5):
        for j in range(5):
            w.append([float(j)/80, float(i)/80, 0.])
    world = np.matrix(w)

    psm2_data = list(load_all('psm2_recordings.txt'))
    psm2_matrix = psm_data_to_matrix(psm2_data)

    endoscope_matrix = np.matrix(list(read_chessboard.load_all('camera_data/endoscope_chesspts.p'))[0])

    T2_E = get_transform("PSM2", "Endoscope", psm2_matrix, endoscope_matrix)
    TE_2 = get_transform("Endoscope", "PSM2", endoscope_matrix, psm2_matrix)

    psm2_e = transform_data("PSM2", "Endoscope", psm2_matrix, T2_E, endoscope_matrix)
    psme_2 = transform_data("Endoscope", "PSM2", endoscope_matrix, TE_2, psm2_matrix)

    T2_W = get_transform("PSM2", "World", psm2_matrix, world)
    psm2_w = transform_data("PSM2", "World", psm2_matrix, T2_W, world)

    TE_W = get_transform("Endoscope", "World", endoscope_matrix, world)
    e_w = transform_data("Endoscope", "World", endoscope_matrix, TE_W, world)
