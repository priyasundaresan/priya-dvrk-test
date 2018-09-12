import numpy as np

"""
This script contains utilities that are used to find the rigid transformation between coordinate frames.
"""

class Transformer(object):

    def __init__(self, psm1_transform="PSM2_to_PSM1.npy", psm2_transform="PSM1_to_PSM2.npy"):
        self._PSM1_to_PSM2 = np.load(psm2_transform)
        self._PSM2_to_PSM1 = np.load(psm1_transform)

    def to_PSM2(self, pt):
        return transform_point(pt, self._PSM1_to_PSM2)

    def to_PSM1(self, pt):
        return transform_point(pt, self._PSM2_to_PSM1)

def transform_point(pt, transform):
    npt = np.ones(4)
    npt[:3] = pt
    return np.dot(transform, npt)

def solve_for_rigid_transformation(inpts, outpts):
    """
    Takes in two sets of corresponding points, returns the rigid transformation matrix from the first to the second.
    """
    assert inpts.shape == outpts.shape
    inpts, outpts = np.copy(inpts), np.copy(outpts)
    inpt_mean = inpts.mean(axis=0)
    print(inpt_mean)
    outpt_mean = outpts.mean(axis=0)
    outpts -= outpt_mean
    inpts -= inpt_mean
    X = inpts.T
    Y = outpts.T
    covariance = np.dot(X, Y.T)
    U, s, V = np.linalg.svd(covariance)
    S = np.diag(s)
    assert np.allclose(covariance, np.dot(U, np.dot(S, V)))
    V = V.T
    idmatrix = np.identity(3)
    idmatrix[2, 2] = np.linalg.det(np.dot(V, U.T))
    R = np.dot(np.dot(V, idmatrix), U.T)
    t = outpt_mean.T - np.dot(R, inpt_mean)
    T = np.zeros((3, 4))
    T[:3,:3] = R
    T[:,3] = t
    return T

def least_squares_plane_normal(points_3d):
    x_list = points_3d[:,0]
    y_list = points_3d[:,1]
    z_list = points_3d[:,2]

    A = np.concatenate((x_list, y_list, np.ones((len(x_list), 1))), axis=1)
    plane = np.matrix(np.linalg.lstsq(A, z_list)[0]).T

    return plane

def distance_to_plane(m, point):
    A = m[0,0]
    B = m[0,1]
    C = -1
    D = m[0,2]
    p0 = np.array([0,0,D])
    p1 = np.array(point)
    n = np.array([A,B,C])/np.linalg.norm(np.array([A,B,C]))
    return np.dot(np.absolute(p0 - p1),n)

def get_good_indices(thresh=0.022):
    camera_points = load_camera_points()
    plane = least_squares_plane_normal(camera_points)
    good_pts = []
    for i in range(camera_points.shape[0]):
        p = camera_points[i,:]
        dist = distance_to_plane(plane, p)
        if abs(dist) > thresh:
            continue
        else:
            good_pts.append(i)
    return good_pts

if __name__ == '__main__':

    psm1_pts = np.load("correspondences/psm1_board_1.npy")[:,:3]
    psm2_pts = np.load("correspondences/psm2_board_1.npy")[:,:3]
    transform1 = solve_for_rigid_transformation(psm1_pts, psm2_pts)
    print(transform1)
    np.save("PSM1_to_PSM2.npy", transform1)

    psm1_pts = np.load("correspondences/psm1_board_1.npy")[:,:3]
    psm2_pts = np.load("correspondences/psm2_board_1.npy")[:,:3]
    transform2 = solve_for_rigid_transformation(psm2_pts, psm1_pts)
    print(transform2)
    np.save("PSM2_to_PSM1.npy", transform2)

    e1 =  np.mean([np.linalg.norm(p2 - transform_point(p1, transform1)) for p1, p2 in zip(psm1_pts, psm2_pts)])
    e2 =  np.mean([np.linalg.norm(p1 - transform_point(p2, transform2)) for p1, p2 in zip(psm1_pts, psm2_pts)])
    print(e1, e2)
