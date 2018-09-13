import Tkinter
import numpy as np
import dvrk

"""
Launching this script creates a GUI that subscribes to PSM1's position_cartesian_current topic and can write this information to file.
"""

def PSM1_callback():
    kdl_pose = psm1.get_current_position()
    x, y, z = kdl_pose.p.x(), kdl_pose.p.y(), kdl_pose.p.z()
    yaw, pitch, roll = np.array(kdl_pose.M.GetEulerZYX())
    psm1_pts.append((x, y, z, yaw, pitch, roll))
    print(np.array(psm1_pts).shape)
    np.save("registration/psm1_pts.npy", np.array(psm1_pts))


def PSM2_callback():
    kdl_pose = psm2.get_current_position()
    x, y, z = kdl_pose.p.x(), kdl_pose.p.y(), kdl_pose.p.z()
    yaw, pitch, roll = np.array(kdl_pose.M.GetEulerZYX())
    psm2_pts.append((x, y, z, yaw, pitch, roll))
    print(np.array(psm2_pts).shape)
    np.save("registration/psm2_pts.npy", np.array(psm2_pts))


def plot_points():
    """
    Plots points in robot_frame. Axes may need to be edited.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    pts=load_robot_points()
    if pts.shape[1] == 0:
        print "no points to show"
        return
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.array(pts[:,0]), np.array(pts[:,1]), np.array(pts[:,2]),c='r')
    ax.set_xlim3d(0, 0.2)
    ax.set_ylim3d(-0.1, 0.1)
    ax.set_zlim3d(-0.15,0.05)
    plt.show()

if __name__ == '__main__':
    psm1 = dvrk.psm('PSM1')
    psm2 = dvrk.psm('PSM2')

    psm1_pts = []
    psm2_pts = []

    top = Tkinter.Tk()
    top.title('Calibration')
    top.geometry('400x200')
    Tkinter.Button(top, text="Record Position PSM1", command = PSM1_callback).pack()
    Tkinter.Button(top, text="Record Position PSM2", command = PSM2_callback).pack()
    top.mainloop()