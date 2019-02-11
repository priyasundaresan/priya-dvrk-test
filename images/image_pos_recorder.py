""" A script for recording images of DVRK PSM1 and PSM2 and the PSM's
corresponding position parameters to memory

=====USAGE=====
Once the PSM's position/image is ready to be recorded, use the
endoscope's R/L camera to take pictures of PSM1/2 respectively.
In the GUI, press the corresponding 'Record PSM1/2' to record
the latest image and the current position simultaneously.

Images are recorded in the format 'rightx.jpg' in memory in the current directory,
where x=1 for the 1st picture taken for the endoscope's right camera, x=2 for
the 2nd, etc.

The file 'psm1_recordings.txt' will read:
'''
{'right1.jpg': [position vector of psm1 in image 1]}
{'right2.jpg': [position vector of psm1 in image 2]}
{'right3.jpg': [position vector of psm1 in image 3]}
...
'''
where the position vector is an np.array with parameters x, y, z,
yaw, pitch, roll.

A similar file will exist for PSM2 with data from the left
camera of the endoscope and the PSM's corresponding positions in those images.
"""
from Tkinter import Tk, Label, Button
import dvrk
import numpy as np
import pickle
import rospy
import robot
import image_subscriber
import pprint

def PSM1_callback():
    kdl_pose = psm1.get_current_position()
    x, y, z = kdl_pose.p.x(), kdl_pose.p.y(), kdl_pose.p.z()
    yaw, pitch, roll = np.array(kdl_pose.M.GetEulerZYX())
    psm1_pts = [x, y, z, yaw, pitch, roll]
    img_sub.right_called = True
    psm1_img_id = 'psm_data/right' + str(img_sub.right_img_id) + '.jpg' # Gets the latest image (corresponding to current position)
    print_position(1, psm1_pts)
    export_position(psm1_file, psm1_pts, psm1_img_id)

def PSM2_callback():
    kdl_pose = psm2.get_current_position()
    x, y, z = kdl_pose.p.x(), kdl_pose.p.y(), kdl_pose.p.z()
    yaw, pitch, roll = np.array(kdl_pose.M.GetEulerZYX())
    psm2_pts = [x, y, z, yaw, pitch, roll]
    img_sub.left_called = True
    psm2_img_id = 'psm_data/left' + str(img_sub.left_img_id) + '.jpg' # Gets the latest image (corresponding to current position)
    print_position(2, psm2_pts)
    export_position(psm2_file, psm2_pts, psm2_img_id)

def print_position(psm, psm_pts):
    """ Prints formatted position parameters (x, y, z, yaw, pitch, roll) """
    print("PSM" + str(psm))
    pairs = zip(['x', 'y', 'z', 'yaw', 'pitch', 'roll'], psm_pts)
    pprint.pprint(list(pairs))
    print('---')

def export_position(file_name, position, img_id):
    """ Writes the PSM position and corresponding latest image to memory """
    cache = {img_id: dict(zip(['x', 'y', 'z', 'yaw', 'pitch', 'roll'], position))}
    pickle.dump(cache, open(file_name, 'ab'))

class GUI:
    def __init__(self, master):

        self.master = master
        master.title("DVRK Pose/Image Recorder")
        master.geometry('400x150')

        text = """Press below to record a PSM's current position
        and corresponding latest image to memory"""
        self.label = Label(master, text=text , relief='groove', wraplength=250)
        self.label.pack()

        self.psm1_button = Button(master, text="Record PSM1", command=PSM1_callback)
        self.psm1_button.pack()

        self.psm2_button = Button(master, text="Record PSM2", command=PSM2_callback)
        self.psm2_button.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

if __name__ == '__main__':

    psm1 = robot.robot('PSM1')
    psm2 = robot.robot('PSM2')

    img_sub = image_subscriber.ImageSubscriber(write=True)

    psm1_file = 'psm1_recordings.txt'
    psm2_file = 'psm2_recordings.txt'

    open(psm1_file, 'wb').close()
    open(psm2_file, 'wb').close()

    root = Tk()
    gui = GUI(root)
    root.mainloop()
