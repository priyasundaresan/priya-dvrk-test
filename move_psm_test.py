import robot
import pprint
import numpy as np
import PyKDL

if __name__ == '__main__':

    psm2 = robot.robot('PSM2')

    kdl_pose = psm2.get_current_position()
    pprint.pprint(kdl_pose)
    print(type(kdl_pose))