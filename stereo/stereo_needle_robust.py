import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import image_geometry
import rospy
from geometry_msgs.msg import PointStamped, Point
import cv2
import cv_bridge
import numpy as np
import rospy, scipy.misc
from sensor_msgs.msg import Image, CameraInfo
import time
import pprint
import pickle
import read_camera
import transform

USE_SAVED_IMAGES = False
USE_SPLIT_VIEW = False

def get_stereo_transform():
    endoscope_chesspts = list(read_camera.load_all('../calibration/endoscope_chesspts.p'))
    camera_info = list(read_camera.load_all('../camera_data/camera_info.p'))
    left_chesspts = np.matrix(list(read_camera.load_all('../camera_data/left_chesspts'))[0])
    right_chesspts = np.matrix(list(read_camera.load_all('../camera_data/right_chesspts'))[0])

    z = np.zeros((25, 1))
    left_chesspts = np.hstack((left_chesspts, z))
    right_chesspts = np.hstack((right_chesspts, z))

    TL_R = transform.get_transform("Left Camera", "Right Camera", left_chesspts, right_chesspts, verbose=False)
    return TL_R

def convertStereo(u, v, disparity, info=None):
    """
    Converts two pixel coordinates u and v along with the disparity to give PointStamped
    """
    stereoModel = image_geometry.StereoCameraModel()
    if info is None:
        with open("camera_data/camera_info.p", "rb") as f:
            info = pickle.load(f)
    stereoModel.fromCameraInfo(info['l'], info['r'])
    (x,y,z) = stereoModel.projectPixelTo3d((u,v), disparity)

    cameraPoint = PointStamped()
    cameraPoint.header.frame_id = info['l'].header.frame_id
    cameraPoint.header.stamp = rospy.Time.now()
    cameraPoint.point = Point(x,y,z)
    return cameraPoint

def projectToPixel(pt, info=None):
    x, y, z = pt
    if info is None:
        with open("camera_data/camera_info.p", "rb") as f:
            info = pickle.load(f)
    stereoModel = image_geometry.StereoCameraModel()
    stereoModel.fromCameraInfo(info['l'], info['r'])
    left, right = stereoModel.project3dToPixel((x, y, z))
    return (left, right)

class EllipseDetector:

    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.left_image = None
        self.right_image = None
        self.corrected_left = None
        self.corrected_right = None
        self.info = {'l': None, 'r': None, 'b': None, 'd': None}
        self.plane = None
        self.area_lower = 300
        self.area_upper = 30000
        self.ellipse_area_lower = 10000
        self.ellipse_area_upper = 200000
        self.TL_R = get_stereo_transform()

        #========SUBSCRIBERS========#
        # image subscribers
        rospy.init_node('circle_detector', anonymous=True)
        rospy.Subscriber("/endoscope/left/image_rect_color", Image,
                         self.left_image_callback, queue_size=1)
        rospy.Subscriber("/endoscope/right/image_rect_color", Image,
                         self.right_image_callback, queue_size=1)
        # info subscribers
        rospy.Subscriber("/endoscope/left/camera_info",
                         CameraInfo, self.left_info_callback)
        rospy.Subscriber("/endoscope/right/camera_info",
                         CameraInfo, self.right_info_callback)

    def left_info_callback(self, msg):
        if self.info['l']:
            return
        self.info['l'] = msg

    def right_info_callback(self, msg):
        if self.info['r']:
            return
        self.info['r'] = msg

    def right_image_callback(self, msg):
        if rospy.is_shutdown():
            return
        if USE_SAVED_IMAGES:
            self.right_image = cv2.imread('right_checkerboard.jpg')
        else:
            self.right_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def left_image_callback(self, msg):
        if rospy.is_shutdown():
            return
        if USE_SAVED_IMAGES:
            self.left_image = cv2.imread('left_checkerboard.jpg')
        else:
            self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        if self.right_image != None:
            self.process_image(self.left_image)

    def get_points_3d(self, left_points, right_points):
        """ this method assumes that corresponding points are in the right order
            and returns a list of 3d points """

        # both lists must be of the same lenghth otherwise return None
        left_points, right_points = sorted(left_points), sorted(right_points)
        print("\nLeft/Right Points Found:")
        print(left_points, right_points)
        if len(left_points) != len(right_points):
            rospy.logerror("The number of left points and the number of right points is not the same")
            return None
        points_3d = []
        for i in range(len(left_points)):
            a = left_points[i]
            b = right_points[i]
            disparity = abs(a[0]-b[0])
            pt = convertStereo(a[0], a[1], disparity, self.info)
            points_3d.append(pt)
        # print(points_3d)
        return points_3d

    def closest_to_centroid(self, contour_points, cX, cY):
        return min(contour_points, key=lambda c: abs(cv2.pointPolygonTest(c,(cX,cY),True)))

    def report(self, contour, area, cX, cY, closest, ellipse_area, angle):
        print('Contour Detected')
        print('Contour Area:', area)
        print('Centroid', cX, cY)
        print('Closest Point', closest[0], closest[1])
        print('Ellipse Area:', ellipse_area)
        print('Angle:', angle)
        print('---')

    def preprocess(self, image):
        image_in = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, s, v = cv2.split(cv2.cvtColor(image_in, cv2.COLOR_RGB2HSV))
        nonSat = s < 180
        disk = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        nonSat = cv2.erode(nonSat.astype(np.uint8), disk)
        v2 = v.copy()
        v2[nonSat == 0] = 0
        glare = v2 > 240;
        glare = cv2.dilate(glare.astype(np.uint8), disk);
        corrected = cv2.inpaint(image_in, glare, 5, cv2.INPAINT_NS)
        gray = cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        return thresh


    # Working now
    def process_image(self, image):
        left, right = [], []
        im2, contours, hierarchy = cv2.findContours(self.preprocess(image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            M = cv2.moments(c)
            area = cv2.contourArea(c)
            if int(M["m00"]) != 0 and (self.area_lower < area < self.area_upper):
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                closest = np.vstack(self.closest_to_centroid(c, cX, cY)).squeeze()
                CX, CY = closest[0], closest[1]
                ellipse = cv2.fitEllipse(c)
                (x,y), (ma,MA), angle = ellipse
                aspect_ratio = ma/MA
                ellipse_area = (np.pi * ma * MA)/4
                left_center = (closest[0], closest[1])

                if (0.75 < aspect_ratio < 1.0) and self.ellipse_area_lower < ellipse_area:

                    left_data = np.matrix([[CX, CY, 0]])
                    right_data = transform.transform_data("Left Camera", "Right Camera", left_data, self.TL_R, data_out=None, verbose=False)
                    right_center = (int(right_data[0,0]), int(right_data[0,1]))

                    left.append(left_center)
                    right.append(right_center)

                    self.report(c, area, cX, cY, closest, ellipse_area, angle)
                    cv2.drawContours(self.left_image, [c], -1, (0, 255, 0), 2)
                    cv2.ellipse(self.left_image, ellipse, (255, 0, 0), 2)
                    cv2.circle(self.left_image, (cX, cY), 7, (255, 255, 255), -1)
                    cv2.putText(self.left_image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.circle(self.left_image, left_center, 10, (0, 0, 0), -1)
                    cv2.circle(self.right_image, right_center, 10, (0, 0, 0), -1)

                    
                # else:
                #     cv2.drawContours(image, [c], -1, (0, 0, 255), 2)
                #     cv2.ellipse(image, ellipse, (0, 0, 255), 2)
                #     cv2.putText(image, "REJECTED", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if len(right) > 0 and len(right) == len(left):
            pts3d = self.get_points_3d(left, right)
            print("Found")
            self.pts = [(p.point.x, p.point.y, p.point.z) for p in pts3d]
            pprint.pprint(self.pts)
            with open('needle_data/needle_points.p', "w+") as f:
                pickle.dump(self.pts, f)
            rospy.signal_shutdown("Finished.")


if __name__ == "__main__":
    a = EllipseDetector()
    while 1:
        if USE_SPLIT_VIEW:
            if a.left_image is None or a.right_image is None:
                continue
            left = cv2.resize(a.left_image, (0, 0), fx=0.5, fy=0.5)
            right = cv2.resize(a.right_image, (0, 0), fx=0.5, fy=0.5)
            frame = np.hstack((left, right))
        else:
            frame = a.left_image
            if frame is None:
                continue
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
