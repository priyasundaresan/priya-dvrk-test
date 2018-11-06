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
import transform
import read_camera

USE_SAVED_IMAGES = False
USE_SPLIT_VIEW = True

def get_stereo_transform():
    endoscope_chesspts = list(read_camera.load_all('calibration/endoscope_chesspts.p'))
    camera_info = list(read_camera.load_all('camera_data/camera_info.p'))
    left_chesspts = np.matrix(list(read_camera.load_all('camera_data/left_chesspts'))[0])
    right_chesspts = np.matrix(list(read_camera.load_all('camera_data/right_chesspts'))[0])

    z = np.zeros((25, 1))
    left_chesspts = np.hstack((left_chesspts, z))
    right_chesspts = np.hstack((right_chesspts, z))

    TL_R = transform.get_transform("Left Camera", "Right Camera", left_chesspts, right_chesspts, verbose=False)
    return TL_R

class EmbeddedNeedleDetector():

    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.left_image = None
        self.right_image = None
        self.info = {'l': None, 'r': None, 'b': None, 'd': None}
        self.plane = None
        self.area_lower = 1800
        self.area_upper = 20000
        self.ellipse_lower = 1300
        self.ellipse_upper = 180000
        self.residual_lower = 700
        self.residual_upper = 2000
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
            self.right_image = cv2.imread('embedded_images/left4.jpg')
        else:
            self.right_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
    def left_image_callback(self, msg):
        if rospy.is_shutdown():
            return
        if USE_SAVED_IMAGES:
            self.left_image = cv2.imread('left_checkerboard.jpg')
        else:
            self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        if self.right_image is not None:
            self.process_image(self.left_image)

    def compute_centroid(self, contour, moments=None):
        if not moments:
            moments = cv2.moments(contour)
        if int(moments["m00"]) == 0:
            return
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return (cx, cy)

    def distance(self, p1, p2):
        return cv2.pointPolygonTest(p1, p2, True)

    def distance_pt_to_contour(self, contour, x, y):
        """Computes the distance from a point to the center (not centroid) of contour"""
        cX, cy = self.compute_centroid(contour)
        center = self.center(contour, cx, cy)
        return abs(self.distance(center, (x, y)))

    def get_ellipse(self, c):
        ellipse = cv2.fitEllipse(c)
        (x,y), (ma,MA), angle = ellipse
        ellipse_aspect = ma/MA
        ellipse_area = (np.pi * ma * MA)/4
        return (ellipse, ellipse_aspect, ellipse_area)

    def center(self, contour, cx, cy):
        return min(contour, key=lambda point: abs(self.distance(point, (cx, cy))))

    def endpoint(self, contour, cx, cy):
        sorted_points = sorted([list(i.squeeze()) for i in contour])
        e1 = np.array(sorted_points[0]).reshape(1, 2)
        e2 = np.array(sorted_points[-1]).reshape(1, 2)
        pt = max([e1, e2], key=lambda e: abs(self.distance(e, (cx, cy))))
        return pt

    def find_residual(self, contours, CX, CY):
        return min(contours, key=lambda c: self.distance_pt_to_contour(c, (CX, CY)))

    def report(self, area, cx, cy, CX, CY, ellipse_area):
        print('Contour Detected')
        print('Contour Area:', area)
        print('Centroid', cx, cy)
        print('Closest Point', CX, CY)
        print('Ellipse Area:', ellipse_area)
        print('---')

    def preprocess(self, image):
    	image_in = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        corrected = np.uint8(cv2.pow(image_in/255.0, 1.4) * 255)
        gray = cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        return thresh


    # Working now
    def process_image(self, image):

        thresh = self.preprocess(image)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # residuals = []

        for c in contours:
            M = cv2.moments(c)
            area = cv2.contourArea(c)

            # if (self.residual_lower < area < self.residual_upper):
            #     residuals.append(c)

            if (self.area_lower < area < self.area_upper):
            	cx, cy = self.compute_centroid(c, M)
            	closest = np.vstack(self.center(c, cx, cy)).squeeze()
                CX, CY = closest[0], closest[1]
            	true_center = (CX, CY)

            	ellipse, ellipse_aspect, ellipse_area = self.get_ellipse(c)

                """Contour is the big protruding part of the needle"""
            	if self.ellipse_lower < ellipse_area < self.ellipse_upper:

                    endpoint = tuple(np.vstack(self.endpoint(c, cx, cy)).squeeze())
                    EX, EY = endpoint[0], endpoint[1]
                    dx, dy = CX - EX, CY - EY
                    OX, OY = CX + dx, CY + dy

                    # Need (OX, OY), (CX, CY) in the right frame
                    opp_array = np.array([OX, OY, 0])
                    center_array = np.array([CX, CY, 0])
                    left_data = np.matrix([opp_array, center_array])

                    right_data = transform.transform_data("Left Camera", "Right Camera", left_data, self.TL_R, data_out=None, verbose=False)
                    right_OX, right_OY = int(right_data[0, 0]), int(right_data[0, 1])
                    right_CX, right_CY = int(right_data[1, 0]), int(right_data[1, 1])
                    
                    cv2.circle(self.right_image, (right_OX, right_OY), 10, (0, 0, 0), -1)
                    cv2.circle(self.right_image, (right_CX, right_CY), 10, (0, 0, 0), -1)
                    cv2.line(self.right_image, (right_OX, right_OY), (right_CX, right_CY), (0, 0, 0), 10)
                    cv2.putText(image, "center", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.circle(image, true_center, 10, (0, 0, 0), -1)
                    cv2.circle(image, (cx, cy), 10, (255, 255, 255), -1)
                    # self.report(area, cx, cy, CX, CY, ellipse_area)
                    cv2.ellipse(image, ellipse, (0, 0, 255), 2)
                    cv2.drawContours(image, [c], 0, (0, 255, 255), 2)
                    
                    # cv2.circle(image, (EX, EY), 10, (0, 170, 0), -1)
                    cv2.circle(image, (OX, OY), 10, (0, 0, 0), -1)
                    # cv2.line(image, true_center, (EX, EY), (255, 0, 0), 10)
                    cv2.line(image, true_center, (OX, OY), (0, 0, 0), 10)
                # else:
                #     cv2.drawContours(image, [c], -1, (0, 0, 255), 2)
                #     cv2.ellipse(image, ellipse, (0, 0, 255), 2)
                #     cv2.putText(image, "REJECTED", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

if __name__ == "__main__":
    a = EmbeddedNeedleDetector()
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
