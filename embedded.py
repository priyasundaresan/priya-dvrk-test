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

USE_SAVED_IMAGES = False
USE_SPLIT_VIEW = True

class EmbeddedNeedleDetector():

    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.left_image = None
        self.right_image = None
        self.info = {'l': None, 'r': None, 'b': None, 'd': None}
        self.plane = None
        self.area_lower = 1500
        self.area_upper = 20000
        self.ellipse_lower = 1300
        self.ellipse_upper = 180000
        self.residual_lower = 700
        self.residual_upper = 2000

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
        if self.left_image is not None:
            self.process_image(self.right_image)

    def left_image_callback(self, msg):
        if rospy.is_shutdown():
            return
        if USE_SAVED_IMAGES:
            self.left_image = cv2.imread('left_checkerboard.jpg')
        else:
            self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        if self.right_image is not None:
            self.process_image(self.left_image)

    def compute_centroid(self, c, moments=None):
        if not moments:
            moments = cv2.moments(c)
        if int(moments["m00"]) == 0:
            return
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        return (cX, cY)

    def distance(self, p1, p2):
        return cv2.pointPolygonTest(p1, p2, True)

    def distance_pt_to_contour(self, c, p):
        """Computes the distance from a point to the center (not centroid) of contour"""
        cX, cY = self.compute_centroid(c)
        center = self.center(c, cX, cY)
        return abs(self.distance(center, p))

    def get_ellipse(self, c):
        ellipse = cv2.fitEllipse(c)
        (x,y), (ma,MA), angle = ellipse
        ellipse_aspect = ma/MA
        ellipse_area = (np.pi * ma * MA)/4
        return (ellipse, ellipse_aspect, ellipse_area)

    def center(self, contour_points, cX, cY):
        return min(contour_points, key=lambda point: abs(self.distance(point, (cX, cY))))

    def endpoint(self, contour_points, cX, cY):
        return min(contour_points, key=lambda point: self.distance(point, (cX, cY)))

    def find_residual(self, contours, Cx, Cy):
        return min(contours, key=lambda c: self.distance_pt_to_contour(c, (Cx, Cy)))

    def report(self, area, cX, cY, closest, ellipse_area):
        print('Contour Detected')
        print('Contour Area:', area)
        print('Centroid', cX, cY)
        print('Closest Point', closest[0], closest[1])
        print('Ellipse Area:', ellipse_area)
        print('---')

    def preprocess(self, image):
    	image_in = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        corrected = np.uint8(cv2.pow(image_in/255.0, 1.4) * 255)
        # h, s, v = cv2.split(cv2.cvtColor(image_in, cv2.COLOR_RGB2HSV))
        # nonSat = s < 180
        # disk = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        # nonSat = cv2.erode(nonSat.astype(np.uint8), disk)
        # v2 = v.copy()
        # v2[nonSat == 0] = 0
        # glare = v2 > 240;
        # glare = cv2.dilate(glare.astype(np.uint8), disk);
        # corrected = cv2.inpaint(image_in, glare, 5, cv2.INPAINT_NS)
        gray = cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        return thresh


    # Working now
    def process_image(self, image):

        thresh = self.preprocess(image)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        residuals = []

        for c in contours:
            M = cv2.moments(c)
            area = cv2.contourArea(c)

            if (self.residual_lower < area < self.residual_upper):
                residuals.append(c)

            if (self.area_lower < area < self.area_upper):
            	cX, cY = self.compute_centroid(c, M)
            	closest = np.vstack(self.center(c, cX, cY)).squeeze()
                Cx, Cy = closest[0], closest[1]
            	true_center = (Cx, Cy)

            	ellipse, ellipse_aspect, ellipse_area = self.get_ellipse(c)

                """Contour is the big protruding part of the needle"""
            	if self.ellipse_lower < ellipse_area < self.ellipse_upper:
                    cv2.putText(image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.circle(image, true_center, 10, (0, 0, 0), -1)
                    # self.report(area, cX, cY, closest, ellipse_area)
                    cv2.ellipse(image, ellipse, (0, 0, 255), 2)
                    cv2.drawContours(image, [c], 0, (0, 255, 255), 2)
                    e = np.vstack(self.endpoint(c, cX, cY)).squeeze()
                    eX, eY = e[0], e[1]
                    cv2.circle(image, (eX, eY), 10, (0, 170, 0), -1)
                    cv2.line(image, true_center, (eX, eY), (255, 0, 0), 10)
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
