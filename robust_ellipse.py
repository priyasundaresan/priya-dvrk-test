import image_geometry
import rospy
from geometry_msgs.msg import PointStamped, Point
import cv2
import cv_bridge
import numpy as np
import rospy, scipy.misc
from sensor_msgs.msg import Image, CameraInfo
import time

USE_SAVED_IMAGES = False
USE_SPLIT_VIEW = True

class EllipseDetector:

    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.left_image = None
        self.right_image = None
        self.info = {'l': None, 'r': None, 'b': None, 'd': None}
        self.plane = None
        self.area_lower = 300
        self.area_upper = 30000
        self.ellipse_area_lower = 5000
        self.ellipse_area_upper = 200000

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
        self.process_image(self.right_image)

    def left_image_callback(self, msg):
        if rospy.is_shutdown():
            return
        if USE_SAVED_IMAGES:
            self.left_image = cv2.imread('left_checkerboard.jpg')
        else:
            self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.process_image(self.left_image)

    def closest_to_centroid(self, contour_points, cX, cY):
        return min(contour_points, key=lambda c: abs(cv2.pointPolygonTest(c,(cX,cY),True)))

    def report(self, contour, area, cX, cY, closest, ellipse_area):
        print('Contour Detected')
        print('Contour Area:', area)
        print('Centroid', cX, cY)
        print('Closest Point', closest[0], closest[1])
        print('Ellipse Area:', ellipse_area)
        print('---')

    def preprocess(self, image):
        image_in = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # scipy.misc.imsave("camera_data/uncorrected.jpg", image_in)
        image_in = cv2.bilateralFilter(image_in, 9, 75, 75)
        h, s, v = cv2.split(cv2.cvtColor(image_in, cv2.COLOR_RGB2HSV))
        nonSat = s < 180
        disk = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        nonSat = cv2.erode(nonSat.astype(np.uint8), disk)
        v2 = v.copy()
        v2[nonSat == 0] = 0
        glare = v2 > 240;
        glare = cv2.dilate(glare.astype(np.uint8), disk);
        corrected = cv2.inpaint(image_in, glare, 5, cv2.INPAINT_NS)
        # scipy.misc.imsave("camera_data/corrected.jpg", corrected)
        gray = cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        scipy.misc.imsave('camera_data/thresh.jpg', thresh)
        return thresh


    # Working now
    def process_image(self, image):
        thresh = self.preprocess(image)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            M = cv2.moments(c)
            area = cv2.contourArea(c)
            if int(M["m00"]) != 0 and (self.area_lower < area < self.area_upper):
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                closest = np.vstack(self.closest_to_centroid(c, cX, cY)).squeeze()
                ellipse = cv2.fitEllipse(c)
                (x,y), (ma,MA), angle = ellipse
                aspect_ratio = ma/MA
                ellipse_area = (np.pi * ma * MA)/4
                if (0.75 < aspect_ratio < 1.0) and (self.ellipse_area_lower < ellipse_area):
                    self.report(c, area, cX, cY, closest, ellipse_area)
                    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                    cv2.ellipse(image, ellipse, (255, 0, 0), 2)
                    cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
                    cv2.putText(image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.circle(image, (closest[0], closest[1]), 10, (0, 0, 0), -1)
                # else:
                #     cv2.drawContours(self.right_image, [c], -1, (0, 0, 255), 2)
                #     cv2.ellipse(self.right_image, ellipse, (0, 0, 255), 2)
                #     cv2.putText(self.right_image, "REJECTED", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # scipy.misc.imsave('camera_data/fitted.jpg', self.right_image)

        
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