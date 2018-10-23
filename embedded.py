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
        self.area_lower = 1600
        self.area_upper = 20000
        self.box_upper = 40000
        self.ellipse_lower = 1300
        self.ellipse_upper = 160000

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

    def closest_to_centroid(self, contour_points, cX, cY):
        return min(contour_points, key=lambda c: abs(cv2.pointPolygonTest(c,(cX,cY),True)))

    def endpoint(self, contour_points, cX, cY):
        return min(contour_points, key=lambda c: (cv2.pointPolygonTest(c,(cX,cY),True)))

    def report(self, contour, area, cX, cY, closest, ellipse_area, box_area):
        print('Contour Detected')
        print('Contour Area:', area)
        print('Centroid', cX, cY)
        print('Closest Point', closest[0], closest[1])
        print('Ellipse Area:', ellipse_area)
        print('Box Area:', box_area)
        print('---')

    def preprocess(self, image):
    	image_in = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_in = np.uint8(cv2.pow(image_in/255.0, 1.2) * 255)
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
        # gray = cv2.blur(gray, (5, 5))
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # if image is self.left_image:
        #     # scipy.misc.imsave('camera_data/threshleft.jpg', thresh)
        # else:
        #     # scipy.misc.imsave('camera_data/threshright.jpg', thresh)
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
                Cx, Cy = closest[0], closest[1]
            	true_center = (Cx, Cy)

            	ellipse = cv2.fitEllipse(c)
            	(x,y), (ma,MA), angle = ellipse
            	ellipse_aspect = ma/MA
            	ellipse_area = (np.pi * ma * MA)/4

            	rect = cv2.minAreaRect(c)
            	(x,y), (dim1, dim2), angle = rect
            	width, height = min(dim1, dim2), max(dim1, dim2)
            	box_area = width * height
            	box = cv2.boxPoints(rect)
            	box = np.int0(box)

            	if box_area < self.box_upper and self.ellipse_lower < ellipse_area < self.ellipse_upper:
                    cv2.putText(image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.circle(image, true_center, 10, (0, 0, 0), -1)
                    self.report(c, area, cX, cY, closest, ellipse_area, box_area)
                    cv2.ellipse(image, ellipse, (0, 0, 255), 2)
                    cv2.drawContours(image,[box],0,(0,255,0),2)
                    rows,cols = image.shape[:2]
                    endpoint = np.vstack(self.endpoint(c, cX, cY)).squeeze()
                    endpt = (endpoint[0], endpoint[1])
                    cv2.circle(image, endpt, 10, (0, 170, 0), -1)
                    line = cv2.fitLine(c, cv2.DIST_L2,0,0.01,0.01)
                    [vx,vy,x,y] = line
                    # [vx1, vy1, x1, y1] = [np.asscalar(i) for i in line]
                    # pull = (int(true_center[0] - 100*vx1), int(true_center[1] - 100*vy1))
                    # cv2.circle(image, pull, 10, (0, 255, 0), -1)
                    lefty = int((-x*vy/vx) + y)
                    righty = int(((cols-x)*vy/vx)+y)
                    cv2.line(image,(cols-1,righty),(0,lefty),(255, 0, 0),2)
                # else:
                #     cv2.drawContours(image, [c], -1, (0, 0, 255), 2)
                #     cv2.ellipse(image, ellipse, (0, 0, 255), 2)
                #     cv2.putText(image, "REJECTED", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # elif 500 < area < 2000:
            #     cv2.drawContours(image, [c], -1, (0, 255, 255), 2)

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
