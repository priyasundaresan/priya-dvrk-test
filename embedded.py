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

class EmbeddedNeedleDetector():

    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.left_image = None
        self.right_image = None
        self.corrected_left = None
        self.corrected_right = None
        self.info = {'l': None, 'r': None, 'b': None, 'd': None}
        self.plane = None
        self.area_lower = 1000
        self.area_upper = 20000
        self.box_upper = 40000
        self.ellipse_lower = 1300
        self.ellipse_upper = 120000

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
        if self.left_image is not None:
        	self.corrected_right = self.preprocess(self.right_image)
        	self.process_image(self.corrected_right)

    def left_image_callback(self, msg):
        if rospy.is_shutdown():
            return
        if USE_SAVED_IMAGES:
            self.left_image = cv2.imread('left_checkerboard.jpg')
        else:
            self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")


    def closest_to_centroid(self, contour_points, cX, cY):
        return min(contour_points, key=lambda c: abs(cv2.pointPolygonTest(c,(cX,cY),True)))

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
    	scipy.misc.imsave("camera_data/uncorrected.jpg", image_in)
        h, s, v = cv2.split(cv2.cvtColor(image_in, cv2.COLOR_RGB2HSV))
        nonSat = s < 180
        disk = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        nonSat = cv2.erode(nonSat.astype(np.uint8), disk)
        v2 = v.copy()
        v2[nonSat == 0] = 0
        glare = v2 > 240;
        glare = cv2.dilate(glare.astype(np.uint8), disk);
        corrected = cv2.inpaint(image_in, glare, 5, cv2.INPAINT_NS)
        scipy.misc.imsave("camera_data/corrected.jpg", corrected)
        img_yuv = cv2.cvtColor(corrected, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        scipy.misc.imsave("camera_data/equalized.jpg", img_output)
        gray = cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY)
        scipy.misc.imsave('camera_data/gray.jpg', gray)
        thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        scipy.misc.imsave('camera_data/thresh.jpg', thresh)
        return thresh


    # Working now
    def process_image(self, image):

        im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            M = cv2.moments(c)
            area = cv2.contourArea(c)
            if int(M["m00"]) != 0 and (self.area_lower < area < self.area_upper):

            	cX = int(M["m10"] / M["m00"])
            	cY = int(M["m01"] / M["m00"])
            	closest = np.vstack(self.closest_to_centroid(c, cX, cY)).squeeze()
            	true_center = (closest[0], closest[1])

            	ellipse = cv2.fitEllipse(c)
            	(x,y), (ma,MA), angle = ellipse
            	ellipse_aspect = ma/MA
            	ellipse_area = (np.pi * ma * MA)/4

            	rect = cv2.minAreaRect(c)
            	(x,y), (dim1, dim2), angle = rect
            	width, height = min(dim1, dim2), max(dim1, dim2)
            	box_area = width * height
            	box_aspect = width / height
            	box = cv2.boxPoints(rect)
            	box = np.int0(box)

            	ratio = max(box_area, ellipse_area)/min(box_area, ellipse_area)

            	if box_area < self.box_upper and self.ellipse_lower < ellipse_area < self.ellipse_upper:
            		# cv2.circle(self.right_image, (cX, cY), 7, (255, 255, 255), -1)
            		cv2.putText(self.right_image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            		cv2.circle(self.right_image, true_center, 10, (0, 0, 0), -1)
            		self.report(c, area, cX, cY, closest, ellipse_area, box_area)
            		cv2.ellipse(self.right_image, ellipse, (0, 0, 255), 2)
            		cv2.drawContours(self.right_image,[box],0,(0,255,0),2)
            		rows,cols = self.right_image.shape[:2]
            		line = cv2.fitLine(c, cv2.DIST_L2,0,0.01,0.01)
            		[vx,vy,x,y] = line
            		lefty = int((-x*vy/vx) + y)
            		righty = int(((cols-x)*vy/vx)+y)
            		cv2.line(self.right_image,(cols-1,righty),(0,lefty),(255, 0, 0),2)
                    
                # else:
                #     cv2.drawContours(self.right_image, [c], -1, (0, 0, 255), 2)
                #     cv2.ellipse(self.right_image, ellipse, (0, 0, 255), 2)
                #     cv2.putText(self.right_image, "REJECTED", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

if __name__ == "__main__":
    a = EmbeddedNeedleDetector()
    while 1:
        frame = a.right_image
        if frame is None:
            continue
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    # rospy.spin()
