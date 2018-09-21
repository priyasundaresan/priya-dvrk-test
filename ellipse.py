import image_geometry
import rospy
from geometry_msgs.msg import PointStamped, Point
import cv2
import cv_bridge
import numpy as np
import rospy, scipy.misc
from sensor_msgs.msg import Image, CameraInfo

USE_SAVED_IMAGES = False

class EllipseDetector:

    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.left_image = None
        self.right_image = None
        self.info = {'l': None, 'r': None, 'b': None, 'd': None}
        self.plane = None

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
            scipy.misc.imsave('camera_data/unfitted_image.jpg', self.left_image)
        if self.right_image != None:
            self.process_image()


    def process_image(self):
        print "processing image"
        inverted = cv2.bitwise_not(cv2.cvtColor(self.left_image, cv2.COLOR_BGR2GRAY))
        blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
        thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)[1]
        im2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(np.array(contours), type(contours))
        ellipse = cv2.fitEllipse(contours)
        left_image_annotated = cv2.ellipse(thresh, ellipse, (0, 255, 0), 2)
        scipy.misc.imsave('camera_data/fitted_image.jpg', left_image_annotated)

if __name__ == "__main__":
    a = EllipseDetector()
    rospy.spin()
