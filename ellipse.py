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

class EllipseDetector:

    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.left_image = None
        self.right_image = None
        self.info = {'l': None, 'r': None, 'b': None, 'd': None}
        self.plane = None
        self.area_lower_bound = 300
        self.area_upper_bound = 40000

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
        if self.left_image != None:
            self.process_image()

    def left_image_callback(self, msg):
        if rospy.is_shutdown():
            return
        if USE_SAVED_IMAGES:
            self.left_image = cv2.imread('left_checkerboard.jpg')
        else:
            self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            scipy.misc.imsave('camera_data/unfitted_image.jpg', self.left_image)
        # if self.right_image != None:
        #     self.process_image()

    def closest_to_centroid(self, contour_points, cX, cY):
        return min(contour_points, key=lambda c: abs(cv2.pointPolygonTest(c,(cX,cY),True)))

    # Working now
    def process_image(self):
        inverted = cv2.bitwise_not(cv2.cvtColor(self.right_image, cv2.COLOR_BGR2GRAY))
        scipy.misc.imsave('camera_data/inverted.jpg', inverted)
        thresh = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)[1]
        scipy.misc.imsave('camera_data/thresh.jpg', thresh)
        im2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            M = cv2.moments(c)
            area = cv2.contourArea(c)
            if int(M["m00"]) != 0 and (self.area_lower_bound < area < self.area_upper_bound):
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                closest = np.vstack(self.closest_to_centroid(c, cX, cY)).squeeze()
                # squeezed = np.vstack(c).squeeze()
                print('\nContour Detected')
                print('Centroid', cX, cY)
                print('Closest Point', closest[0], closest[1])
                cv2.drawContours(self.right_image, [c], -1, (0, 255, 0), 2)
                cv2.circle(self.right_image, (cX, cY), 7, (255, 0, 0), -1)
                cv2.putText(self.right_image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(self.right_image, (closest[0], closest[1]), 10, (0, 0, 0), -1)

        scipy.misc.imsave('camera_data/fitted_image.jpg', self.right_image)

        
if __name__ == "__main__":
    a = EllipseDetector()
    while 1:
        frame = a.right_image
        if frame is None:
            continue
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    # rospy.spin()
