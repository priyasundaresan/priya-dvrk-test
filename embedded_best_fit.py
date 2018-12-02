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
        self.area_lower = 1800
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

    def compute_centroid(self, contour, moments=None):
        """Computes the centroid of a given contour"""
        if not moments:
            moments = cv2.moments(contour)
        if int(moments["m00"]) == 0:
            return
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])
        return (centroid_x, centroid_y)

    def distance(self, p1, p2):
        """Computes the distance between two points"""
        if type(p1) == np.ndarray:
            return cv2.pointPolygonTest(p1, p2, True)
        x1, y1 = p1
        x2, y2 = p2
        return np.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))

    def distance_pt_to_contour(self, contour, x, y):
        """Computes the distance from a point to the center (not centroid) of contour"""
        centroid_x, centroid_y = self.compute_centroid(contour)
        center = self.center(contour, centroid_x, centroid_y)
        return abs(self.distance(center, (x, y)))

    def get_ellipse(self, c):
        """Returns the points, aspect ratio, and area of the best-fit ellipse to a given contour"""
        ellipse = cv2.fitEllipse(c)
        (x,y), (ma,MA), angle = ellipse
        ellipse_aspect = ma/MA
        ellipse_area = (np.pi * ma * MA)/4
        return (ellipse, ellipse_aspect, ellipse_area)

    def center(self, contour, centroid_x, centroid_y):
        """Finds the center of the contour (closest point on contour to centroid (center of mass))
        i.e. finds the midpoint/pickup point of a needle
        """
        return min(contour, key=lambda point: abs(self.distance(point, (centroid_x, centroid_y))))

    def find_residual(self, point, contours):
        """Finds the closest contour from a given list to the current point.
        Used for locating the smaller residual corresponding to the large protruding part
        of a needle"""
        x, y = point
        return min(contours, key=lambda c: self.distance_pt_to_contour(c, x, y))

    def report(self, area, centroid_x, centroid_y, CX, CY, ellipse_area):
        """Reports contour information"""
        print('Contour Detected')
        print('Contour Area:', area)
        print('Centroid', centroid_x, centroid_y)
        print('Closest Point', CX, CY)
        print('Ellipse Area:', ellipse_area)
        print('---')

    def preprocess(self, image):
        """Gamma correction (darkening) for better contrast --> grayscale --> otsu threshold"""
    	image_in = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        corrected = np.uint8(cv2.pow(image_in/255.0, 1.4) * 255)
        gray = cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        return thresh


    # Working now
    def process_image(self, image):
        """ Locates all needle contours in the image, pairs corresponding
        large and small needle protrusions, and identifies a grasp point (center)
        and pull point (pull_x, pull_y) for the needle. Marks up the image
        to show needle contours, centers, centroids, and best fit lines. """

        thresh = self.preprocess(image)
        # Get all contours of the image
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # All potential smaller-end needle protrusions
        residuals = [c for c in contours if 600 < cv2.contourArea(c) < 2000]

        for c in contours:

            # Get moments and area for given contour
            M = cv2.moments(c)
            area = cv2.contourArea(c)

            # Throw out all non-needle contours
            if (self.area_lower < area < self.area_upper):

                # Compute the centroid (center of mass) and center of the given needle
            	centroid_x, centroid_y = self.compute_centroid(c, M)
            	closest = np.vstack(self.center(c, centroid_x, centroid_y)).squeeze()
                cx, cy = closest[0], closest[1]
            	center = (cx, cy)

                # Fit an ellipse to the contour
            	ellipse, ellipse_aspect, ellipse_area = self.get_ellipse(c)

                """Contour is the big protruding part of the needle"""
            	if self.ellipse_lower < ellipse_area < self.ellipse_upper:

                    # Report/display the large residual
                    cv2.putText(image, "centroid", (centroid_x - 20, centroid_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.circle(image, center, 10, (0, 0, 0), -1)
                    # cv2.circle(image, (centroid_x, centroid_y), 10, (255, 255, 255), -1)
                    self.report(area, centroid_x, centroid_y, cx, cy, ellipse_area)
                    # cv2.ellipse(image, ellipse, (0, 0, 255), 2)
                    cv2.drawContours(image, [c], 0, (0, 255, 255), 2)
                    
                    # Find the corresponding small residual and markup
                    residual = self.find_residual(center, residuals)
                    residual_centroid = self.compute_centroid(residual)
                    cv2.putText(image, "residual", residual_centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                    cv2.drawContours(image, [residual], 0, (255, 255, 255), 2)
                    cv2.circle(image, residual_centroid, 10, (255, 0, 0), -1)
                    
                    # Fit a line to the small residual
                    [vx, vy, x, y] = cv2.fitLine(residual, cv2.DIST_L2,0,0.01,0.01)
                    dx, dy = np.asscalar(vx), np.asscalar(vy)
                    # rows, cols = image.shape[:2]
                    # lefty = int((-x*vy/vx) + y)
                    # righty = int(((cols-x)*vy/vx)+y)
                    # cv2.line(image,(cols-1,righty),(0,lefty),(0,255,0),2)

                    """Finds a pull point (relative to contour center) in the direction
                    of the best fit line of the smaller residual and opposite 
                    (not towards) the smaller residual """
                    if self.distance(residual_centroid, center) > \
                       self.distance(residual_centroid, (cx + dx, cy + dy)):
                        dx, dy = -dx, -dy
                    pull_x = int(cx + 100*dx)
                    pull_y = int(cy + 100*dy)
                    cv2.circle(image, (pull_x, pull_y), 10, (0, 0, 0), -1)
                    cv2.line(image, center, (pull_x, pull_y), (0, 0, 0), 2)

                
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
