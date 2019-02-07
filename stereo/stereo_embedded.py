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
import transform
import read_camera

USE_SAVED_IMAGES = False
USE_SPLIT_VIEW = True

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
        self.ellipse_upper = 195000 #play with this, was 180000 before
        self.residual_lower = 250 #play with this, was 250 before
        self.residual_upper = 2000 #play with this, was 2000 before
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

    def find_residual(self, point, contours):
        """Finds the closest contour from a given list to the current point.
        Used for locating the smaller residual corresponding to the large protruding part
        of a needle"""
        x, y = point
        if len(contours) > 0:
            return min(contours, key=lambda c: self.distance_pt_to_contour(c, x, y))
        return None

    def report(self, area, cx, cy, CX, CY, ellipse_area, flag=None):
        if flag is not None:
            print(flag)
        print('Contour Detected')
        print('Contour Area:', area)
        print('Centroid', cx, cy)
        print('Closest Point', CX, CY)
        print('Ellipse Area:', ellipse_area)
        print('---')

    def preprocess(self, image):
    	image_in = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        corrected = np.uint8(cv2.pow(image_in/255.0, 1.4) * 255)
        # scipy.misc.imsave("camera_data/gamma_corrected.jpg", corrected)
        gray = cv2.cvtColor(image_in, cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # scipy.misc.imsave("camera_data/thresh.jpg", thresh)
        return thresh


    # Working now
    def process_image(self, image):

        thresh = self.preprocess(image)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # All potential smaller-end needle protrusions
        residuals = [c for c in contours if self.residual_lower < cv2.contourArea(c) < self.residual_upper]

        for r in residuals:
            cv2.drawContours(image, [r], 0, (0, 255, 0), 2)

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
                    cv2.circle(image, center, 10, (255, 0, 0), -1)
                    # cv2.circle(image, (centroid_x, centroid_y), 10, (255, 255, 255), -1)
                    self.report(area, centroid_x, centroid_y, cx, cy, ellipse_area, 'LARGE RESIDUAL')
                    # cv2.ellipse(image, ellipse, (0, 0, 255), 2)
                    cv2.drawContours(image, [c], 0, (0, 255, 255), 2)
                    
                    # Find the corresponding small residual and markup
                    residual = self.find_residual(center, residuals)
                    if residual is not None:
                        print("SMALL RESIDUAL", cv2.contourArea(residual))
                        print(self.get_ellipse(residual)[-2])
                        residual_centroid = self.compute_centroid(residual)
                        cv2.putText(image, "residual", residual_centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
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
                        pull_x = int(cx + 200*dx)
                        pull_y = int(cy + 200*dy)
                        cv2.circle(image, (pull_x, pull_y), 10, (0, 0, 0), -1)
                        cv2.line(image, center, (pull_x, pull_y), (0, 0, 0), 2)

                        # Compute points in right camera frame (residual center, contour center, pull point)
                        left_center = np.matrix([cx, cy, 0])
                        left_pull = np.matrix([pull_x, pull_y, 0])
                        right_center = transform.transform_data("Left Frame", "Right Frame", left_center, self.TL_R, verbose=False)
                        right_pull = transform.transform_data("Left", "Right", left_pull, self.TL_R, verbose=False)
                        right_cx = int(right_center[0, 0])
                        right_cy = int(right_center[0, 1])
                        right_pull_x = int(right_pull[0, 0])
                        right_pull_y = int(right_pull[0, 1])
                        cv2.circle(self.right_image, (right_cx, right_cy), 10, (0, 0, 0), -1)
                        cv2.circle(self.right_image, (right_pull_x, right_pull_y), 10, (0, 0, 0), -1)
                        cv2.line(self.right_image, (right_cx, right_cy), (right_pull_x, right_pull_y), (0, 0, 0), 2)
            # elif 250 < area < 500:
            #     cv2.drawContours(image, [c], 0, (0, 255, 255), 2)

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
