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

class EmbeddedNeedleDetector:

    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.left_image = None
        self.right_image = None
        self.info = {'l': None, 'r': None, 'b': None, 'd': None}
        self.plane = None
        self.area_lower = 1700
        self.area_upper = 20000
        self.ellipse_lower = 1300
        self.ellipse_upper = 180000

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
            self.corrected_left = self.preprocess(self.left_image)
            self.corrected_right = self.preprocess(self.right_image)
            self.process_image(self.corrected_left, self.corrected_right)

    def left_image_callback(self, msg):
        if rospy.is_shutdown():
            return
        if USE_SAVED_IMAGES:
            self.left_image = cv2.imread('left_checkerboard.jpg')
        else:
            self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def get_points_3d(self, left_points, right_points):
        """ this method assumes that corresponding points are in the right order
            and returns a list of 3d points """

        # both lists must be of the same lenghth otherwise return None
        # left_points, right_points = sorted(left_points), sorted(right_points)
        print("\nLeft/Right points:")
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
        return points_3d

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
        print(self.get_ellipse(contour)[2])
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
        if image is self.left_image:
            scipy.misc.imsave("camera_data/left_corrected.jpg", corrected)
        else:
            scipy.misc.imsave("camera_data/right_corrected.jpg", corrected)
        gray = cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        return thresh


    # Working now
    def process_image(self, *images):

        left, right = [], []
        residuals = []

        for image in images:

            im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:

                M = cv2.moments(c)
                area = cv2.contourArea(c)

                if (self.area_lower < area < self.area_upper):
                    cx, cy = self.compute_centroid(c, M)
                    closest = np.vstack(self.center(c, cx, cy)).squeeze()
                    CX, CY = closest[0], closest[1]
                    true_center = (CX, CY)

                    ellipse, ellipse_aspect, ellipse_area = self.get_ellipse(c)

                    if self.ellipse_lower < ellipse_area < self.ellipse_upper:

                        endpoint = tuple(np.vstack(self.endpoint(c, cx, cy)).squeeze())
                        EX, EY = endpoint[0], endpoint[1]
                        dx, dy = CX - EX, CY - EY
                        OX, OY = CX + dx, CY + dy
                        opposite = (OX, OY)

                        if image is self.corrected_right:
                            right.append(true_center)
                            right.append(opposite)

                            cv2.putText(self.right_image, "center", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            cv2.circle(self.right_image, true_center, 10, (0, 0, 0), -1)
                            self.report(area, cx, cy, CX, CY, ellipse_area)
                            cv2.ellipse(self.right_image, ellipse, (0, 0, 255), 2)
                            cv2.drawContours(self.right_image, [c], 0, (0, 255, 255), 2)

                            # cv2.circle(self.right_image, endpoint, 10, (255, 255, 255), -1)
                            cv2.circle(self.right_image, opposite, 10, (0, 0, 0), -1)
                            # cv2.line(self.right_image, true_center, endpoint, (255, 255, 255), 10)
                            cv2.line(self.right_image, true_center, opposite, (0, 0, 0), 10)
                        else:
                            left.append(true_center)
                            left.append(opposite)

                            cv2.putText(self.left_image, "center", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            cv2.circle(self.left_image, true_center, 10, (0, 0, 0), -1)
                            self.report(area, cx, cy, CX, CY, ellipse_area)
                            cv2.ellipse(self.left_image, ellipse, (0, 0, 255), 2)
                            cv2.drawContours(self.left_image, [c], 0, (0, 255, 255), 2)

                            # cv2.circle(self.left_image, endpoint, 10, (255, 255, 255), -1)
                            cv2.circle(self.left_image, opposite, 10, (0, 0, 0), -1)
                            # cv2.line(self.left_image, true_center, endpoint, (255, 255, 255), 10)
                            cv2.line(self.left_image, true_center, opposite, (0, 0, 0), 10)
                else:
                    residuals.append(c)
                    # else:
                    #     cv2.drawContours(image, [c], -1, (0, 0, 255), 2)
                    #     cv2.ellipse(image, ellipse, (0, 0, 255), 2)
                    #     cv2.putText(image, "REJECTED", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
        if len(right) > 0 and len(right) == len(left):
            for pair in zip(right, left):
                print(self.distance(np.array(pair[0]).reshape(1, 2), (pair[1])))
            pts3d = self.get_points_3d(left, right)
            print("Found")
            self.pts = [(p.point.x, p.point.y, p.point.z) for p in pts3d]
            pprint.pprint(self.pts)
            with open('needle_data/needle_points.p', "w+") as f:
                pickle.dump(self.pts, f)
            rospy.signal_shutdown("Finished.")


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
            frame = a.right_image
            if frame is None:
                continue
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
