import image_geometry
import rospy
from geometry_msgs.msg import PointStamped, Point
import cv2
import cv_bridge
import numpy as np
import rospy, scipy.misc
from sensor_msgs.msg import Image, CameraInfo
import time
import pickle
import sys
import pprint

USE_SAVED_IMAGES = False

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

class EllipseDetector:

    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.left_image = None
        self.right_image = None
        self.info = {'l': None, 'r': None, 'b': None, 'd': None}
        self.plane = None
        self.area_lower_bound = 700
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
        # if self.left_image != None:
        #     self.process_image()

    def left_image_callback(self, msg):
        if rospy.is_shutdown():
            return
        if USE_SAVED_IMAGES:
            self.left_image = cv2.imread('left_checkerboard.jpg')
        else:
            self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            scipy.misc.imsave('camera_data/unfitted_image.jpg', self.left_image)
        if self.right_image is not None:
            self.process_image([self.left_image, self.right_image])

    def get_points_3d(self, left_points, right_points):
        """ this method assumes that corresponding points are in the right order
            and returns a list of 3d points """

        # both lists must be of the same lenghth otherwise return None
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

    def closest_to_centroid(self, contour_points, cX, cY):
        return min(contour_points, key=lambda c: abs(cv2.pointPolygonTest(c,(cX,cY), True)))

    def annotate(self, image, cX, cY, true_center, contours, area):
        print('\nContour Detected')
        print('Centroid', (cX, cY))
        print('Closest Point', true_center)
        print('Area', area)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 7, (255, 0, 0), -1)
        cv2.putText(image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.circle(image, true_center, 10, (0, 0, 0), -1)

    # Working now
    def process_image(self, images):
        left, right = [], []
        for image in images:
            inverted = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            scipy.misc.imsave('camera_data/inverted.jpg', inverted)
            thresh = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY_INV)[1]
            scipy.misc.imsave('camera_data/thresh.jpg', thresh)
            im2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                M = cv2.moments(c)
                area = cv2.contourArea(c)
                if int(M["m00"]) != 0 and (self.area_lower_bound < area < self.area_upper_bound):
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    closest = np.vstack(self.closest_to_centroid(c, cX, cY)).squeeze()
                    true_center = (closest[0], closest[1])
                    if image is self.left_image:
                        left.append(true_center)
                        self.annotate(image, cX, cY, true_center, [c], area)
                        scipy.misc.imsave('camera_data/fitted_image.jpg', image)
                    else:
                        right.append(true_center)
        if len(left) == len(right) and len(left) > 0 and len(right) > 0:
            pts3d = self.get_points_3d(left, right)
            print("Found")
            self.pts = [(p.point.x, p.point.y, p.point.z) for p in pts3d]
            pprint.pprint(self.pts)
            with open('needle_data/needle_points.p', "w+") as f:
                pickle.dump(self.pts, f)
            rospy.signal_shutdown("Finished.")
        else:
            print("left", len(left))
            print("right", len(right))

if __name__ == "__main__":
    a = EllipseDetector()
    while 1:
        frame = a.left_image
        if frame is None:
            continue
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
