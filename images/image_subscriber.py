import cv_bridge
import rospy
import scipy.misc
from sensor_msgs.msg import Image, CameraInfo

class ImageSubscriber:

    def __init__(self, write=False):
        self.right_image = None
        self.left_image = None
        self.info = {'l': None, 'r': None}
        self.bridge = cv_bridge.CvBridge()
        self.write = write
        self.left_called = False
        self.right_called = False
        self.left_img_id = 0
        self.right_img_id = 0


        #========SUBSCRIBERS========#
        # image subscribers
        rospy.Subscriber("/endoscope/left/image_rect_color", Image,
                         self.left_image_callback)
        rospy.Subscriber("/endoscope/right/image_rect_color", Image,
                         self.right_image_callback)
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
        self.right_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        if self.write:
            if self.right_called:
                scipy.misc.imsave('right' + str(self.right_img_id) + '.jpg', self.right_image)
                self.right_img_id += 1
                self.right_called = False

    def left_image_callback(self, msg):
        if rospy.is_shutdown():
            return
        # rospy.sleep(.5)
        self.left_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        if self.write:
            if self.left_called:
                scipy.misc.imsave('left' + str(self.left_img_id) + '.jpg', self.left_image)
                self.left_img_id += 1
                self.left_called = False

if __name__ == "__main__":
    a = ImageSubscriber(write=True)
    rospy.spin()
