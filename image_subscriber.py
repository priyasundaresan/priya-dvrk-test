import cv_bridge
import rospy
import scipy.misc
from sensor_msgs.msg import Image, CameraInfo

class ImageSubscriber:

    def __init__(self, write=False, queue=None):
        self.right_image = None
        self.left_image = None
        self.info = {'l': None, 'r': None}
        self.bridge = cv_bridge.CvBridge()
        self.write = write
        self.queue = queue
        self.left_img_id = 0
        self.right_img_id = 0


        #========SUBSCRIBERS========#
        # image subscribers
        rospy.init_node('image_saver', anonymous=True)
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
        self.right_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        if self.write:
            self.right_img_id += 1
            scipy.misc.imsave('right' + str(right_img_id) + '.jpg', self.right_image)

    def left_image_callback(self, msg):
        if rospy.is_shutdown():
            return
        # rospy.sleep(.5)
        self.left_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        if self.write:
            self.left_img_id += 1
            scipy.misc.imsave('left' + str(left_img_id) + '.jpg', self.left_image)
        if self.queue is not None:
            self.queue.put(self.left_image)

if __name__ == "__main__":
    a = ImageSubscriber(write=True)
    rospy.spin()
