#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import message_filters

from sensor_msgs.msg import Image
from ocr_localizer import OCRLocalizer
from histogram_localizer import HistogramLocalizer
from cropper import Cropper

from spice_up_coordinator.srv._IDXAcquisition import IDXAcquisition,IDXAcquisitionRequest


class IDXFinder:
    def __init__(self):
        rospy.init_node('idx_finder_client')


        self._bridge = CvBridge()
        path = '/home/jau/ros/catkin_ws/src/idx_finder/scripts/debug_imgs/'
        mask_path = path + 'og_mask.png'
        color_path = path + 'og_color.png'

        rospy.wait_for_service('idx_finder_server')

        # Create test request
        target_spice = "salt"

        mask = cv2.imread(mask_path)
        color = cv2.imread(color_path)

        mask_msg = self.cv2_to_ros(mask)
        color_msg = self.cv2_to_ros(color)

        idx_request = IDXAcquisitionRequest()
        
        idx_request.target_spice = target_spice
        idx_request.mask = mask_msg
        idx_request.color_frame = color_msg
        idx_acquisition_service_handle = rospy.ServiceProxy('idx_finder_server', IDXAcquisition)
        print("[IDXFinderClient] : Requesting target spice IDX")
        idx_acquisition_service_response = idx_acquisition_service_handle(idx_request)
        
        target_idx = idx_acquisition_service_response.idx
        print("[IDXFinderClient] : Recieved idx: "+str(target_idx))

    def ros_to_cv2(self, frame: Image, desired_encoding="bgr8"):
        return self._bridge.imgmsg_to_cv2(frame, desired_encoding=desired_encoding)
    
    def cv2_to_ros(self, frame: np.ndarray):
        return self._bridge.cv2_to_imgmsg(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), encoding="rgb8")


   



if __name__ == '__main__':
    rospy.init_node('idx_finder_client')
    server = IDXFinder()
    rospy.spin()
