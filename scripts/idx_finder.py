#!/usr/bin/env python3

import rospy
import math
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image, CameraInfo

from ocr_localizer import OCRLocalizer
from histogram_localizer import HistogramLocalizer
from cropper import Cropper

from idx_finder.srv import IDXAcquisition, IDXAcquisitionResponse
from matplotlib import pyplot as plt

class IDXFinder:
    def __init__(self):
        rospy.init_node('idx_finder_server')

        self.use_ocr = True

        self.bridge = CvBridge()
        self.load_params()

        self.voc0_img_path = '/home/jau/ros/catkin_ws/src/idx_finder/scripts/last_voc0.png' # ATM it seems as if absolute path is needed
        self.voc1_img_path = '/home/jau/ros/catkin_ws/src/idx_finder/scripts/last_voc1.png'

        # Set up callbacks
        color_img_sub = message_filters.Subscriber(self.color_topic_name,Image, queue_size=1)
        
        # Debugging
        self.debug_col_pub = rospy.Publisher("debug_color", Image, queue_size=1)
        self.debug_mask_pub = rospy.Publisher("debug_mask", Image, queue_size=1)
        self.debug_all_mask_pub = rospy.Publisher("debug_all_mask", Image, queue_size=1)
        self.debug_histogram_pub = rospy.Publisher("debug_histogram", Image, queue_size=1)
        self.debug_voc0_pub = rospy.Publisher("debug_voc0", Image, queue_size=1)
        self.debug_voc1_pub = rospy.Publisher("debug_voc1", Image, queue_size=1)
        
        # Set up time synchronizer
        ts = message_filters.TimeSynchronizer([color_img_sub], 10)
        ts.registerCallback(self.synch_image_callback)

        # Start service
        self.service = rospy.Service("idx_finder_server", IDXAcquisition, self.service_request_callback)

        rospy.loginfo("[IDXFinder] : "+str("Initialized"))

    
    def synch_image_callback(self, color_msg):
        try:
            cv_color_img = self.bridge.imgmsg_to_cv2(color_msg).copy()
            self.last_image_color = cv_color_img

            # Update cropped images
            self.cropper = Cropper(self.last_image_color)
            if self.cropper.status == "success":
                cv2.imwrite(self.voc0_img_path,self.cropper.voc0_img)
                cv2.imwrite(self.voc1_img_path,self.cropper.voc1_img)

                # ID salt and pepper
                self.coms_dict["salt"] = self.cropper.c2_center
                self.quadrant_dict["salt"] = self.cropper.locate(self.coms_dict["salt"])

                # Pepper localization
                all_idxs = set([0,1,2,3])
                self.quadrant_dict["pepper"] = all_idxs.difference(self.cropper.occupied_idxs).pop()

                # Debug
                self.debug_col_pub.publish(self.bridge.cv2_to_imgmsg(self.cropper.col_cropped,encoding="rgb8"))

                if self.cropper.voc0_img is not None:
                    self.debug_voc0_pub.publish(self.bridge.cv2_to_imgmsg(self.cropper.voc0_img,encoding="rgb8"))
                
                if self.cropper.voc1_img is not None:
                    self.debug_voc1_pub.publish(self.bridge.cv2_to_imgmsg(self.cropper.voc1_img,encoding="rgb8"))
                #self.debug_all_mask_pub.publish(self.bridge.cv2_to_imgmsg(self.all_mask,encoding="passthrough"))

        except CvBridgeError as e:
            print(e)
        


    def service_request_callback(self, request):
        target = request.color_profile
        print("[IDXFinder] : Received request for:", target)
        
        # Prepare response
        response = IDXAcquisitionResponse()
        response.success = False
        response.idx = -1
        
        if target in self.quadrant_dict: # Salt or pepper was requested
            response.idx = self.quadrant_dict[target]
            response.success = True
        else: # Oil or vinegar was requested
            if self.last_image_color is not None:

                oil_com = None
                vinegar_com = None

                # Try OCR approach
                if self.use_ocr:
                    ocr_success = False
                    self.voc0_ocr_localizer = OCRLocalizer(self.voc0_img_path)
                    
                    if self.voc0_ocr_localizer.status == "oil":
                        oil_com = self.cropper.c0_center
                        vinegar_com = self.cropper.c1_center
                        ocr_success = True
                        print("OCR success for VOC0")
                    elif self.voc0_ocr_localizer.status == "vinegar":
                        oil_com = self.cropper.c1_center
                        vinegar_com = self.cropper.c0_center
                        ocr_success = True
                        print("OCR success for VOC0")
                    
                    if not ocr_success:
                        print("OCR failed for VOC0")
                        self.voc1_ocr_localizer = OCRLocalizer(self.voc1_img_path)
                        if self.voc1_ocr_localizer.status == "oil":
                            oil_com = self.cropper.c1_center
                            vinegar_com = self.cropper.c0_center
                            ocr_success = True
                            print("OCR success for VOC1")
                        elif self.voc1_ocr_localizer.status == "vinegar":
                            oil_com = self.cropper.c0_center
                            vinegar_com = self.cropper.c1_center
                            ocr_success = True
                            print("OCR success for VOC1")
                    if not ocr_success:
                        print("OCR failed for VOC1")

                    # Delete OCR localizer
                    if self.voc0_ocr_localizer is not None:
                        self.voc0_ocr_localizer = None
                    if self.voc1_ocr_localizer is not None:
                        self.voc1_ocr_localizer = None
                    
                if not self.use_ocr or not ocr_success: # Try histogram approach
                    HL = HistogramLocalizer(self.cropper.voc0_img,self.cropper.voc1_img)

                    # Debug
                    if HL.hist_img is not None:
                        self.debug_histogram_pub.publish(self.bridge.cv2_to_imgmsg(HL.hist_img,encoding="rgb8"))

                    # Vinegar/Oil classification decision
                    if HL.voc0_has_higher_hue_peak:
                        oil_com = self.cropper.c0_center
                        vinegar_com = self.cropper.c1_center
                    elif HL.voc1_has_higher_hue_peak:
                        oil_com = self.cropper.c1_center
                        vinegar_com = self.cropper.c0_center
                    else:
                        print("Same peak x --> deciding via mean")
                        if HL.voc0_has_higher_hue_mean:
                            oil_com = self.cropper.c0_center
                            vinegar_com = self.cropper.c1_center
                        elif HL.voc1_has_higher_hue_mean:
                            oil_com = self.cropper.c1_center
                            vinegar_com = self.cropper.c0_center

                        self.quadrant_dict = {}
                        return response


                self.coms_dict["oil"] = oil_com
                self.coms_dict["vinegar"] = vinegar_com

                self.quadrant_dict["oil"] = self.cropper.locate(self.coms_dict["oil"])
                self.quadrant_dict["vinegar"] = self.cropper.locate(self.coms_dict["vinegar"])


                # Finalize response
                response.idx = self.quadrant_dict[target]
                if response.idx in [0,1,2,3]:
                    response.success = True

        self.quadrant_dict = {}
        return response

 
    
    def load_params(self):

        # Save which contour center is in which quadrant
        self.quadrant_dict = {}
        # Save com of four bottles
        self.coms_dict = {}

        # Declare cropper
        self.cropper = None
        self.voc0_ocr_localizer = None
        self.voc1_ocr_localizer = None

        self.debug_image = None
        self.all_mask = None
        self.min_pixel_count_valid_mask = 10
       
        sim = True
        if sim: 
            self.color_topic_name = "/camera/color/image_raw"
            self.depth_topic_name = "/camera/aligned_depth_to_color/image_raw"
            self.cam_info_topic_name = "/camera/color/camera_info"
        else: # TODO
            self.color_topic_name = "/dynaarm_REALSENSE/color/image_raw"
            self.depth_topic_name = "/dynaarm_REALSENSE/aligned_depth_to_color/image_raw"
            self.cam_info_topic_name = "/dynaarm_REALSENSE/aligned_depth_to_color/camera_info"


if __name__ == '__main__':
    rospy.init_node('idx_finder_server')
    server = IDXFinder()
    rospy.spin()
