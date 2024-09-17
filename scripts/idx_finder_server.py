#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool

from sensor_msgs.msg import Image
from idx_finder.srv import IDXAcquisition, IDXAcquisitionResponse

from ocr_localizer import OCRLocalizer
from histogram_localizer import HistogramLocalizer
from cropper import Cropper



class IDXFinder:
    def __init__(self):
        rospy.init_node('idx_finder_server')

        self.use_ocr = True

        self.bridge = CvBridge()
        self.load_params()

        self.voc0_img_path = '/home/jau/ros/catkin_ws/src/idx_finder/scripts/last_voc0.png' # ATM it seems as if absolute path is needed
        self.voc1_img_path = '/home/jau/ros/catkin_ws/src/idx_finder/scripts/last_voc1.png'

        # Set up callbacks
        self.shutdown_sub = rospy.Subscriber("/shutdown_spice_up",Bool,self.shutdown_cb)
        #color_img_sub = message_filters.Subscriber(self.color_topic_name,Image, queue_size=1)
        
        # Debugging
        self.debug_col_pub = rospy.Publisher("debug_color", Image, queue_size=1)
        self.debug_mask_pub = rospy.Publisher("debug_mask", Image, queue_size=1)
        self.debug_all_mask_pub = rospy.Publisher("debug_all_mask", Image, queue_size=1)
        self.debug_histogram_pub = rospy.Publisher("debug_histogram", Image, queue_size=1)
        self.debug_voc0_pub = rospy.Publisher("/debug_voc0", Image, queue_size=1)
        self.debug_voc1_pub = rospy.Publisher("/debug_voc1", Image, queue_size=1)
        self.debug_blob_pub = rospy.Publisher("debug_blob", Image, queue_size=1)
        
        # Set up time synchronizer
        #ts = message_filters.TimeSynchronizer([color_img_sub], 10)
        #ts.registerCallback(self.synch_image_callback)

        # Start service
        self.service = rospy.Service("idx_finder_server", IDXAcquisition, self.service_request_callback)

        print("[IDXFinder] : Initialized")


    def shutdown_cb(self,signal):
        if signal.data:
            print("[IDXFinder] : Shutting down")
            rospy.signal_shutdown("Job done")

    
    def update_croppings(self):
        # Update cropped images
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


    def ros_to_cv2(self, frame: Image, desired_encoding="bgr8"):
        return self.bridge.imgmsg_to_cv2(frame, desired_encoding=desired_encoding)



    def service_request_callback(self, request):

        # Prepare response
        response = IDXAcquisitionResponse()
        response.success = False
        response.idx = -1

        target = request.target_spice
        target_mask_has_five_contours = request.has_five_contours
        print("[IDXFinder] : Received request for:", target)

        color_frame = self.ros_to_cv2(request.color_frame)

        if not self.cropper_init:
            original_mask_cv = self.ros_to_cv2(request.mask,desired_encoding="passthrough").astype(np.uint8)
            original_mask_cv = cv2.cvtColor(original_mask_cv,cv2.COLOR_BGR2GRAY)
        
            self.cropper = Cropper(color_frame,original_mask_cv,target_mask_has_five_contours)
            if self.cropper.status == "FAIL":
                print("[IDXFinder] : Cropper failed. Aborting!")
                return response
            
            self.cropper_init = True
            self.quadrant_dict = self.cropper.quadrant_dict
            print("[IDXFinder] : Cropper initialized")
            
        #self.update_croppings()
        print("[IDXFinder] : Qdict: "+str(self.quadrant_dict))

        if target in self.quadrant_dict: # Salt or pepper was requested
            response.idx = self.quadrant_dict[target]
            response.success = True
        else: # Oil or vinegar was requested
            
            oil_com = None
            vinegar_com = None

            # Try OCR approach
            if self.use_ocr:
                ocr_success = False
                self.vocA_ocr_localizer = OCRLocalizer(self.cropper.vocA_img_path)
                
                if self.vocA_ocr_localizer.status == "oil":
                    oil_com = self.cropper.cA_center
                    vinegar_com = self.cropper.cB_center
                    ocr_success = True
                    print("OCR success for VOCA")
                elif self.vocA_ocr_localizer.status == "vinegar":
                    oil_com = self.cropper.cB_center
                    vinegar_com = self.cropper.cA_center
                    ocr_success = True
                    print("OCR success for VOCA")
                
                if not ocr_success:
                    print("OCR failed for VOCA")
                    self.vocB_ocr_localizer = OCRLocalizer(self.cropper.vocB_img_path)
                    if self.vocB_ocr_localizer.status == "oil":
                        oil_com = self.cropper.cA_center
                        vinegar_com = self.cropper.cB_center
                        ocr_success = True
                        print("OCR success for VOCB")
                    elif self.vocB_ocr_localizer.status == "vinegar":
                        oil_com = self.cropper.cB_center
                        vinegar_com = self.cropper.cA_center
                        ocr_success = True
                        print("OCR success for VOCB")
                if not ocr_success:
                    print("OCR failed for VOCB")

            if not self.use_ocr or not ocr_success: # Try histogram approach
                # HL is agnostic to approach : 0 == A, 1 == B
                HL = HistogramLocalizer(self.cropper.vocA_img,self.cropper.vocB_img)
                if HL.status == "FAIL":
                    return response
                
                # Debug
                if HL.hist_img is not None:
                    self.debug_histogram_pub.publish(self.bridge.cv2_to_imgmsg(HL.hist_img,encoding="rgb8"))
                    print("Hist img published")
                    self.debug_voc0_pub.publish(self.bridge.cv2_to_imgmsg(self.cropper.vocA_img,encoding="rgb8"))
                    self.debug_voc1_pub.publish(self.bridge.cv2_to_imgmsg(self.cropper.vocB_img,encoding="rgb8"))

                # Vinegar/Oil classification decision
                if HL.voc0_has_higher_hue_mean:
                    oil_com = self.cropper.cB_center
                    vinegar_com = self.cropper.cA_center
                else:
                    oil_com = self.cropper.cA_center
                    vinegar_com = self.cropper.cB_center
                '''
                else:
                    print("Same hue peak location --> deciding via mean")
                    if HL.voc0_has_higher_hue_peak:
                        oil_com = self.cropper.cB_center
                        vinegar_com = self.cropper.cA_center
                    elif HL.voc1_has_higher_hue_mean:
                        oil_com = self.cropper.cA_center
                        vinegar_com = self.cropper.cB_center
                '''
            self.coms_dict["oil"] = oil_com
            self.coms_dict["vinegar"] = vinegar_com

            self.quadrant_dict["oil"] = self.cropper.locate(self.coms_dict["oil"])
            self.quadrant_dict["vinegar"] = self.cropper.locate(self.coms_dict["vinegar"])


            # Finalize response
            response.idx = self.quadrant_dict[target]
            if response.idx in [0,1,2,3]:
                response.success = True
                
        print("[IDXFinder] : Response sent")
        return response

 
    
    def load_params(self):

        self.debug_imgs_path='/home/jau/ros/catkin_ws/src/idx_finder/scripts/debug_imgs/'


        # Save which contour center is in which quadrant
        self.quadrant_dict = {}
        # Save com of four bottles
        self.coms_dict = {}

        # Declare cropper
        self.cropper = None
        self.cropper_init = False
        self.vocA_ocr_localizer = None
        self.vocB_ocr_localizer = None

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
