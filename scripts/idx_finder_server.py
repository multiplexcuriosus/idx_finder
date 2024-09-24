#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from std_msgs.msg import Bool

from sensor_msgs.msg import Image
from idx_finder.srv import FindIndex,FindIndexResponse

from ocr_localizer import OCRLocalizer
from histogram_localizer import HistogramLocalizer
from cropper import Cropper



class IndexFinder:
    def __init__(self):
        '''
        The IndexFinder receives a FindIndexRequest and returns a FindIndexResponse.
        He first uses the request-mask to get cropped out color images from all spice bottles.
        For each of those four color images (voc0-3) he creates a brightness histogram.
        Each spice has a spice-index (just an enumeration) and each quadrant has a "location-index". 
        The spice, whose cropped out color image has the lowest brightness-mean in the brightness-histogram is assumed to be pepper.
        Analogously the second lowest brightness-mean is associated with salt. 

        Now two cropped out color images remain which must belong to oil and vinegar (vocA & vocB).
        First, OCR is run on vocA/B. If any specified token is found, vinegar and oil are successfully identified.
        If OCR failed, a hue histogram for vocA/B is created. 
        It is assumed, that the vocA/B with the more-right hue peak-location corresponds to vinegar.

        '''
        rospy.init_node('index_finder_server')

        self.load_params()

        # Set up callbacks
        self.shutdown_sub = rospy.Subscriber("/shutdown_spice_up",Bool,self.shutdown_cb)

        # Debugging
        self.debug_col_pub = rospy.Publisher("debug_color", Image, queue_size=1)
        self.debug_mask_pub = rospy.Publisher("debug_mask", Image, queue_size=1)
        self.debug_all_mask_pub = rospy.Publisher("debug_all_mask", Image, queue_size=1)
        self.debug_histogram_pub = rospy.Publisher("debug_histogram", Image, queue_size=1)
        self.debug_voc0_pub = rospy.Publisher("debug_voc0", Image, queue_size=1)
        self.debug_voc1_pub = rospy.Publisher("debug_voc1", Image, queue_size=1)
        self.debug_blob_pub = rospy.Publisher("debug_blob", Image, queue_size=1)
        
        # Start service
        self.service = rospy.Service("find_index_service", FindIndex, self.service_cb)

        print("[IndexFinder] : Initialized")
    
    def service_cb(self, request):

        # Prepare response
        response = FindIndexResponse()
        response.success = False
        response.idx = -1

        target_spice = request.target_spice
        target_mask_has_five_contours = request.has_five_contours
        print("[IndexFinder] : Received request for:", target_spice)

        color_frame = self.ros_to_cv2(request.color_frame)

        if not self.cropper_is_initialized:
            original_mask_cv = self.ros_to_cv2(request.mask,desired_encoding="passthrough").astype(np.uint8)
            original_mask_cv = cv2.cvtColor(original_mask_cv,cv2.COLOR_BGR2GRAY)
        
            self.cropper = Cropper(color_frame,original_mask_cv,target_mask_has_five_contours,self.debug)
            if self.cropper.status == "FAIL":
                print("[IndexFinder] : Cropper failed. Aborting!")
                return response
            
            self.cropper_is_initialized = True
            self.quadrant_dict = self.cropper.quadrant_dict
            print("[IndexFinder] : Cropper initialized")
            
        print("[IndexFinder] : Qdict: " + str(self.quadrant_dict))

        if target_spice in self.quadrant_dict: # Salt or pepper was requested, which the Cropper already localized
            response.idx = self.quadrant_dict[target_spice]
            response.success = True
        else: # Oil or vinegar was requested
            
            # Reset coms of oil and vinegar
            oil_com = None
            vinegar_com = None

            # Try OCR 
            if self.use_ocr:
                ocr_success = False
                self.vocA_ocr_localizer = OCRLocalizer(self.cropper.vocA_img_path,self.debug)
                
                if self.vocA_ocr_localizer.status == "oil":
                    oil_com = self.cropper.cA_center
                    vinegar_com = self.cropper.cB_center
                    ocr_success = True
                    print("[IndexFinder] : OCR success for VOCA")
                elif self.vocA_ocr_localizer.status == "vinegar":
                    oil_com = self.cropper.cB_center
                    vinegar_com = self.cropper.cA_center
                    ocr_success = True
                    print("[IndexFinder] : OCR success for VOCA")
                
                if not ocr_success:
                    print("[IndexFinder] : OCR failed for VOCA")
                    self.vocB_ocr_localizer = OCRLocalizer(self.cropper.vocB_img_path,self.debug)
                    if self.vocB_ocr_localizer.status == "oil":
                        oil_com = self.cropper.cA_center
                        vinegar_com = self.cropper.cB_center
                        ocr_success = True
                        print("[IndexFinder] : OCR success for VOCB")
                    elif self.vocB_ocr_localizer.status == "vinegar":
                        oil_com = self.cropper.cB_center
                        vinegar_com = self.cropper.cA_center
                        ocr_success = True
                        print("[IndexFinder] : OCR success for VOCB")
                if not ocr_success:
                    print("[IndexFinder] : OCR failed for VOCB")

            if not self.use_ocr or not ocr_success: # Try histogram approach
                # HL is agnostic to approach : 0 == A, 1 == B
                HL = HistogramLocalizer(self.cropper.vocA_img,self.cropper.vocB_img)
                if HL.status == "FAIL":
                    return response
                
                # Debug
                if HL.hist_img is not None:
                    self.debug_histogram_pub.publish(self.bridge.cv2_to_imgmsg(HL.hist_img,encoding="rgb8"))

                # Vinegar/Oil classification decision
                if HL.voc0_has_higher_hue_peak:
                    oil_com = self.cropper.cB_center
                    vinegar_com = self.cropper.cA_center
                elif HL.voc1_has_higher_hue_peak:
                    oil_com = self.cropper.cA_center
                    vinegar_com = self.cropper.cB_center
                else:
                    print("[IndexFinder] : Same hue peak location --> deciding via mean")
                    if HL.voc0_has_higher_hue_mean:
                        oil_com = self.cropper.cB_center
                        vinegar_com = self.cropper.cA_center
                    elif HL.voc1_has_higher_hue_mean:
                        oil_com = self.cropper.cA_center
                        vinegar_com = self.cropper.cB_center

            self.coms_dict["oil"] = oil_com
            self.coms_dict["vinegar"] = vinegar_com

            self.quadrant_dict["oil"] = self.cropper.locate(self.coms_dict["oil"])
            self.quadrant_dict["vinegar"] = self.cropper.locate(self.coms_dict["vinegar"])


            # Finalize response
            response.idx = self.quadrant_dict[target_spice]
            if response.idx in [0,1,2,3]:
                response.success = True
                
        print("[IndexFinder] : Response sent")
        return response
    
    def load_params(self):

        self.debug = rospy.get_param("index_finder/debug")

        self.use_ocr = True

        self.bridge = CvBridge()

        #self.home = rospy.get_param("index_finder/HOME")
        #self.voc0_img_path = self.home +'temp_data/last_voc0.png' # It seems as if absolute path is needed
        #self.voc1_img_path = self.home +'temp_data/last_voc1.png'

        self.quadrant_dict = {} # Which contour center is in which quadrant
        self.coms_dict = {} # Center of mass (com) for each blob in bottles mask 

        self.cropper = None
        self.cropper_is_initialized = False
        self.vocA_ocr_localizer = None # Vinegar-Oil-Color img A
        self.vocB_ocr_localizer = None # Vinegar-Oil-Color img B

        self.all_mask = None
        self.min_pixel_count_valid_mask = 10
       
        sim = rospy.get_param("spice_up_coordinator/in_simulation_mode")
        if sim: 
            self.color_topic_name = "/camera/color/image_raw"
            self.depth_topic_name = "/camera/aligned_depth_to_color/image_raw"
            self.cam_info_topic_name = "/camera/color/camera_info"
        else: # TODO
            self.color_topic_name = "/dynaarm_REALSENSE/color/image_raw"
            self.depth_topic_name = "/dynaarm_REALSENSE/aligned_depth_to_color/image_raw"
            self.cam_info_topic_name = "/dynaarm_REALSENSE/aligned_depth_to_color/camera_info"

    # Utils -----------------------------------------------------------

    def ros_to_cv2(self, frame: Image, desired_encoding="bgr8"):
        return self.bridge.imgmsg_to_cv2(frame, desired_encoding=desired_encoding)

    def shutdown_cb(self,signal):
        if signal.data:
            print("[IndexFinder] : Shutting down")
            rospy.signal_shutdown("Job done")
    # -----------------------------------------------------------------

if __name__ == '__main__':
    rospy.init_node('index_finder_server')
    server = IndexFinder()
    rospy.spin()
