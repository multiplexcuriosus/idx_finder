import numpy as np
import cv2
import rospy
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

class Cropper:
    def __init__(self,og_color,og_mask_raw,five_contours_found,debug) -> None:
        '''
        The Cropper recieves a color-img, a mask and the information whether or not the create_mask_server found 5 contours in the mask.
        Then he does the following:
        1.) If five_contours_found==True, simple mask operations are done to get a mask which only contains the four bottles --> all_bottles_mask
            If five contours_found==False, a brightness threshold is applied to the color img and various cleaning operations are performed
            to create a mask with only the bottle masks --> all_bottles_mask
        2.) Blob detection is performed on the all_bottles_mask, and based on the center of masses of the blobs the bottles are separated into the four quadrants.
        3.) Based on the bounding box of each blob a new image is created for each spice bottle (except black if thresholding was done). 
            The cropped masks are then further eroded to remove noisy image patches. 
            Through a bitwise-and between the color image and the cropped masks we obtain: Cropped eroded color images for each bottle --> spice_col_tight
        4.) The 3-4 spice_col_tight images are transformed into HSV space and a brightness histogram is generated. 
        The four spice_col_tight images will have four different distributions in this histogram. 
        If tresholding was done, then the pepper-location-index corresponds to the location-index of the missing blob (which was too dark to survive the threshold).
        If not then it is now assumed that the pepper-location-index corresponds to the spice-location-index of the spice_col_tight image with the lowest brightness-mean.
        Analogously, the salt-location-index corresponds to the the spice-location-index of the spice_col_tight image with the second lowest brightness-mean.
        5.) The salt- and pepper-location indices are saved and also the spice_col_tight imgs for vinegar and oil.
        '''

        # Init params
        self.init_params()
        self.debug = debug

        self.check_if_none_and_save(og_color,og_mask_raw)
        
        # Clean og mask
        og_mask_clean = self.clean_mask(mask=og_mask_raw,k=10)
        if self.debug:
            img_path = self.debug_imgs_path + 'og_mask_clean.png'
            cv2.imwrite(img_path,og_mask_clean)

        # Create mask containing just the bottles
        self.all_bottles_mask = self.get_all_bottles_mask(og_color,og_mask_clean,five_contours_found)

        # Create color img containing just the bottles
        all_bottles_color = cv2.bitwise_and(og_color, og_color, mask=self.all_bottles_mask)

        # Blob detection
        self.blobs_bgr = cv2.cvtColor(self.all_bottles_mask, cv2.COLOR_GRAY2BGR)
        contours, hier = cv2.findContours(self.all_bottles_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        # Abort if at this point there are less than 3 contours
        num_cont = len(contours)    
        print("Found "+str(num_cont)+" holes")      
        if num_cont < 3:
            print("[IDXServer.Croppper] : Less than three holes found! Aborting")
            return
        
        four_holes_found = num_cont == 4
        if four_holes_found:
            print("[IDXServer.Croppper] : Using 4-hole-approach")
        else:
            print("[IDXServer.Croppper] : Using thresh-approach")

        # Use blob detection to get tightly cropped color images of the four bottles (3 if thresholding was used)
        self.create_tight_spice_images(og_color,contours,four_holes_found)

        # Create brightness histogram for each image
        bhist_evs_sorted = self.get_brightness_histogram_means(four_holes_found)

        # Draw conclusions
        if four_holes_found: # Localize salt and pepper based on brightness histogram
            pepper_loc_idx = self.spiceidx_to_locidx[bhist_evs_sorted[0][1]]
            salt_loc_idx = self.spiceidx_to_locidx[bhist_evs_sorted[1][1]]
        else: # Localize salt and pepper by assuming that the pepper  was "thresholded away" and salt is the smallest blob
            pepper_loc_idx = self.spice3_loc_idx
            salt_loc_idx = self.spice2_loc_idx

        self.quadrant_dict["pepper"] = pepper_loc_idx
        self.quadrant_dict["salt"] = salt_loc_idx

        self.status = "success"

        # PREPARE OIL-VINEGAR-CLASSIFICATION -------------------------------------------
        salt_pepper_loc_idxs = [pepper_loc_idx,salt_loc_idx]
        oil_vinegar_loc_idxs = list(self.all_loc_idxs.difference(salt_pepper_loc_idxs))
        print("[IDXServer.Croppper] : oil vinegar loc idxs: "+str(oil_vinegar_loc_idxs))
        
        vocA_loc_idx = oil_vinegar_loc_idxs[0]
        vocB_loc_idx = oil_vinegar_loc_idxs[1]
        vocA_idx = self.locidx_to_spiceidx[vocA_loc_idx]
        vocB_idx = self.locidx_to_spiceidx[vocB_loc_idx]

        # Prepare img for ocr
        self.vocA_img = self.spice_img_dict[vocA_idx] 
        self.vocB_img = self.spice_img_dict[vocB_idx]

        cv2.imwrite(self.vocA_img_path,self.vocA_img)
        cv2.imwrite(self.vocB_img_path,self.vocB_img)

        # Prepare contour centers for ocr
        self.cA_center = self.spiceidx_to_com[vocA_idx]
        self.cB_center = self.spiceidx_to_com[vocB_idx]

        # debug:
        if self.debug:
            print("[IDXServer.Croppper] : saving debug imgs")

            img_path = self.debug_imgs_path + 'all_color.png'
            cv2.imwrite(img_path,all_bottles_color)

            hist_img = self.createHistIMG(self.bhist_dict,
                                          self.spice_bhist_peak_dict ,
                                          self.spice_bhist_ev_dict)
            img_path = self.debug_imgs_path + 'brighness_hist_img.png'
            cv2.imwrite(img_path,hist_img)

    def init_params(self):

        self.status = "FAIL"

        self.home = rospy.get_param("index_finder/HOME")
        self.debug_imgs_path = self.home + 'debug_imgs/'
        self.vocA_img_path = self.home+'temp_data/vocA.png'
        self.vocB_img_path = self.home+'temp_data/vocB.png'

        self.all_loc_idxs = set([0,1,2,3])

        # Histogram visualization
        self.spice_bhist_ev_dict = {} 
        self.spice_bhist_peak_dict = {}

        self.bhist_dict = {}
        self.quadrant_dict = {}
        self.spice_img_dict = {}
        self.spice0_loc_idx = {}
        self.spiceidx_to_com = {}
        self.centroid = None

        self.ov_loc_idxs = None # Oil vinegar idx

        # Debug imgs
        self.hist_img = None
        self.spice0_col_img = None
        self.spice1_col_img = None

    def get_brightness_histogram_means(self,four_holes_found):
        # Brightness histogram expectation value for each spice
        bhist_evs = [(self.get_brightness_exp_val(self.spice_img_dict,0),0),
                     (self.get_brightness_exp_val(self.spice_img_dict,1),1),
                     (self.get_brightness_exp_val(self.spice_img_dict,2),2)]
        if four_holes_found:
            bhist_evs.append((self.get_brightness_exp_val(self.spice_img_dict,3),3))

        bhist_evs_sorted = sorted(bhist_evs, key=lambda tu: tu[0])
        if self.debug:
            print("[IDXServer.Croppper] : bhist_evs: "+str(bhist_evs))
            print("[IDXServer.Croppper] : bhist_evs_sorted: "+str(bhist_evs_sorted))
        return bhist_evs_sorted

    def create_tight_spice_images(self,og_color,contours,four_holes_found):
        
        # Contour processing
        cntsSorted = sorted(contours, key=lambda x: -cv2.contourArea(x))

        c0 = cntsSorted[0] # largest contour
        c0_center = self.get_center_of_countour(c0)

        c1 = cntsSorted[1] # second largest contour
        c1_center = self.get_center_of_countour(c1)

        c2 = cntsSorted[2] # third largest contour
        c2_center = self.get_center_of_countour(c2)

        if four_holes_found:
            c3 = cntsSorted[3] # fourth largest contour
            c3_center = self.get_center_of_countour(c3)

            self.centroid = (int(np.average([c0_center[0],c1_center[0],c2_center[0],c3_center[0]])),
                             int(np.average([c0_center[1],c1_center[1],c2_center[1],c3_center[1]])))
        else:
            self.centroid = (int(np.average([c0_center[0],c1_center[0],c2_center[0]])),
                             int(np.average([c0_center[1],c1_center[1],c2_center[1]])))

        self.spice0_loc_idx = self.get_spice_location_index_by_com(c0_center)
        self.spice1_loc_idx = self.get_spice_location_index_by_com(c1_center)
        self.spice2_loc_idx = self.get_spice_location_index_by_com(c2_center)

        if four_holes_found:
            self.spice3_loc_idx = self.get_spice_location_index_by_com(c3_center)
        else:
            self.spice3_loc_idx = list(self.all_loc_idxs.difference([self.spice0_loc_idx,
                                                                     self.spice1_loc_idx,
                                                                     self.spice2_loc_idx]))[0]

        # Key: spice, value: location idx
        self.spiceidx_to_locidx = {0: self.spice0_loc_idx,
                                    1: self.spice1_loc_idx,
                                    2: self.spice2_loc_idx,
                                    3: self.spice3_loc_idx}
        print("[IDXServer.Croppper] : spiceidx_to_locidx: "+str(self.spiceidx_to_locidx))

        self.locidx_to_spiceidx = {self.spice0_loc_idx : 0,
                              self.spice1_loc_idx : 1,
                              self.spice2_loc_idx : 2,
                              self.spice3_loc_idx : 3}
        
        # Key: spice, value: blob center of mass (com)
        self.spiceidx_to_com = {0: c0_center,
                           1: c1_center,
                           2: c2_center}
        if four_holes_found:
            self.spiceidx_to_com[3] = c3_center
        
        # Cropped color imgs for each spice
        self.spice_img_dict = {0: self.get_color_img_cropped_to_contour(cntsSorted,og_color,0),
                          1: self.get_color_img_cropped_to_contour(cntsSorted,og_color,1),
                          2: self.get_color_img_cropped_to_contour(cntsSorted,og_color,2)}
        if four_holes_found:
            self.spice_img_dict[3] = self.get_color_img_cropped_to_contour(cntsSorted,og_color,3)

    def get_all_bottles_mask(self,og_color,og_mask_clean,five_contours_found):
        all_bottles_mask = None
        if five_contours_found: 
            all_bottles_mask = self.get_mask_of_holes(og_mask_clean)
        else: 
            og_mask_no_holes = self.remove_holes(og_mask_clean)
            if self.debug:
                img_path = self.debug_imgs_path + 'idx_finder_og_mask_no_holes.png'
                cv2.imwrite(img_path,og_mask_no_holes)
            brightness_thresh = rospy.get_param("index_finder/brightness_threshold")
            og_color_hsv = cv2.cvtColor(og_color, cv2.COLOR_RGB2HSV)
            all_range = np.array([[0, 0, brightness_thresh], [180, 255, 255]], dtype=np.uint16)
            all_bottles_mask = cv2.inRange(og_color_hsv, all_range[0], all_range[1])
            if self.debug:
                img_path = self.debug_imgs_path + 'all_bottles_mask_inrange.png'
                cv2.imwrite(img_path,all_bottles_mask)
            all_bottles_mask = cv2.bitwise_and(all_bottles_mask, all_bottles_mask, mask=og_mask_no_holes)
            if self.debug:
                img_path = self.debug_imgs_path + 'all_bottles_mask_bitwise_and.png'
                cv2.imwrite(img_path,all_bottles_mask)

            # Cut away top 20% of all_bottles_mask to remove potential glare from top shelf plane
            h,w = all_bottles_mask.shape
            border = int(0.2*h)
            all_bottles_mask[0:border,:] = 0
            if self.debug:
                img_path = self.debug_imgs_path + 'all_bottles_mask_top_20percent_cut_off.png'
                cv2.imwrite(img_path,all_bottles_mask)
            
        all_bottles_mask = self.clean_mask(all_bottles_mask)
        if self.debug:
            img_path = self.debug_imgs_path + 'all_bottles_mask_clean.png'
            cv2.imwrite(img_path,all_bottles_mask)
        return all_bottles_mask

    def check_if_none_and_save(self,og_color,og_mask_raw):
        if og_color is None:
            print("Cropper: ERROR: color img is None")
            return
        if self.debug:
            img_path = self.debug_imgs_path + 'og_color.png'
            cv2.imwrite(img_path,og_color)
        
        if og_mask_raw is None:
            print("Cropper: ERROR: mask  is None")
            return
        if self.debug:
            img_path = self.debug_imgs_path + 'og_mask_raw.png'
            cv2.imwrite(img_path,og_mask_raw)

    def get_color_img_cropped_to_contour(self,contours,og_color,idx):
        contour = contours[idx]
        spice_mask = self.get_mask_of_contour(contour)
        spice_mask_bbox = self.crop_to_contour(contour,spice_mask)
        spice_color_bbox = self.crop_to_contour(contour,og_color)
        if self.debug:
            img_path = self.debug_imgs_path +'spice'+str(idx)+'_mask.png'
            cv2.imwrite(img_path,spice_mask)

            img_path = self.debug_imgs_path +'spice'+str(idx)+'_mask_bbox.png'
            cv2.imwrite(img_path,spice_mask_bbox)

            img_path = self.debug_imgs_path +'spice'+str(idx)+'_color_bbox.png'
            cv2.imwrite(img_path,spice_color_bbox)

        # Crop and erode
        spice_mask_tight,og_color_tight = self.crop_and_erode(spice_mask_bbox,spice_color_bbox)

        spice_col_tight = cv2.bitwise_and(og_color_tight, og_color_tight, mask=spice_mask_tight)

        if self.debug:
            img_path = self.debug_imgs_path + 'spice'+str(idx)+'_tight_mask.png'
            cv2.imwrite(img_path,spice_mask_tight)
            img_path = self.debug_imgs_path + 'spice'+str(idx)+'_col_tight.png'
            cv2.imwrite(img_path,spice_col_tight)
        return spice_col_tight
    
    def crop_and_erode(self,spice_mask_bbox_binary,spice_color_bbox):
        spice_mask_bbox_bgr = cv2.cvtColor(spice_mask_bbox_binary,cv2.COLOR_GRAY2BGR)

        # Cut away top 20% TODO: Replace with projecting top shelf plate
        h,w,_ = spice_mask_bbox_bgr.shape
        row_start = int(h*0.2)
        row_end = int(h*1.0)
        col_start = int(w*0.0)
        col_end = int(w*1.0)
        cropped_mask = spice_mask_bbox_bgr[row_start:row_end,col_start:col_end]
        cropped_col = spice_color_bbox[row_start:row_end,col_start:col_end]
        
        k = 5
        cleaning_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
        cropped_mask_clean = cv2.morphologyEx(cropped_mask, cv2.MORPH_OPEN, cleaning_kernel)

        k = 10
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
        cropped_mask_clean_eroded = cv2.erode(cropped_mask_clean, erosion_kernel)

        cropped_mask_clean_eroded_gray = cv2.cvtColor(cropped_mask_clean_eroded,cv2.COLOR_BGR2GRAY)

        return cropped_mask_clean_eroded_gray.astype(np.uint8),cropped_col

    def get_brightness_exp_val(self,spice_img_dict,idx):
        spice_bhist = self.get_brightness_histogram(spice_img_dict[idx])
        L = len(spice_bhist)
        start = 1
        values = np.arange(start,L)
        weights = spice_bhist[start:]
        spice_bhist_peak = self.get_peak_of_histogram(spice_bhist)
        spice_bhist_ev = self.compute_expectation_value(values,weights)

        self.bhist_dict[idx] = spice_bhist
        self.spice_bhist_ev_dict[idx] = spice_bhist_ev
        self.spice_bhist_peak_dict[idx] = spice_bhist_peak
    
        return spice_bhist_ev

    def compute_expectation_value(self,values, weights):
        values = np.asarray(values)
        weights = np.asarray(weights)
        return (np.dot(values,weights)) / weights.sum()

    def createHistIMG(self,bhists,peaks,evs):

        h0 = bhists[0] # histograms
        h1 = bhists[1] 
        h2 = bhists[2]
        p0 = peaks[0] # peaks
        p1 = peaks[1]
        p2 = peaks[2]
        m0 = evs[0] # means
        m1 = evs[1]
        m2 = evs[2]

        fig, ax = plt.subplots(1)
        
        ax.plot(h0, color='blue', label="sct0")
        ax.plot(h1, color='red', label="sct1")
        ax.plot(h2, color='green', label="sct2")

        ax.plot(p0[0],p0[1],'o',color='blue')
        ax.plot(p1[0],p1[1],'o',color='red')
        ax.plot(p2[0],p2[1],'o',color='green')

        plt.axvline(x=m0, color='blue', linestyle='--')
        plt.axvline(x=m1, color='red', linestyle='--')
        plt.axvline(x=m2, color='green', linestyle='--')


        if 3 in bhists and 3 in peaks and 3 in evs:
            h3 = bhists[3]
            ax.plot(h3, color='magenta', label="sct3")
            p3 = peaks[3]
            ax.plot(p3[0],p3[1],'o',color='magenta')
            m3 = evs[3]
            plt.axvline(x=m3, color='magenta', linestyle='--')

        plt.xlabel("Brightness")
        plt.ylabel("N pixels")
        plt.yticks([]) 
        ax.legend(loc="upper left")
        #plt.show()
        fig.canvas.draw()   
        hist = np.array(fig.canvas.renderer.buffer_rgba())
        hist_img = cv2.cvtColor(hist, cv2.COLOR_RGBA2RGB)
        return hist_img
 
    def remove_holes(self,mask_cv):
        orignal_mask_bgr = cv2.cvtColor(mask_cv,cv2.COLOR_GRAY2BGR)
        contours, hier = cv2.findContours(mask_cv,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        cntsSorted = sorted(contours, key=lambda x: -cv2.contourArea(x))
        c0 = cntsSorted[0] # largest contour
        cv2.drawContours(orignal_mask_bgr,[c0],0,(0,255,0),-1)
        og_mask_hsv = cv2.cvtColor(orignal_mask_bgr, cv2.COLOR_BGR2HSV)
        mask_no_holes =  cv2.inRange(og_mask_hsv,np.array([50,0,0]),np.array([70,255,255]))
        
        return mask_no_holes

    def get_mask_of_holes(self,mask):
        # Create mask with no holes
        mask_no_holes = self.remove_holes(mask)
        img_path = self.debug_imgs_path + 'mask_no_holes.png'
        cv2.imwrite(img_path,mask_no_holes)

        # Create inv og mask --> inv of shelf
        og_mask_inv = cv2.bitwise_not(mask)
        img_path = self.debug_imgs_path + 'og_mask_inv.png'
        cv2.imwrite(img_path,og_mask_inv)

        # Use inv mask and no holes mask to create mask where only four bottles are visible
        all_mask = cv2.bitwise_and(mask_no_holes,og_mask_inv)
        img_path = self.debug_imgs_path + 'all_mask.png'
        cv2.imwrite(img_path,all_mask)
        return all_mask

    def clean_mask(self,mask,k=5):
        # Apply structuring element
        cleaning_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
        clean_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cleaning_kernel)

        # The MORPH_OPEN operation can open black holes in the white blobs, which is why we close these again with a MORPH_CLOSE operation
        k = 5
        cleaning_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
        clean_close = cv2.morphologyEx(clean_open, cv2.MORPH_CLOSE, cleaning_kernel)
        
        return clean_close    

    def crop_to_contour(self,cont,col_img):
        x,y,w,h = cv2.boundingRect(cont) # a - angle
        x0 = int(x)
        x1 = int(x+w)
        y0 = int(y)
        y1 = int(y+h)
        return col_img[y0:y1,x0:x1]
    
    def get_center_of_countour(self,cont):
        M = cv2.moments(cont)
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    def get_mask_of_contour(self,cont):
        vom_bgr = cv2.cvtColor(self.all_bottles_mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vom_bgr,[cont],0,(0,255,0),-1)
        vom0_hsv = cv2.cvtColor(vom_bgr, cv2.COLOR_BGR2HSV)
        return cv2.inRange(vom0_hsv,np.array([50,0,0]),np.array([70,255,255]))
        
    def get_spice_location_index_by_com(self,com):
        if com[0] < self.centroid[0] and com[1] < self.centroid[1]:
            return  0
        elif com[0] > self.centroid[0] and com[1] < self.centroid[1]:
            return  1
        elif com[0] < self.centroid[0] and com[1] > self.centroid[1]:
            return  2
        elif com[0] > self.centroid[0] and com[1] > self.centroid[1]:
            return  3
        
    def get_brightness_histogram(self,img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2]
        l = 1
        return cv2.calcHist([v],[0],None,[255],[0,255])[l:]

    def get_peak_of_histogram(self,hist):
        hist = hist[:,0] 
        peaks_x, _ = find_peaks(hist, height=0)
        peaks_y = hist[peaks_x]
        peaks = list(zip(peaks_x,peaks_y))
        peaks_sorted = sorted(peaks, key=lambda p: -p[1])
        if len(peaks_sorted) > 0:
            return peaks_sorted[0]
        else:
            return (-1,-1)
