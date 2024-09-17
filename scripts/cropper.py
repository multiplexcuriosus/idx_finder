import numpy as np
import cv2
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

class Cropper:
    def __init__(self,og_color,og_mask,five_contours_found) -> None:
        
        # Init params
        self.init()

        if self.debug:
            print("og_color.shape: "+str(og_color.shape))
            print("og_mask.shape: "+str(og_mask.shape))

        if og_color is None:
            print("Cropper: ERROR: color img is None")
            return
        
        if og_mask is None:
            print("Cropper: ERROR: depth img  is None")
            return

         # Create all-bottles-mask
        all_bottles_mask = None
        if five_contours_found:
            print("[IDXServer.Croppper] : Using 5-contour-approach")
            all_bottles_mask = self.get_all_mask(og_mask)
        else:
            print("[IDXServer.Croppper] : Using thresh-approach")
            og_color_hsv = cv2.cvtColor(og_color, cv2.COLOR_RGB2HSV)
            all_range = np.array([[0, 0, 200], [180, 255, 255]], dtype=np.uint16)
            all_bottles_mask = cv2.inRange(og_color_hsv, all_range[0], all_range[1])
            all_bottles_mask = cv2.bitwise_and(all_bottles_mask, all_bottles_mask, mask=og_mask)

       
        self.all_bottles_mask = self.clean_mask(all_bottles_mask)

        # All color
        all_bottles_color = cv2.bitwise_and(og_color, og_color, mask=self.all_bottles_mask)

        # Blob detection
        self.blobs_bgr = cv2.cvtColor(self.all_bottles_mask, cv2.COLOR_GRAY2BGR)
        contours, hier = cv2.findContours(self.all_bottles_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        # Abort if no contours found        
        if len(contours) == 0:
            print("[IDXServer.Croppper] : No contours found! Aborting")
            return


        # Contour processing
        cntsSorted = sorted(contours, key=lambda x: -cv2.contourArea(x))

        c0 = cntsSorted[0] # largest contour
        c0_center = self.get_center_of_countour(c0)

        c1 = cntsSorted[1] # second largest contour
        c1_center = self.get_center_of_countour(c1)

        c2 = cntsSorted[2] # third largest contour
        c2_center = self.get_center_of_countour(c2)

        c3 = cntsSorted[3] # fouth largest contour
        c3_center = self.get_center_of_countour(c3)

        self.centroid = (int(np.average([c0_center[0],c1_center[0],c2_center[0],c3_center[0]])),
                         int(np.average([c0_center[1],c1_center[1],c2_center[1],c3_center[1]])))

        self.spice0_loc_idx = self.locate(c0_center)
        self.spice1_loc_idx = self.locate(c1_center)
        self.spice2_loc_idx = self.locate(c2_center)
        self.spice3_loc_idx = self.locate(c3_center)

        # Key: spice, value: location idx
        spiceidx_to_locidx = {0: self.spice0_loc_idx,
                              1: self.spice1_loc_idx,
                              2: self.spice2_loc_idx,
                              3: self.spice3_loc_idx}
        print("spiceidx_to_locidx: "+str(spiceidx_to_locidx))

        locidx_to_spiceidx = {self.spice0_loc_idx : 0,
                              self.spice1_loc_idx : 1,
                              self.spice2_loc_idx : 2,
                              self.spice3_loc_idx : 3}
        
        # Key: spice, value: com
        spiceidx_to_com = {0: c0_center,
                           1: c1_center,
                           2: c2_center,
                           3: c3_center}
        
        # Cropped color imgs for each spice
        spice_img_dict = {0: self.get_color_frame_cropped(cntsSorted,og_color,0),
                          1: self.get_color_frame_cropped(cntsSorted,og_color,1),
                          2: self.get_color_frame_cropped(cntsSorted,og_color,2),
                          3: self.get_color_frame_cropped(cntsSorted,og_color,3)}

        # Brightness histogram expectation value for each spice
        bhist_evs = [(self.get_brightness_exp_val(spice_img_dict,0),0),
                     (self.get_brightness_exp_val(spice_img_dict,1),1),
                     (self.get_brightness_exp_val(spice_img_dict,2),2),
                     (self.get_brightness_exp_val(spice_img_dict,3),3)]

        print("bhist_evs: "+str(bhist_evs))

        bhist_evs_sorted = sorted(bhist_evs, key=lambda tu: tu[0])
        print("bhist_evs_sorted: "+str(bhist_evs_sorted))

        # Draw conclusions
        pepper_loc_idx = spiceidx_to_locidx[bhist_evs_sorted[0][1]]
        salt_loc_idx = spiceidx_to_locidx[bhist_evs_sorted[1][1]]

        self.quadrant_dict["pepper"] = pepper_loc_idx
        self.quadrant_dict["salt"] = salt_loc_idx

        self.status = "success"

        # PREPARE NEXT STAGE -------------------------------------------
        sp_loc_idxs = [pepper_loc_idx,salt_loc_idx]
        all_loc_idxs = set([0,1,2,3])
        ov_loc_idxs = list(all_loc_idxs.difference(sp_loc_idxs))
        print("ov loc idxs: "+str(ov_loc_idxs))
        
        vocA_loc_idx = ov_loc_idxs[0]
        vocB_loc_idx = ov_loc_idxs[1]
        vocA_idx = locidx_to_spiceidx[vocA_loc_idx]
        vocB_idx = locidx_to_spiceidx[vocB_loc_idx]

        # Prepare img for ocr
        self.vocA_img = spice_img_dict[vocA_idx] 
        self.vocB_img = spice_img_dict[vocB_idx]

        cv2.imwrite(self.vocA_img_path,self.vocA_img)
        cv2.imwrite(self.vocB_img_path,self.vocB_img)

        # Prepare contour centers for ocr
        self.cA_center = spiceidx_to_com[vocA_idx]
        self.cB_center = spiceidx_to_com[vocB_idx]
        
        if self.debug:
            hist_img = self.createHistIMG(self.bhist_dict[0],
                                          self.bhist_dict[1],
                                          self.bhist_dict[2],
                                          self.bhist_dict[3],
                                          self.spice_bhist_peak_dict[0] ,
                                          self.spice_bhist_peak_dict[1] ,
                                          self.spice_bhist_peak_dict[2] ,
                                          self.spice_bhist_peak_dict[3] ,
                                          self.spice_bhist_ev_dict[0],
                                          self.spice_bhist_ev_dict[1],
                                          self.spice_bhist_ev_dict[2],
                                          self.spice_bhist_ev_dict[3],)
            img_path = self.debug_imgs_path + 'hist_img.png'
            cv2.imwrite(img_path,hist_img)
        

        # debug:
        if self.debug:
            img_path = self.debug_imgs_path + 'og_color.png'
            cv2.imwrite(img_path,og_color)

            img_path = self.debug_imgs_path + 'og_mask.png'
            cv2.imwrite(img_path,og_mask)

            img_path = self.debug_imgs_path + 'all_mask_clean.png'
            cv2.imwrite(img_path,self.all_bottles_mask)

            img_path = self.debug_imgs_path + 'all_color.png'
            cv2.imwrite(img_path,all_bottles_color)


    def get_color_frame_cropped(self,contours,og_color,idx):
        contour = contours[idx]
        spice_mask = self.get_cont_mask(contour)
        spice_mask_bbox = self.crop_to_contour(contour,spice_mask)
        spice_color_bbox = self.crop_to_contour(contour,og_color)
        print("spice_mask_bbox shape: "+str(spice_mask_bbox.shape))
        print("spice_color_bbox shape: "+str(spice_color_bbox.shape))


        spice_mask_tight,og_color_tight = self.tight_crop(spice_mask_bbox,spice_color_bbox)
        #print("spice mask tight dtype: "+str(spice_mask_tight.dtype))
        #print("og_color_tight tight dtype: "+str(og_color_tight.dtype))
        print("spice mask tight shape: "+str(spice_mask_tight.shape))
        print("og_color_tight shape: "+str(og_color_tight.shape))
        spice_col_tight = cv2.bitwise_and(og_color_tight, og_color_tight, mask=spice_mask_tight)

        if self.debug:
            img_path = self.debug_imgs_path + 'spice'+str(idx)+'_tight_mask.png'
            cv2.imwrite(img_path,spice_mask_tight)
            img_path = self.debug_imgs_path + 'spice'+str(idx)+'_col_tight.png'
            cv2.imwrite(img_path,spice_col_tight)
            #img_path = self.debug_imgs_path + 'spice'+str(idx)+'_col_tight_cropped.png'
            #cv2.imwrite(img_path,spice_col_tight_cropped)
        return spice_col_tight
    
    def tight_crop(self,spice_mask_bbox_binary,spice_color_bbox):
        #print("shape: "+str(img.shape))
        spice_mask_bbox_bgr = cv2.cvtColor(spice_mask_bbox_binary,cv2.COLOR_GRAY2BGR)

        h,w,_ = spice_mask_bbox_bgr.shape
        row_start = int(h*0.0)
        row_end = int(h*1.0)
        col_start = int(w*0.0)
        col_end = int(w*1.0)
        cropped_mask = spice_mask_bbox_bgr[row_start:row_end,col_start:col_end]
        cropped_col = spice_color_bbox[row_start:row_end,col_start:col_end]
        #cv2.imshow("cropped",cropped_image)
        #cv2.waitKey(0)
        
        k = 5
        cleaning_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
        cropped_mask_clean = cv2.morphologyEx(cropped_mask, cv2.MORPH_CLOSE, cleaning_kernel)

        #cv2.imshow("cropped_clean",cropped_clean)
        #cv2.waitKey(0)

        k = 15
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k,k))
        cropped_mask_clean_eroded = cv2.erode(cropped_mask_clean, erosion_kernel)

        cropped_mask_clean_eroded_gray = cv2.cvtColor(cropped_mask_clean_eroded,cv2.COLOR_BGR2GRAY)
        print('cropped_mask_clean_eroded_gray shape: '+str(cropped_mask_clean_eroded_gray.shape))
        
        #th,eroded_binary = cv2.threshold(eroded_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #print('eroded_binary shape: '+str(eroded_binary.shape))
        #cv2.imshow("eroded",eroded)
        #cv2.waitKey(0)
        #

        return cropped_mask_clean_eroded_gray.astype(np.uint8),cropped_col

    def get_brightness_exp_val(self,spice_img_dict,idx):
        spice_bhist = self.get_brightness_histogram(spice_img_dict[idx])
        L = len(spice_bhist)
        start = 1
        V = np.arange(start,L)
        spice_bhist_peak = self.get_peak_of_histogram(spice_bhist)
        spice_bhist_ev = self.expected_value(V,spice_bhist[start:])

        self.bhist_dict[idx] = spice_bhist
        self.spice_bhist_ev_dict[idx] = spice_bhist_ev
        self.spice_bhist_peak_dict[idx] = spice_bhist_peak
    
        return spice_bhist_ev

    def expected_value(self,values, weights):
        values = np.asarray(values)
        weights = np.asarray(weights)
        return (np.dot(values,weights)) / weights.sum()

    def createHistIMG(self,h0,h1,h2,h3,p0,p1,p2,p3,m0,m1,m2,m3):
        fig, ax = plt.subplots(1)
        
        ax.plot(h0, color='blue', label="s0")
        ax.plot(h1, color='red', label="s1")
        ax.plot(h2, color='green', label="s2")
        ax.plot(h3, color='orange', label="s3")
        
        ax.plot(p0[0],p0[1],'o',color='blue')
        ax.plot(p1[0],p1[1],'o',color='red')
        ax.plot(p2[0],p2[1],'o',color='green')
        ax.plot(p3[0],p3[1],'o',color='orange')

        plt.axvline(x=m0, color='blue', linestyle='--')
        plt.axvline(x=m1, color='red', linestyle='--')
        plt.axvline(x=m2, color='green', linestyle='--')
        plt.axvline(x=m3, color='orange', linestyle='--')

        ax.legend(loc="upper right")
        #plt.show()
        fig.canvas.draw()
        hist = np.array(fig.canvas.renderer.buffer_rgba())
        hist_img = cv2.cvtColor(hist, cv2.COLOR_RGBA2RGB)
        return hist_img


    def init(self):

        self.status = "FAIL"
        self.debug = True

        self.spice_bhist_ev_dict = {}
        self.spice_bhist_peak_dict = {}
        self.bhist_dict = {}

        self.quadrant_dict = {}

        self.ov_loc_idxs = None # Oil vinegar idx

        # Debug imgs
        self.hist_img = None
        self.spice0_col_img = None
        self.spice1_col_img = None

        self.vocA_img_path = '/home/carmen/ros/catkin_ws/src/idx_finder/scripts/vocA.png'
        self.vocB_img_path = '/home/carmen/ros/catkin_ws/src/idx_finder/scripts/vocB.png'

        self.debug_imgs_path='/home/carmen/ros/catkin_ws/src/idx_finder/scripts/debug_imgs/'


    def remove_holes(self,mask_cv):
        orignal_mask_bgr = cv2.cvtColor(mask_cv,cv2.COLOR_GRAY2BGR)
        contours, hier = cv2.findContours(mask_cv,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        cntsSorted = sorted(contours, key=lambda x: -cv2.contourArea(x))
        c0 = cntsSorted[0] # largest contour
        cv2.drawContours(orignal_mask_bgr,[c0],0,(0,255,0),-1)
        og_mask_hsv = cv2.cvtColor(orignal_mask_bgr, cv2.COLOR_BGR2HSV)
        mask_no_holes =  cv2.inRange(og_mask_hsv,np.array([50,0,0]),np.array([70,255,255]))
        
        return mask_no_holes

    def get_all_mask(self,mask):
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

    def clean_mask(self,mask):
        # Apply structuring element
        k = 10
        cleaning_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
        morph = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cleaning_kernel)
        '''
        k = 10
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k,k))
        morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        '''
        #morph = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return morph

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

    def get_cont_mask(self,cont):
        vom_bgr = cv2.cvtColor(self.all_bottles_mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vom_bgr,[cont],0,(0,255,0),-1)
        vom0_hsv = cv2.cvtColor(vom_bgr, cv2.COLOR_BGR2HSV)
        return cv2.inRange(vom0_hsv,np.array([50,0,0]),np.array([70,255,255]))
        
    def locate(self,com):
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
