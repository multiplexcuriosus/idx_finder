import numpy as np
import cv2
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

class Cropper:
    def __init__(self,og_color,og_mask) -> None:
        
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
        #og_mask = cv2.cvtColor(og_mask,cv2.COLOR_BGR2GRAY)
        all_bottles_mask = self.get_all_mask(og_mask)
        self.all_bottles_mask = self.clean_mask(all_bottles_mask)

        # All color
        all_bottles_color = cv2.bitwise_and(og_color, og_color, mask=self.all_bottles_mask)

        # Blob detection
        blobs_bgr = cv2.cvtColor(self.all_bottles_mask, cv2.COLOR_GRAY2BGR)
        contours, hier = cv2.findContours(self.all_bottles_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        # Count submasks in shelf mask
        num_cont = len(contours)
        print("[IDXServer.Croppper] : Found %s countours" % num_cont)
    
        if num_cont < 3:
            print("[IDXServer.Croppper] : Switching to thresh approach...")
            # ...TODO
            return 

        # Assume 4 countours were found
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

        self.spice0_idx = self.locate(c0_center)
        self.spice1_idx = self.locate(c1_center)
        self.spice2_idx = self.locate(c2_center)
        self.spice3_idx = self.locate(c3_center)

        spiceidx_to_locidx = {0: self.spice0_idx,
                             1: self.spice1_idx,
                             2: self.spice2_idx,
                             3: self.spice3_idx}

        print("spice0_idx: "+str(self.spice0_idx))
        print("spice1_idx: "+str(self.spice1_idx))
        print("spice2_idx: "+str(self.spice2_idx))
        print("spice3_idx: "+str(self.spice3_idx))

        #self.occupied_idxs = list([self.spice0_col_idx,self.spice1_col_idx,self.spice2_col_idx])

        # Mask detection
        spice0_mask = self.get_cont_mask(c0)
        spice1_mask = self.get_cont_mask(c1)
        spice2_mask = self.get_cont_mask(c2)
        spice3_mask = self.get_cont_mask(c3)

        # Get color_img
        spice0_col = cv2.bitwise_and(og_color, og_color, mask=spice0_mask)
        spice1_col = cv2.bitwise_and(og_color, og_color, mask=spice1_mask)
        spice2_col = cv2.bitwise_and(og_color, og_color, mask=spice2_mask)
        spice3_col = cv2.bitwise_and(og_color, og_color, mask=spice3_mask)

        # Get color cropped
        spice0_col_cropped = self.get_color_cropped(c0,spice0_col)
        spice1_col_cropped = self.get_color_cropped(c1,spice1_col)
        spice2_col_cropped = self.get_color_cropped(c2,spice2_col)
        spice3_col_cropped = self.get_color_cropped(c3,spice3_col)

        voc_img_dict = {0: spice0_col_cropped,
                        1: spice1_col_cropped,
                        2: spice2_col_cropped,
                        3: spice3_col_cropped}

        self.spice0_col_img = spice0_col_cropped
        self.spice1_col_img = spice1_col_cropped
        self.spice2_col_img = spice2_col_cropped
        self.spice3_col_img = spice3_col_cropped

        # Sort by brightness mean
        spice0_bhist = self.get_brightness_histogram(self.spice0_col_img)
        spice1_bhist = self.get_brightness_histogram(self.spice1_col_img)
        spice2_bhist = self.get_brightness_histogram(self.spice2_col_img)
        spice3_bhist = self.get_brightness_histogram(self.spice3_col_img)

        spice0_bhist_peak = self.get_peak_of_histogram(spice0_bhist)
        spice1_bhist_peak = self.get_peak_of_histogram(spice1_bhist)
        spice2_bhist_peak = self.get_peak_of_histogram(spice2_bhist)
        spice3_bhist_peak = self.get_peak_of_histogram(spice3_bhist)

        L = len(spice0_bhist)
        V = np.arange(0,L)
        spice0_bhist_ev = self.expected_value(V,spice0_bhist)
        spice1_bhist_ev = self.expected_value(V,spice1_bhist)
        spice2_bhist_ev = self.expected_value(V,spice2_bhist)
        spice3_bhist_ev = self.expected_value(V,spice3_bhist)

       
        bhist_evs = [(spice0_bhist_ev,0),
                     (spice1_bhist_ev,1),
                     (spice2_bhist_ev,2),
                     (spice3_bhist_ev,3)]

        bhist_evs_sorted = sorted(bhist_evs, key=lambda tu: tu[0])
       
        print("bhist_evs_sorted: "+str(bhist_evs_sorted))
        print("bhist_evs_sorted[0]: "+str(bhist_evs_sorted[0]))
        print("bhist_evs_sorted[1]: "+str(bhist_evs_sorted[1]))

        # Draw conclusions
        pepper_idx = spiceidx_to_locidx[bhist_evs_sorted[0][1]]
        salt_idx = spiceidx_to_locidx[bhist_evs_sorted[1][1]]

        self.quadrant_dict["pepper"] = pepper_idx
        self.quadrant_dict["salt"] = salt_idx
        print("pepper idx: "+str(pepper_idx))
        print("salt_idx: "+str(salt_idx))

        self.status = "success"

        # PREPARE NEXT STAGE -------------------------------------------
        sp_idxs = [pepper_idx,salt_idx]
        all_idxs = set([0,1,2,3])
        ov_idxs = list(all_idxs.difference(sp_idxs))
        #print("ov idxs: "+str(ov_idxs))
        
        vocA_idx = ov_idxs[0]
        vocB_idx = ov_idxs[1]

        # Prepare img for ocr
        self.vocA_img = voc_img_dict[vocA_idx] 
        self.vocB_img = voc_img_dict[vocB_idx]

        cv2.imwrite(self.vocA_img_path,self.vocA_img)
        cv2.imwrite(self.vocB_img_path,self.vocB_img)

        # Prepare contour centers for ocr
        self.cA_center = spiceidx_to_locidx[vocA_idx]
        self.cB_center = spiceidx_to_locidx[vocB_idx]

        # DONE -------------------------------------------


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

            img_path = self.debug_imgs_path + 'spice0_col_cropped.png'
            cv2.imwrite(img_path,spice0_col_cropped)
            
            img_path = self.debug_imgs_path + 'spice1_col_cropped.png'
            cv2.imwrite(img_path,spice1_col_cropped)
            
            img_path = self.debug_imgs_path + 'spice0_mask.png'
            cv2.imwrite(img_path,spice0_mask)
            img_path = self.debug_imgs_path + 'spice1_mask.png'
            cv2.imwrite(img_path,spice1_mask)
            img_path = self.debug_imgs_path + 'spice2_mask.png'
            cv2.imwrite(img_path,spice2_mask)
            img_path = self.debug_imgs_path + 'spice3_mask.png'
            cv2.imwrite(img_path,spice3_mask)
            
            img_path = self.debug_imgs_path + 'spice0_col.png'
            cv2.imwrite(img_path,spice0_col)
            img_path = self.debug_imgs_path + 'spice1_col.png'
            cv2.imwrite(img_path,spice1_col)
            img_path = self.debug_imgs_path + 'spice2_col.png'
            cv2.imwrite(img_path,spice2_col)
            img_path = self.debug_imgs_path + 'spice3_col.png'
            cv2.imwrite(img_path,spice3_col)

            hist_img = self.createHistIMG(spice0_bhist,
                                          spice1_bhist,
                                          spice2_bhist,
                                          spice3_bhist,
                                          spice0_bhist_peak,
                                          spice1_bhist_peak,
                                          spice2_bhist_peak,
                                          spice3_bhist_peak,
                                          spice0_bhist_ev,
                                          spice1_bhist_ev,
                                          spice2_bhist_ev,
                                          spice3_bhist_ev)
            img_path = self.debug_imgs_path + 'hist_img.png'
            cv2.imwrite(img_path,hist_img)

            cv2.drawContours(blobs_bgr,[c0],0,(0,255,0),2)
            cv2.circle(blobs_bgr,c0_center,5,(255,0,0),-1)
            cv2.drawContours(blobs_bgr,[c1],0,(0,255,0),2)
            cv2.circle(blobs_bgr,c1_center,5,(255,0,0),-1)
            cv2.drawContours(blobs_bgr,[c2],0,(0,255,0),2)
            cv2.circle(blobs_bgr,c2_center,5,(255,0,0),-1)
            cv2.drawContours(blobs_bgr,[c3],0,(0,255,0),2)
            cv2.circle(blobs_bgr,c3_center,5,(255,0,0),-1)
            cv2.circle(blobs_bgr,self.centroid,5,(0,0,255),-1)
            self.blob_img = blobs_bgr
            img_path = self.debug_imgs_path + 'blobs_bgr.png'
            cv2.imwrite(img_path,blobs_bgr)


        # emergency plan
        #hsv_img_og = cv2.cvtColor(self.col_cropped, cv2.COLOR_RGB2HSV)
        # All mask
        #all_range = np.array([[0, 0, 50], [180, 255, 255]], dtype=np.uint16)
        #all_mask = cv2.inRange(hsv_img_og, all_range[0], all_range[1])

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

        self.quadrant_dict = {}

        self.ov_idxs = None

        # Debug imgs
        self.hist_img = None
        self.spice0_col_img = None
        self.spice1_col_img = None
        self.blob_img = None

        self.vocA_img_path = '/home/jau/ros/catkin_ws/src/idx_finder/scripts/vocA.png'
        self.vocB_img_path = '/home/jau/ros/catkin_ws/src/idx_finder/scripts/vocB.png'

        self.debug_imgs_path='/home/jau/ros/catkin_ws/src/idx_finder/scripts/debug_imgs/'


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

        
        all_mask = cv2.bitwise_and(mask_no_holes,og_mask_inv)
        img_path = self.debug_imgs_path + 'all_mask.png'
        cv2.imwrite(img_path,all_mask)
        return all_mask


    def clean_mask(self,mask):
        # Apply structuring element
        k = 10
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k,k))
        morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        #morph = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return morph

    def get_color_cropped(self,cont,col_img):
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