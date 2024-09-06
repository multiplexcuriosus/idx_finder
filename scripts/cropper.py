import numpy as np
import cv2

class Cropper:
    def __init__(self,img) -> None:

        cropped = img.copy()

        self.status = "FAIL"

        self.image_vert_crop = 70
        self.image_hor_crop = 190
        self.padding = 40
        
        '''
        lower_x = 479-self.image_vert_crop
        lower_y = 639-self.image_hor_crop
        cropped[0:479,0:self.image_hor_crop,:] = 0 
        cropped[0:479,lower_y:639,:] = 0 
        cropped[0:self.image_vert_crop,0:639,:] = 0 
        cropped[lower_x:479,0:639,:] = 0 
        '''
        self.col_cropped = cropped

         # Debug imgs
        self.hist_img = None
        self.voc0_img = None
        self.voc1_img = None

        self.debug = False

        hsv_img_og = cv2.cvtColor(self.col_cropped, cv2.COLOR_RGB2HSV)

        # All mask
        all_range = np.array([[0, 0, 30], [180, 255, 255]], dtype=np.uint16)
        all_mask = cv2.inRange(hsv_img_og, all_range[0], all_range[1])
        if self.debug:
            cv2.imshow("All mask",all_mask)

        # All mask clean
        self.all_mask_clean = self.clean_mask(all_mask)
        if self.debug:
            cv2.imshow("All mask clean",self.all_mask_clean)

        # All color
        all_color = cv2.bitwise_and(img, img, mask=self.all_mask_clean)
        if self.debug:
            cv2.imshow("All color",all_color)

        if self.debug:
            cv2.waitKey(0)

        # Blob detection
        blobs_bgr = cv2.cvtColor(self.all_mask_clean, cv2.COLOR_GRAY2BGR)
        contours, hier = cv2.findContours(self.all_mask_clean,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 3:
            return 

        cntsSorted = sorted(contours, key=lambda x: -cv2.contourArea(x))

        c0 = cntsSorted[0] # largest contour
        self.c0_center = self.get_center_of_countour(c0)

        c1 = cntsSorted[1] # second largest contour
        self.c1_center = self.get_center_of_countour(c1)

        c2 = cntsSorted[2] # third largest contour
        self.c2_center = self.get_center_of_countour(c2)

        self.centroid = (int(np.average([self.c0_center[0],self.c1_center[0],self.c2_center[0]])),int(np.average([self.c0_center[1],self.c1_center[1],self.c2_center[1]])))

        self.voc0_idx = self.locate(self.c0_center)
        self.voc1_idx = self.locate(self.c1_center)
        self.voc2_idx = self.locate(self.c2_center)
        self.occupied_idxs = list([self.voc0_idx,self.voc1_idx,self.voc2_idx])

        if self.debug:
            cv2.drawContours(blobs_bgr,[c0],0,(0,255,0),2)
            cv2.circle(blobs_bgr,self.c0_center,5,(255,0,0),-1)
            cv2.drawContours(blobs_bgr,[c1],0,(0,255,0),2)
            cv2.circle(blobs_bgr,self.c1_center,5,(255,0,0),-1)
            cv2.drawContours(blobs_bgr,[c2],0,(0,255,0),2)
            cv2.circle(blobs_bgr,self.c2_center,5,(255,0,0),-1)
            cv2.circle(blobs_bgr,self.centroid,5,(0,0,255),-1)
            cv2.imshow('Blobs', blobs_bgr)

        # Mask detection
        vom0 = self.get_cont_mask(c0)
        vom1 = self.get_cont_mask(c1)
        if self.debug and False:
            cv2.imshow("VOM 1",vom1)
            cv2.imshow("VOM 0",vom0)


        # Get color_img
        voc0 = cv2.bitwise_and(img, img, mask=vom0)
        voc1 = cv2.bitwise_and(img, img, mask=vom1)
        if self.debug:
            cv2.imshow("VOC 1",voc1)
            cv2.imshow("VOC 0",voc0)

        # Get color cropped
        voc0_cropped = self.get_color_cropped(c0,voc0)
        self.voc0_img = voc0_cropped
        voc1_cropped = self.get_color_cropped(c1,voc1)
        self.voc1_img = voc1_cropped

        if self.debug:
            cv2.imshow("VOC0 cropped",voc0_cropped)
            cv2.imshow("VOC1 cropped",voc1_cropped)

        self.status = "success"

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
        vom_bgr = cv2.cvtColor(self.all_mask_clean, cv2.COLOR_GRAY2BGR)
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