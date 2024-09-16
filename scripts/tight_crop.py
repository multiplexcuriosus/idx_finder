import cv2

idx = 3
path = '/home/jau/Desktop/debug_imgs/spice'+str(idx)+'_mask.png'
img = cv2.imread(path)
cv2.imshow("og img",img)
cv2.waitKey(0)

def tight_crop(img):
    print("shape: "+str(img.shape))
    h,w,_  = img.shape

    row_start = int(h*0.5)
    row_end = int(h*1.0)
    col_start = int(w*0.0)
    col_end = int(w*1.0)
    cropped_image = img[row_start:row_end,col_start:col_end]
    cv2.imshow("cropped",cropped_image)
    cv2.waitKey(0)
    
    k = 10
    cleaning_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    cropped_clean = cv2.morphologyEx(cropped_image, cv2.MORPH_CLOSE, cleaning_kernel)
    cv2.imshow("cropped_clean",cropped_clean)
    cv2.waitKey(0)

    k = 10
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k,k))
    eroded = cv2.erode(cropped_clean, erosion_kernel)
    cv2.imshow("eroded",eroded)
    cv2.waitKey(0)

tight_crop(img)