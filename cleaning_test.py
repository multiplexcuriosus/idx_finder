import cv2

idx = 0
path = '/home/jau/ros/catkin_ws/src/idx_finder/debug_imgs/all_bottles_mask_clean.png'
img = cv2.imread(path)
cv2.imshow("og img",img)
cv2.waitKey(0)

def clean(img):

    k = 5
    cleaning_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    cropped_clean = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cleaning_kernel)
    cv2.imshow("cropped_clean",cropped_clean)
    cv2.waitKey(0)


clean(img)