import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import find_peaks

class HistogramLocalizer:

    def __init__(self,voc0_cropped,voc1_cropped) -> None:
        
        self.status = "success"
        self.hist_img = None

        if voc0_cropped is None or voc1_cropped is None:
            print("HL: ERROR: img is None")
            self.status ="FAIL"
            return

        self.debug = False

        # Vinegar/Oil hue hist
        voc0_hue_hist = self.get_hue_histogram(voc0_cropped)
        voc1_hue_hist = self.get_hue_histogram(voc1_cropped)

        # Vinegar/Oil hue hist peak
        voc0_hue_peak_point = self.get_peak_of_histogram(voc0_hue_hist) 
        voc1_hue_peak_point = self.get_peak_of_histogram(voc1_hue_hist)

        # Normalize Vinegar/Oil hue hist
        voc0_hist_sum = voc0_hue_hist.sum()
        voc1_hist_sum = voc1_hue_hist.sum()
        voc0_hue_peak_height = voc0_hue_peak_point[1] / voc0_hist_sum
        voc1_hue_peak_height = voc1_hue_peak_point[1] / voc1_hist_sum

        voc0_hue_hist /= (voc0_hist_sum)
        voc1_hue_hist /= (voc1_hist_sum)

        # Make the two hist distributions equal height
        fac = voc0_hue_peak_height/voc1_hue_peak_height
        voc1_hue_hist *= fac
        voc1_hue_peak_height *= fac

        # Vinegar/Oil hue hist Expectec Value
        L = len(voc0_hue_hist)
        V = np.arange(1,L)
        voc0_hue_mean = self.expected_value(V,voc0_hue_hist[1:])  
        voc1_hue_mean = self.expected_value(V,voc1_hue_hist[1:]) 

        print("voc0 hue peak x: "+str(voc0_hue_peak_point[0]))
        print("voc1 hue peak x: "+str(voc1_hue_peak_point[0]))
        print("voc0_hue_mean: "+str(voc0_hue_mean))
        print("voc1_hue_mean: "+str(voc1_hue_mean))

        if self.debug or False:
            #print("voc0_hue_peak_point: "+str(voc0_hue_peak_point))
            #print("voc1_hue_peak_point: "+str(voc1_hue_peak_point))

            fig, ax = plt.subplots(1)
            ax.plot(voc0_hue_hist[1:], color='blue', label="Hue VOC0")
            ax.plot(voc1_hue_hist[1:], color='red', label="Hue VOC1")
            ax.plot(voc0_hue_peak_point[0],voc0_hue_peak_height,'o',color='blue')
            ax.plot(voc1_hue_peak_point[0],voc1_hue_peak_height,'o',color='red')
            plt.axvline(x=voc0_hue_mean, color='blue', linestyle='--',label="mean")
            plt.axvline(x=voc1_hue_mean, color='red', linestyle='--',label="mean")
            ax.legend(loc="upper right")
            #plt.show()
            fig.canvas.draw()
            hist = np.array(fig.canvas.renderer.buffer_rgba())
            self.hist_img = cv2.cvtColor(hist, cv2.COLOR_RGBA2RGB)


        # Vinegar/Oil classification decision
        self.voc0_has_higher_hue_peak = False
        self.voc1_has_higher_hue_peak = False
        self.voc0_has_higher_hue_mean = False
        self.voc1_has_higher_hue_mean = False

        if voc0_hue_peak_point[0] > voc1_hue_peak_point[0]:
            print("Larger peak x:   VOC0")
            self.voc0_has_higher_hue_peak = True
        elif voc0_hue_peak_point[0] < voc1_hue_peak_point[0]:
            print("Larger peak x:   VOC1")
            self.voc1_has_higher_hue_peak = True
        else:
            print("Same hue peak x")

        if voc0_hue_mean > voc1_hue_mean:
            print("Larger hue mean:   VOC0")
            self.voc0_has_higher_hue_mean = True
        elif voc0_hue_mean < voc1_hue_mean:
            print("Larger hue mean:   VOC1")
            self.voc1_has_higher_hue_mean = True
        else:
            print("Same hue mean")


        if self.debug:
            cv2.waitKey(0)

    def expected_value(self,values, weights):
        values = np.asarray(values)
        weights = np.asarray(weights)
        return (np.dot(values,weights))

    def get_hue_histogram(self,voc):
        voc_hsv = cv2.cvtColor(voc, cv2.COLOR_RGB2HSV)
        h, s, v = voc_hsv[:,:,0], voc_hsv[:,:,1], voc_hsv[:,:,2]
        l = 0
        return cv2.calcHist([h],[0],None,[180],[0,180])[l:]

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

'''
voc0_img = cv2.imread('/home/jau/ros/catkin_ws/src/color_mask_localizer/scripts/last_voc0.png')
voc1_img = cv2.imread('/home/jau/ros/catkin_ws/src/color_mask_localizer/scripts/last_voc1.png')

HL = HistogramLocalizer(voc0_img,voc1_img)
'''