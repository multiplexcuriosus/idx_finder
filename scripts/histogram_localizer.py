import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import find_peaks

class HistogramLocalizer:

    def __init__(self,vocA_cropped,vocB_cropped,debug) -> None:
        
        self.status = "success"
        self.hist_img = None

        if vocA_cropped is None or vocB_cropped is None:
            print("HL: ERROR: img is None")
            self.status ="FAIL"
            return

        self.debug = debug

        # Vinegar/Oil hue hist
        vocA_hue_hist = self.get_hue_histogram(vocA_cropped)
        vocB_hue_hist = self.get_hue_histogram(vocB_cropped)

        # Vinegar/Oil hue hist peak
        vocA_hue_peak_point = self.get_peak_of_histogram(vocA_hue_hist) 
        vocB_hue_peak_point = self.get_peak_of_histogram(vocB_hue_hist)

        # Normalize Vinegar/Oil hue hist
        vocA_hist_sum = vocA_hue_hist.sum()
        vocB_hist_sum = vocB_hue_hist.sum()
        vocA_hue_peak_height = vocA_hue_peak_point[1] / vocA_hist_sum
        vocB_hue_peak_height = vocB_hue_peak_point[1] / vocB_hist_sum

        vocA_hue_hist /= (vocA_hist_sum)
        vocB_hue_hist /= (vocB_hist_sum)

        # Make the two hist distributions equal height
        fac = vocA_hue_peak_height/vocB_hue_peak_height
        vocB_hue_hist *= fac
        vocB_hue_peak_height *= fac

        # Vinegar/Oil hue hist Expectec Value
        L = len(vocA_hue_hist)
        V = np.arange(1,L)
        vocA_hue_mean = self.compute_expected_value(V,vocA_hue_hist[1:])  
        vocB_hue_mean = self.compute_expected_value(V,vocB_hue_hist[1:]) 

        print("vocA hue peak x: "+str(vocA_hue_peak_point[0]))
        print("vocB hue peak x: "+str(vocB_hue_peak_point[0]))
        print("vocA_hue_mean: "+str(vocA_hue_mean))
        print("vocB_hue_mean: "+str(vocB_hue_mean))

        if self.debug:
            #print("vocA_hue_peak_point: "+str(vocA_hue_peak_point))
            #print("vocB_hue_peak_point: "+str(vocB_hue_peak_point))

            fig, ax = plt.subplots(1)
            ax.plot(vocA_hue_hist[1:], color='blue', label="Hue vocA")
            ax.plot(vocB_hue_hist[1:], color='red', label="Hue vocB")
            ax.plot(vocA_hue_peak_point[0],vocA_hue_peak_height,'o',color='blue')
            ax.plot(vocB_hue_peak_point[0],vocB_hue_peak_height,'o',color='red')
            plt.axvline(x=vocA_hue_mean, color='blue', linestyle='--',label="mean")
            plt.axvline(x=vocB_hue_mean, color='red', linestyle='--',label="mean")
            ax.legend(loc="upper right")
            #plt.show()
            fig.canvas.draw()
            hist = np.array(fig.canvas.renderer.buffer_rgba())
            self.hist_img = cv2.cvtColor(hist, cv2.COLOR_RGBA2RGB)


        # Vinegar/Oil classification decision bools
        self.vocA_has_higher_hue_peak = False
        self.vocB_has_higher_hue_peak = False
        self.vocA_has_higher_hue_mean = False
        self.vocB_has_higher_hue_mean = False

        if vocA_hue_peak_point[0] > vocB_hue_peak_point[0]:
            print("Larger peak x:   vocA")
            self.vocA_has_higher_hue_peak = True
        elif vocA_hue_peak_point[0] < vocB_hue_peak_point[0]:
            print("Larger peak x:   vocB")
            self.vocB_has_higher_hue_peak = True
        else:
            print("Same hue peak x")

        if vocA_hue_mean > vocB_hue_mean:
            print("Larger hue mean:   vocA")
            self.vocA_has_higher_hue_mean = True
        elif vocA_hue_mean < vocB_hue_mean:
            print("Larger hue mean:   vocB")
            self.vocB_has_higher_hue_mean = True
        else:
            print("Same hue mean")


    def compute_expected_value(self,values, weights):
        values = np.asarray(values)
        weights = np.asarray(weights)
        return (np.dot(values,weights))

    def get_hue_histogram(self,voc):
        voc_hsv = cv2.cvtColor(voc, cv2.COLOR_RGB2HSV)

        # Do not consider black_pixels
        voc_hsv[voc_hsv == 0] = -1

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
vocA_img = cv2.imread('/home/jau/ros/catkin_ws/src/idx_finder/temp_data/vocA.png')
vocB_img = cv2.imread('/home/jau/ros/catkin_ws/src/idx_finder/temp_data/vocB.png')

HL = HistogramLocalizer(vocA_img,vocB_img,True)
'''