# The idx_finder node
## Overview
The `idx_finder` communicates to the spice_up_coordinator through the `FindIndex` service.
The idx_finder contains three modules:
* Cropper
* OCRLocalizer
* HistogramLocalizer

The following diagram gives an overview over the information flow:
![sa_slide_extraction-2](https://github.com/user-attachments/assets/2a42cc32-e6af-4621-adc9-8cda157343b8)
The Cropper does some pre-processing, finds the salt- & pepper-location-indices and passes on cropped imgs of oil and vinegar to the OCRLocalizer.
If the OCRLocalizer succeeds in localizing oil and vinegar, this information is directly returned to the `spice_up_action_server`. 
If not, the cropped images are passed on to the HistogramLocalizer. The HistogramLocalizer always comes to a conclusion about the localization of oil and vinegar, which is then returned to the `spice_up_action_server`.

## Parameters in parameter file
The `index_finder.yaml` file contains the following parameters:
* `HOME` : Insert here the path to the ros package home directory, e.g "/home/jau/ros/catkin_ws/src/idx_finder/"
* `brightness_threshold` : Insert here the brightness threshold for the Cropper-thresholding-approach
* `debug` : bool : set to True to generate debug_imgs 

## Cropper
The `index_finder_server` receives a `FindIndex` request, which contains a color-img, a mask and the information of whether or not the mask contains 5 contours or not (`has_five_contours`). If `has_five_contours == True`, the Cropper continues with the 4-hole-approach, else, with the thresholding-approach

Thus for the rest of the Cropper-pipeline, a distinction is made between thresholding-approach & 4-hole-approach, since they make different assumptions about how many blobs are present in the `all_bottles_mask`.

### 4-hole-approach
Branch of Cropper-pipeline, usecase: `has_five_contours == True` or `all_bottles_mask` has 4 holes.
![sa_slide_extraction (1)-4](https://github.com/user-attachments/assets/4ca0650f-4486-4bf5-982d-f2b570d9fb6e)
1. In `get_all_bottles_mask`: From the information given in the `FindIndexRequest` the `all_bottles_mask` is obtained in the `get_mask_of_holes`-method. This is done with simple mask operations (inversion, bitwise_and). From now on it is assumed that there are four blobs in the `all_bottles_mask`.
2. The `all_bottles_color` img is created by bitwise-and of the `all_bottles_mask` and `og_color` img.
3. The `spice_color_bbox` and `spice_col_tight` imgs are created with the `create_tight_spice_images`-method.
4. The brightness-histogram-means are computed in the `get_brightness_histogram_means`-method.
5. The salt-pepper classification is made based on the means from `get_brightness_histogram_means`.

### Thresh-approach
Branch of Cropper-pipeline, usecase: `has_five_contours == False`
![sa_slide_extraction-5](https://github.com/user-attachments/assets/44413717-0e95-4656-a05b-c084d974ae0f)
1. In `get_all_bottles_mask`: A brightness threshold is applied to `og_color` and the result is bitwise anded with `og_mask_no_holes` to obtain the `all_bottles_mask`.  From now on it is assumed that there are three blobs in the `all_bottles_mask`.
2. The `all_bottles_color` img is created by bitwise-and of the `all_bottles_mask` and `og_color` img.
3. The pepper-location-index is assumed to be the "empty" quadrant in the `all_bottles_mask`. The salt-location-index is assumed to be the location-index of the smallest blob in the `all_bottles_mask`.
4. & 5. The `spice_color_bbox` and `spice_col_tight` imgs are created with the `create_tight_spice_images`-method.

## Spice-location-index vs. spice-index
![sa_slide_extraction-6](https://github.com/user-attachments/assets/35ce1c53-4a76-4a2d-bddd-a9cafe3baf07)
The location-index ({0,1,2,3}) denotes the quadrant a spice can be in. The spice-index stems from the order the `spice_col_tight` were obtained. 
Origin of spice-index: Contour detection is done on the `all_bottles_mask`, i.e 3-4 contours are extracted from the blobs.
The spice beneath contour0 is refered to as spice0 and so on. Genereally spice0 and spice1 correspond to oil and vinegar, since those (mostly) have the largest blobs and the output of the contour-search is ordered by size (largest first).

## OCRLocalizer
The `index_finder_server` instantiates two OCRLocalizers, one for each `spice_col_tight` img the Cropper returns.  

Wrapper for the `easyocr` package. The wrapper allows to specify string-tokens to be searched in an image (henceforth called `oil-token` or `vinegar-token`. More precisely, these tokens are compared to the tokens which the `easyocr.Reader` finds in the image, which he stores in the `result` variable. All the tokens in `result` have an associated certainty score between 0-1. The wrapper currently defines a match as: An  `oil-token` or `vinegar-token` token is contained (case-angnostic) in one of the `result` tokens or the other way around.  

Example: Suppose the `oil-tokens` are "OIL","OI","IL","O". If `result` contains e.g a single "O" with certainty score > 0.1, the OCRLocalizer returns "OIL" as result.

## Histogram-Localizer
The `index_finder_server` instantiates one HistogramLocalizer, with the two `spice_col_tight` imgs the Cropper returns.
The HistogramLocalizer then creates a hue histogram of the two `spice_col_tight` imgs.
The histogram is normalized (divided by sum of distribution data points), such that different bottle patch sizes have no influence on the hue distribution.
For ease of visual interpretation, the two distributions are also normalized to the same peak height. 
Inside the HistogramLocalizer the two `spice_col_tight` imgs are refered to as vocA & vocB (voc -> vinegar-oil-color)
        





