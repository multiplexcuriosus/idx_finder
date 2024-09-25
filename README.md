# The idx_finder node
## Overview
The `idx_finder` communicates to the spice_up_coordinator through the `FindIndex` service.
The idx_finder contains three modules:
* Cropper
* OCRLocalizer
* HistogramLocalizer

The following diagram gives an overview over the information flow:
![sa_slide_extraction-2](https://github.com/user-attachments/assets/dfa406e3-daf2-415a-a3bc-d2a7d2a0ce85)

## Cropper
The `index_finder_server` receives a `FindIndex` request, which contains a color-img, a mask and the information of whether or not the mask contains 5 contours or not (`has_five_contours`). If `has_five_contours == True`, the Cropper continues with the 4-hole-approach, else, with the thresholding-approach

**Important: If the `all_bottles_mask` was obtained via thresholding, it is assumed that it only contains three blobs: vinegar, oil, salt. Pepper is assumed to have been suppressed by the thresholding.**  

Thus for the rest of the Cropper-pipeline, a distinction is made between thresholding-approach & 4-hole-approach, since they make different assumptions about how many blobs are present in the `all_bottles_mask`.

### 4-hole-approach
![sa_slide_extraction-4](https://github.com/user-attachments/assets/7e2f8815-7cf9-4dbe-9b8c-99bbe5e49744)
1. In `get_all_bottles_mask`: From the information given in the `FindIndexRequest` the `all_bottles_mask` is obtained in the `get_mask_of_holes`-method. This is done with simple mask operations (inversion, bitwise_and). From now on it is assumed that there are four blobs in the `all_bottles_mask`.
2. The `all_bottles_color` img is created by bitwise-and of the `all_bottles_mask` and `og_color` img.
3. & 4. The `spice_color_bbox` and `spice_col_tight` imgs are created with the `create_tight_spice_images`-method.
5.  The brightness-histogram-means are computed in the `get_brightness_histogram_means`-method.
6.  The salt-pepper classification is made based on the means from `get_brightness_histogram_means`.

### Thresh-approach
![sa_slide_extraction-5](https://github.com/user-attachments/assets/04e27a41-a48c-42c6-9785-6dbbdc426fc0)
1. In `get_all_bottles_mask`: A brightness threshold is applied to `og_color` and the result is bitwise anded with `og_mask_no_holes` to obtain the `all_bottles_mask`.  From now on it is assumed that there are three blobs in the `all_bottles_mask`.
2. The `all_bottles_color` img is created by bitwise-and of the `all_bottles_mask` and `og_color` img.
3. The pepper-location-index is assumed to be the "empty" quadrant in the `all_bottles_mask`. The salt-location-index is assumed to be the location-index of the smallest blob in the `all_bottles_mask`.
4. & 5. The `spice_color_bbox` and `spice_col_tight` imgs are created with the `create_tight_spice_images`-method.



