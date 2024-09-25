markdown_extensions:
  - sane_lists

# The idx_finder node
## Overview
The `idx_finder` communicates to the spice_up_coordinator through the `FindIndex` service.
The idx_finder contains three modules:
* Cropper
* OCRLocalizer
* HistogramLocalizer

The following diagram gives an overview over the information flow:
![sa_slide_extraction-2](https://github.com/user-attachments/assets/2a42cc32-e6af-4621-adc9-8cda157343b8)

## Cropper
The `index_finder_server` receives a `FindIndex` request, which contains a color-img, a mask and the information of whether or not the mask contains 5 contours or not (`has_five_contours`). If `has_five_contours == True`, the Cropper continues with the 4-hole-approach, else, with the thresholding-approach

Thus for the rest of the Cropper-pipeline, a distinction is made between thresholding-approach & 4-hole-approach, since they make different assumptions about how many blobs are present in the `all_bottles_mask`.

### 4-hole-approach
![sa_slide_extraction-4](https://github.com/user-attachments/assets/123fbb49-8d50-4082-89a6-78d806a3646c)
1. In `get_all_bottles_mask`: From the information given in the `FindIndexRequest` the `all_bottles_mask` is obtained in the `get_mask_of_holes`-method. This is done with simple mask operations (inversion, bitwise_and). From now on it is assumed that there are four blobs in the `all_bottles_mask`.
2. The `all_bottles_color` img is created by bitwise-and of the `all_bottles_mask` and `og_color` img.
3. & 4. The `spice_color_bbox` and `spice_col_tight` imgs are created with the `create_tight_spice_images`-method.

5.  The brightness-histogram-means are computed in the `get_brightness_histogram_means`-method.
6.  The salt-pepper classification is made based on the means from `get_brightness_histogram_means`.

### Thresh-approach
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



