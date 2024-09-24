# The idx_finder node
## Overview
The idx_finder communicates to the spice_up_coordinator through the `FindIndex` service.
The idx_finder contains three modules:
* Cropper
* OCRLocalizer
* HistogramLocalizer
The following diagram gives an overview over the information flow
![sa_slide_extraction-2](https://github.com/user-attachments/assets/dfa406e3-daf2-415a-a3bc-d2a7d2a0ce85)

## Cropper
The `index_finder_server` recieves a `FindIndex` request, which contains a color-img, a mask and the information if the mask contains 5 contours or not (`has_five_contours`).
..
![sa_slide_extraction-3](https://github.com/user-attachments/assets/8153ff6c-d4be-48b3-8910-f6fb5c15f96b)
![sa_slide_extraction-4](https://github.com/user-attachments/assets/7e2f8815-7cf9-4dbe-9b8c-99bbe5e49744)
![sa_slide_extraction-5](https://github.com/user-attachments/assets/04e27a41-a48c-42c6-9785-6dbbdc426fc0)


## Information flow
This diagram depicts the relation between the spice_up_coordinator and the mentioned nodes.
![spice_up_nodes](https://github.com/user-attachments/assets/94ca1baa-e273-4804-a574-ece3452ac3f9)
The numbers indicate the sequence of the events and the colors the nodes which are either requesting or responding.
