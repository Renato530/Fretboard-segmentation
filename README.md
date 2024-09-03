# Fretboard-segmentation
This is a projet to segment the fretboard of a guitar. The pipeline will be explained in the Usage section. 

## Installation
### Pre-requisites
1. Install miniconda with python version 3.7

### On terminal command steps
1. Clone this repository
2. Creaate the environnment with the following command (only one time):
    ```
    conda env create -f environment.yml
    ```
 
## Usage
In order to use the following fretboard segmentation, you must make sure to have the hed-model folder in your repository. This folder contains de deep learning model for the edge detection.

1. Start the script in a terminal panel with the following command:
    ```
    python FREG-SEG.py -c (filename)
    ```
2. Select the directory of the guitar images for the segmentation procedure.
3. Select the directory to save the fretboard masks obtained.
4. Select a region of interest which includes the second half of the guitar (fretboard).

## Descriptions
### Select ROI
Open CV is used to select a general ROI that will be applied to all the images of the given folder. This steps helps reduce the background noise effects on the image and isolate area of interest.

### Edge detection
Canny for edge detection depends on thresholds which do not work in all environment conditions of the image. For example, this approach is very sensitive to bright light conditions. Hence, the holistically-nested edge detection (HED) is a deep learning implementation of an edge detector. This approach is more suited to extract the edges of the fretboard regardless of background conditions.

### Fretboard edges extraction
In order to extract the edges of the fretboard, we first rotate the edge detection result to align the guitar with the horizontal plane using hough transform. Then a convolution mask with the shape of a horizontal line is applied to enhance the main edges of the fretboard. 

A probabilistic Hough transform is applied on the rotated image. Using a simple clustering algorithm, we obtain 2 classes: top edge and bottom edge. A linear regression is then applied on each class to get the main lines that define the fretboard edges.

### Fretboard extraction
First, the rotated image is decomposed using a wavelet decomposition. We select the vertical details of the image to apply a convolution mask with the shape of a vertical line that defines de frets. We use a peak finding algorithm to detect the peaks of pixel intensity of the image to define the nut and the last fret from the nut. Finally, we use the peak of the nut and the peak of the last fret to extract the intersection points with the fretboard edge lines to construct a boundary box that defines the mask of the fretboard.


