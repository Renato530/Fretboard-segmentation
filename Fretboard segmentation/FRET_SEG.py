'''
    File name: FRET_SEG.py
    Author: Renato Castillo
    Date created: 6/11/2021
    Date last modified: 8/11/2021
    Python Version: 3.7
'''

import matplotlib.pyplot as plt
import numpy as np
import cv2
import datetime
import argparse
from skimage.draw import polygon2mask
# Local modules
import FRETBOARD


def main(fileName):
    """FRET_SEG.py segments the fretboard of the guitar of images.
    The script ask users to input a prefix, select the ROI in a frame and manually annoted if segmentation mistakes
    are present.
    Parameters
    ----------
    fileName : string
        Prefix of the images in the folder.
    Returns
    -------
    None.
    """
    # Create a class object for the holistically-nested edge detection (HED)
    class CropLayer(object):
        def __init__(self, params, blobs):
            self.startX = 0
            self.startY = 0
            self.endX = 0
            self.endY = 0

        def getMemoryShapes(self, inputs):
            (inputShape, targetShape) = (inputs[0], inputs[1])
            (batchSize, numChannels) = (inputShape[0], inputShape[1])
            (H, W) = (targetShape[2], targetShape[3])

            self.startX = int((inputShape[3] - targetShape[3]) / 2)
            self.startY = int((inputShape[2] - targetShape[2]) / 2)
            self.endX = self.startX + W
            self.endY = self.startY + H
            return [[batchSize, numChannels, H, W]]

        def forward(self, inputs):
            return [inputs[0][:, :, self.startY:self.endY,
                    self.startX:self.endX]]

    # Load the guitar images from specified folder
    guitar_images = FRETBOARD.load_images_from_folder(fileName)
    print("Loading images to workspace...")

    # Chose folder to save the masks
    maskDirectory = FRETBOARD.select_folder()

    # Only to visually validate the obtained masks
    overlayDirectory = FRETBOARD.select_folder()

    # Load the HED model
    HED_files = FRETBOARD.load_HED()
    protoPath = HED_files[0]
    modelPath = HED_files[1]
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    cv2.dnn_registerLayer("Crop", CropLayer)

    # Prepare the frame to segment the fretboard
    print("Using the first frame to select ROI...\n")
    guitar = guitar_images[0]

    # Selection of ROI
    r, _ = FRETBOARD.select_ROI(guitar)
    guitar_nCrop = cv2.cvtColor(guitar, cv2.COLOR_BGR2GRAY)
    mask_nCrop = np.zeros(guitar_nCrop.shape)
    mask_nCrop[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_nCrop_erode = cv2.erode(mask_nCrop, kernel, iterations=5)
    begin_time = datetime.datetime.now()
    print("Start of the operation: ", begin_time)
    for i in range(len(guitar_images)):
        guitar = guitar_images[i]
        guitar_nCrop = cv2.cvtColor(guitar, cv2.COLOR_BGR2GRAY)
        guitar_Crop_whole = guitar_nCrop * mask_nCrop
        ImCrop = guitar[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        # HED
        (H, W) = ImCrop.shape[:2]
        blob = cv2.dnn.blobFromImage(ImCrop, scalefactor=1.0, size=(W, H),
                                     swapRB=False, crop=False)
        net.setInput(blob)
        hed = net.forward()
        hed = cv2.resize(hed[0, 0], (W, H))
        hed = (255 * hed).astype("uint8")
        hed = cv2.threshold(hed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        HED_whole = np.zeros(guitar_nCrop.shape)
        HED_whole[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])] = hed
        HED_whole = HED_whole * mask_nCrop_erode

        # Rotate image
        linesP = FRETBOARD.Hough_linesP(hed)
        angle = FRETBOARD.find_guitar_angle(linesP)
        rot_HED = FRETBOARD.rotate_image(HED_whole, angle)

        # Wavelet transform
        rot_mask = FRETBOARD.rotate_image(mask_nCrop_erode, angle)
        rot_img = FRETBOARD.rotate_image(guitar_Crop_whole, angle)
        output_HL3 = FRETBOARD.wavelet_transform(rot_img)
        frets = cv2.threshold(np.uint8(output_HL3 * rot_mask), np.min(output_HL3), np.max(output_HL3),
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        frets = FRETBOARD.frets_extraction(frets * rot_mask, kernel_lenght=100, niterations=1)

        # Fretboard edges extraction
        fretboard_edges = FRETBOARD.fretboardEdges(rot_HED, kernel_lenght=100, niterations=1)
        fretboard_edges = cv2.threshold(np.uint8(fretboard_edges), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Construct bounding box for the fretboard
        peaks_frets = FRETBOARD.vertical_proj_peaks(frets, alpha=0.1)
        pts, _, _ = FRETBOARD.mask_pts(fretboard_edges, peaks_frets, threshold=450, vmin=1, vmax=-1)

        fretboard_mask = FRETBOARD.fretboard_mask(guitar_Crop_whole, pts, angle)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))

        mask_dilate = cv2.dilate(fretboard_mask, dilate_kernel, iterations=1)
        foreground = cv2.cvtColor(mask_dilate * 255, cv2.COLOR_GRAY2BGR)
        background = cv2.cvtColor(guitar, cv2.COLOR_RGB2BGR)
        dst = cv2.addWeighted(background, 1, foreground, 0.5, 0)

        filename = "mask_" + str(i+1) + ".png"
        FRETBOARD.write_masks(maskDirectory, filename, mask_dilate * 255)

        filename2 = "overlay_" + str(i+1) + ".png"
        FRETBOARD.write_masks(overlayDirectory, filename2, dst)

    print("Time it took to finish (hr:min:sec): ", datetime.datetime.now() - begin_time)

    while True:
        option = str(input('Keep current segmentation of fretboards?' + ' (y/n): ')).lower().strip()
        if option[0] == 'n':
            n = int(input("Please enter the number of the frame to correct the segmentation: "))
            mask = FRETBOARD.fretbord_correction(guitar_images[n-1])

            filename = "mask_" + str(n) + ".png"
            FRETBOARD.write_masks(maskDirectory, filename, mask * 255)

            filename2 = "overlay_" + str(n) + ".png"
            foreground = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
            background = cv2.cvtColor(guitar_images[n-1], cv2.COLOR_RGB2BGR)
            dst = cv2.addWeighted(background, 1, foreground, 0.5, 0)
            FRETBOARD.write_masks(overlayDirectory, filename2, dst)
        else:
            break
if __name__ == '__main__':
    # Insert argument -c filename when running the python script
    parser = argparse.ArgumentParser()
    # Select filename with -c filename.
    parser.add_argument('-c', type = str, dest = 'filename', help = 'Enter the name of the file', required = True)
    args = parser.parse_args().__dict__
    filename = args["filename"]
    main(filename)
















