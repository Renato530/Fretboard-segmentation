import numpy as np
import os
import cv2
import tkinter 
import matplotlib.pyplot as plt
import math
import pywt

from tkinter import messagebox
from tkinter import filedialog
from scipy import ndimage
from scipy.signal import find_peaks
from skimage.draw import polygon2mask
from skimage.morphology import skeletonize
from sklearn import linear_model


def load_HED():
    """Load the holistically-nested edge detection (HED)
    Parameters
    ----------
    None.
    Returns
    -------
    files_HED : list strings
        List of directory files that contain the HED model and deploy txt
    """
    f_path = os.path.dirname(os.path.realpath("__file__")) + "\hed_model"
    files = []
    for filename in os.listdir(f_path):
        f = os.path.join(f_path,filename)
        if f is not None:
            files.append(f)

    return files

def select_folder():
    """Select a directory to save files
    Parameters
    ----------
    None.
    Returns
    -------
    filepath : string
        Directory to save the files
    """
    main_win = tkinter.Tk()
    main_win.withdraw()
    main_win.attributes('-topmost', True) 
    filepath = filedialog.askdirectory(parent=main_win, initialdir= "/", title='Please select a directory of the guitar frames')
    return filepath


def load_images_from_folder(filename):
    """Load all the images from a directory
    Parameters
    ----------
    filename : string or None
        Name of the image before numbering of frames (e.g frame_01 - filename = "frame) (prefix)
    Returns
    -------
    imagesFromFolder : list of 3D np.arrays
        Images from the directory
    """
    main_win = tkinter.Tk()
    main_win.withdraw()
    main_win.attributes('-topmost', True) 
    
    filepath = filedialog.askdirectory(parent=main_win, initialdir="/", title='Please select a directory to export data')
    
        
    imagesFromFolder = []
    if filename is None:
        filenames = os.listdir(filepath)
    else:    
        filenames = sorted(os.listdir(filepath), key=lambda fn: int(fn.split('.')[0][len(filename):]))
        
    for f_n in filenames:
        img = plt.imread(os.path.join(filepath, f_n))
        if img is not None:
            imagesFromFolder.append(img)
    
    return imagesFromFolder

def select_ROI(image):
    """Select the region of interest on the image
    Parameters
    ----------
    image : 3D np.array
        Guitar RGB image
    Returns
    -------
    r : list int
        Coordinates of the ROI
    imCrop : 3D np.array
        Region of interest selected by the user
    """
    img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    r = cv2.selectROI("Select ROI", img_rgb)
    imCrop = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    cv2.imshow("Image Crop", cv2.cvtColor(imCrop, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return r, imCrop

def cannyDetection(image, l = 7):
    """Canny edge detection of the image
    Parameters
    ----------
    image : 3D np.array
        Guitar RGB image
    l : int
        Length of the kernel to do gaussian blurring before applying the canny edge detection
    Returns
    -------
    canny_edges : np.array
        Edges detected on the image
    """
    blurring = np.uint8(cv2.GaussianBlur(image, (l, l), 0))
    ret, th = cv2.threshold(np.uint8(image), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canny_edges = cv2.Canny(blurring, int(ret//2), ret)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    canny_edges = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, kernel)
    canny_edges = cv2.dilate(canny_edges, kernel, iterations=2)
    return canny_edges

def Hough_linesP(edges, threshold = 500):
    """Calculate the Probabilistic Hough Transform of an edge detection image
    Parameters
    ----------
    edges : np.array
        Gray image that defines the edges detected of the RGB image
    threshold : int
        Threshold applied to define a line in the Hough Transform accumulator
    Returns
    -------
    linesP : list np.array
        List of the coordinates of the beginning and the end of the lines detected
    """
    linesP = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, 0, 0)
    return linesP
    
def find_guitar_angle(linesP):
    """Finds the main angle between the fretboard of the guitar and the horizontal plane
    Parameters
    ----------
    linesP : list np.array
        List of the coordinates of the beginning and the end of the lines detected
    Returns
    -------
    guitar_angle : int
        Angle between the fretboard and the horizontal plane
    """
    angles = np.zeros((len(linesP), 1))
    for i in range(len(linesP)):
        line = linesP[i][0]
        v1 = [line[2] - line[0], line[3] - line[1]]
        v2 = [1, 0]
        u1 = v1 / np.linalg.norm(v1)
        u2 = v2 / np.linalg.norm(v2)
        dot_product = np.dot(u1, u2)
        angles[i] = np.rad2deg(np.arccos(dot_product))
        if v1[1] > 0:
            angles[i] = -angles[i]
    
    angles = angles[~np.isnan(angles)] # remove NaN
    angles = angles[angles != 0] # remove zero elements
    guitar_angle = np.median(angles)
    return round(guitar_angle)

def rotate_image(image, guitar_angle):
    """Rotate the image to align the fretboard with the horizontal plane
    Parameters
    ----------
    image : 3D np.array
        Guitar RGB image
    guitar_angle : int
        Angle between the fretboard and the horizontal plane
    Returns
    -------
    rotatedImg : 3D np.array
        The image with the fretboard aligned with the horizontal plane
    """
    h, w = image.shape
    cX, cY = (w//2, h//2)
    M = cv2.getRotationMatrix2D((cX, cY), -guitar_angle, 1.0)
    rotatedImg = cv2.warpAffine(image, M, (w, h))
    return rotatedImg

def sobel_masks(image):
    """Applies sobels a horizontal sobel mask to the image
    Parameters
    ----------
    image : 3D np.array
        Guitar RGB image
    Returns
    -------
    absGradX : np.array
        Gray image of the horizontal gradient of the image
    """
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta)
    absGradX = cv2.convertScaleAbs(grad_x)
    return absGradX

def wavelet_transform(image):
    """Decompose the image to horizontal, vertical and diagonal details with a wavelet transform
    Parameters
    ----------
    image : 3D np.array
        Guitar RGB image
    Returns
    -------
    output_HL3 : np.array
        Vertical details obtained with a depth 3 wavelet decomposition
    """
    r, c = image.shape
    width = int(image.shape[1])
    height = int(image.shape[0])
    coeffs = pywt.dwt2(image, 'bior1.3')
    LL, (LH, HL, HH) = coeffs
    dsize = (width, height)
    output_LL = cv2.resize(LL[0:r//2, 0:c//2], dsize)
    coeffs2 = pywt.dwt2(output_LL, 'bior1.3')
    LL2, (LH2, HL2, HH2) = coeffs2
    output_LL2 = cv2.resize(LL2[0:r//2,0:c//2], dsize)
    coeffs3 = pywt.dwt2(output_LL2, 'bior1.3')
    LL3, (LH3, HL3, HH3) = coeffs3
    output_HL3 = cv2.resize(HL3[0:r//2, 0:c//2], dsize)
    return output_HL3


def frets_extraction(vertical_image, kernel_lenght = 200, niterations=1):
    """Applies a convolution to enhance the pixels corresponding to the frets and applies a skeletonization algorithm
    Parameters
    ----------
    vertical_image : np.array
        Grayscale image with the enhanced vertical details
    kernel_lenght : int
        Length of the kernel of the convolution mask to enhance the frets
    niterations : int
        Number of iterations that the erosion applies
    Returns
    -------
    frets : np.array
        Binary image with the frets detected
    """

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_lenght))
    frets = cv2.morphologyEx(vertical_image, cv2.MORPH_OPEN, vertical_kernel, iterations=niterations)
    frets = cv2.medianBlur(np.uint8(frets)*255, 9)
    frets = skeletonize(frets, method = "lee")
    return frets

def fretboardEdges(edges,kernel_lenght = 50, niterations=1):
    """Select the region of interest on the image
    Parameters
    ----------
    edges : np.array
        Edges detection on the guitar image with HED or canny
    kernel_lenght : int
        Length of the convolution mask to extract horizontal details
    niterations : int
        Number of iterations that the erosion applies
    Returns
    -------
    im_horizontal : np.array
        Binary image that contains the edges of the fretboard
    """
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_lenght, 1))
    fretEdges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel, iterations=niterations)
    fretEdges = cv2.dilate(fretEdges, horizontal_kernel, iterations=2)
    return fretEdges

def vertical_proj_peaks(frets, alpha = 0.2):
    """Applies a vertical projection on the frets image and detect peaks that corresponds to the position of the frets
    Parameters
    ----------
    frets : np.array
        Binary image that contains the frets detected
    alpha : int
        Parameter that defines a condition to detect a peak using prominence
    Returns
    -------
    peaks_vert : list int
        List of coordinates of the peaks the detected on the vertical projection (correspond to the position of the frets)
    """
    vertical_proj = np.sum(frets, 0)
    peaks_vert, _ = find_peaks(vertical_proj, prominence=alpha*np.max(np.abs(vertical_proj))) 
    return peaks_vert

def mask_pts(frets, peaks_vert, threshold=450, vmin=0, vmax=-1):
    """Coordinates of the boundary box of the fretboard
    Parameters
    ----------
    frets : np.array
        Binary image with the frets detected
    peaks_vert : list int
        List of coordinates of the peaks the detected on the vertical projection (correspond to the position of the frets)
    threshold : int
        Threshold for the lines detected by the Probabilistic Hough Transform
    vmin : int
    vmax : int
    Returns
    -------
    pts : list np.array
        List of the coordinates of the boundary box
    """
    linesP = Hough_linesP(frets, threshold)
    points = np.zeros((len(linesP), 4))
    for i in range(len(linesP)):
        points[i] = linesP[i][0]
    points.reshape(2 * len(linesP), 2)

    th = (np.max(points[:, 1]) + np.min(points[:, 1])) / 2
    pts_1 = []
    pts_2 = []
    for i in range(len(linesP)):
        if points[i, 1] >= th:
            pts_1.append(points[i])
        else:
            pts_2.append(points[i])
    pts_1 = np.array(pts_1)
    pts_2 = np.array(pts_2)

    topEdge, bottomEdge = clustering_pts(pts_1, pts_2)
    x1 = np.zeros([len(topEdge), 1])
    x2 = np.zeros([len(bottomEdge), 1])
    y1 = topEdge[:, 1]
    y2 = bottomEdge[:, 1]
    for i in range(len(topEdge)):
        x1[i] = topEdge[i, 0]

    for i in range(len(bottomEdge)):
        x2[i] = bottomEdge[i, 0]
    regr1 = linear_model.LinearRegression()
    regr2 = linear_model.LinearRegression()

    regr1.fit(x1, y1)
    regr2.fit(x2, y2)

    xx1 = np.array([[peaks_vert[vmin]], [peaks_vert[vmax]]])
    y1_pred = regr1.predict(xx1)
    y2_pred = regr2.predict(xx1)

    p1 = np.int0([peaks_vert[vmin], y1_pred[0]])
    p2 = np.int0([peaks_vert[vmin], y2_pred[0]])
    p3 = np.int0([peaks_vert[vmax], y2_pred[1]])
    p4 = np.int0([peaks_vert[vmax], y1_pred[1]])
    pts = np.array([p1, p2, p3, p4])

    return pts

def clustering_pts(pts_1, pts_2):
    """Select the region of interest on the image
    Parameters
    ----------
    pts_1 : np.array
        Coordinates of the top class of the fretboard
    pts_2 : np.array
        Coordinates of the bottom class of the fretboard
    Returns
    -------
    topEdge : np.array
        Coordinates of the top edge of the fretboard
    bottomEdge : np.array
        Coordinates of the bottom edge of the fretboard
    """
    maximum = np.max(pts_1[:, 1])
    minimum = np.min(pts_1[:, 1])
    distance = maximum - minimum
    moy = (maximum + minimum)/2
    topEdge = pts_1
    while distance > 30:
        pt = []
        for i in range(len(topEdge)):
            if topEdge[i, 1] < moy:
                pt.append(topEdge[i])
        topEdge = np.array(pt)
        maximum = np.max(topEdge[:, 1])
        minimum = np.min(topEdge[:, 1])
        distance = maximum - minimum
        moy = (maximum + minimum)/2

    maximum = np.max(pts_2[:, 1])
    minimum = np.min(pts_2[:, 1])
    distance = maximum - minimum
    moy = (maximum + minimum)/2
    bottomEdge = pts_2

    while distance > 30:
        pt = []
        for i in range(len(bottomEdge)):
            if bottomEdge[i, 1] < moy:
                pt.append(bottomEdge[i])
        bottomEdge = np.array(pt)
        maximum = np.max(bottomEdge[:, 1])
        minimum = np.min(bottomEdge[:, 1])
        distance = maximum - minimum
        moy = (maximum + minimum)/2

    return topEdge, bottomEdge

def fretboard_mask(image, pts, guitar_angle):
    """Creates the fretboard mask
    Parameters
    ----------
    image : 3D np.array
        Guitar RGB image
    pts : list np.array
        List of the coordinates of the boundary box
    guitar_angle : int
        Angle between the fretboard of the guitar and the horizontal plane
    Returns
    -------
    mask : np.array
        Binary image that contains the fretboard mask
    """
    image_shape = image.shape
    polygon = np.array([pts[0][::-1], pts[1][::-1], pts[2][::-1], pts[3][::-1]])
    mask = np.uint8(1*polygon2mask(image_shape, polygon))
    mask = rotate_image(mask, -guitar_angle)
    return mask

def write_masks(filepath, filename, mask):
    """Write the mask on the selected directory from the user
    Parameters
    ----------
    filepath : string
        Directory of the folder to save the mask obtained
    filename : string
        Prefix for the file
    mask : np.array
        Binary image that contains the fretboard mask
    Returns
    -------
    None.
    """
    directory = os.path.join(filepath, filename)
    cv2.imwrite(directory, mask)

def fretbord_correction(image):
    """Manually annotate the image to obtain a fretboard mask
    Parameters
    ----------
    image : 3D np.array
        Guitar RGB image
    Returns
    -------
    mask : np.array
        Binary image that contains the fretboard mask
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    a = []
    b = []
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            a.append(x)
            b.append(y)
            cv2.circle(image, (x, y), 2, (0, 0, 255), thickness=5)
            cv2.putText(image, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (255, 0, 0), thickness=3)
            cv2.imshow("image", image)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    c = np.array([[a[0], b[0]], [a[2], b[2]], [a[3], b[3]], [a[1], b[1]]])
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    polygon = np.array([c[0][::-1], c[1][::-1], c[2][::-1], c[3][::-1]])

    mask = np.uint8(1*polygon2mask(img.shape, polygon))
    return mask