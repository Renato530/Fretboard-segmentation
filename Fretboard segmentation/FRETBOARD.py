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


def load_model_v2():
    f_path = os.path.dirname(os.path.realpath("__file__")) + "\hed_model"
    files = []
    for filename in os.listdir(f_path):
        f = os.path.join(f_path,filename)
        if f is not None:
            files.append(f)

    return files

def select_folder():
    main_win = tkinter.Tk()
    main_win.withdraw()
    main_win.attributes('-topmost', True) 
    f_path = filedialog.askdirectory(parent=main_win, initialdir= "/", title='Please select a directory of the guitar frames')
    return f_path


def load_images_from_folder(f_n):
    main_win = tkinter.Tk()
    main_win.withdraw()
    main_win.attributes('-topmost', True) 
    
    f_path =  filedialog.askdirectory(parent=main_win, initialdir= "/", title='Please select a directory to export data')
    
        
    images = []
    if f_n == None:
        filenames = os.listdir(f_path)
    else:    
        filenames = filenames = sorted(os.listdir(f_path), key=lambda fn: int(fn.split('.')[0][len(f_n):]))
        
    for filename in filenames:
        img = plt.imread(os.path.join(f_path,filename))
        if img is not None:
            images.append(img)
    
    return images

def select_ROI(img):
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    r = cv2.selectROI("Select ROI", img_rgb)
    imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    cv2.imshow("Image Crop", cv2.cvtColor(imCrop,cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return r,imCrop

def pre_traitement(img,l = 7):
    blurring = np.uint8(cv2.GaussianBlur(img,(l,l),0))
    ret,th = cv2.threshold(np.uint8(img),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    canny_edges = cv2.Canny(blurring,int(ret//4),ret)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    canny_edges = cv2.morphologyEx(canny_edges,cv2.MORPH_CLOSE,kernel)
    canny_edges = cv2.dilate(canny_edges, kernel, iterations=2)
    return canny_edges

def Hough_linesP(img,threshold = 500):
    linesP = cv2.HoughLinesP(img, 1, np.pi / 180, threshold, 0, 0)
    return linesP
    
def find_guitar_angle(linesP):
    angles = np.zeros((len(linesP),1))
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        v1 = [l[2]-l[0],l[3]-l[1]]
        v2 = [1,0]
        u1 = v1 / np.linalg.norm(v1)
        u2 = v2 / np.linalg.norm(v2)
        dot_product = np.dot(u1,u2)
        angles[i] = np.rad2deg(np.arccos(dot_product))
        if v1[1] > 0:
            angles[i] = -angles[i]
    
    angles = angles[~np.isnan(angles)] # remove NaN
    angles = angles[angles != 0] # remove zero elements
    guitar_angle = np.median(angles)
    return round(guitar_angle)

def rotate_image(img,guitar_angle):
    h,w = img.shape
    cX, cY = (w//2,h//2)
    M = cv2.getRotationMatrix2D((cX,cY),-guitar_angle,1.0)
    rot_img = cv2.warpAffine(img,M,(w,h))
    return rot_img

def sobel_masks(img):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta)
    absGradX = cv2.convertScaleAbs(grad_x)
    return absGradX

def wavelet_transform(img):
    r,c = img.shape
    width = int(img.shape[1])
    height = int(img.shape[0])
    coeffs = pywt.dwt2(img,'bior1.3')
    LL,(LH,HL,HH) = coeffs
    dsize = (width,height)
    output_LL = cv2.resize(LL[0:r//2,0:c//2],dsize)
    coeffs2 = pywt.dwt2(output_LL,'bior1.3')
    LL2,(LH2,HL2,HH2) = coeffs2
    output_LL2 = cv2.resize(LL2[0:r//2,0:c//2],dsize)
    # output_HL2 = cv2.resize(HL2[0:r//2,0:c//2],dsize)
    coeffs3 = pywt.dwt2(output_LL2, 'bior1.3')
    LL3,(LH3,HL3,HH3) = coeffs3
    output_HL3 = cv2.resize(HL3[0:r//2,0:c//2],dsize)
    return output_HL3


def frets_extraction(absGradX, kernel_lenght = 200, niterations=1):
    kernel_lenght = np.array(absGradX).shape[1]//kernel_lenght
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,kernel_lenght))
    img_vert = cv2.erode(absGradX,vertical_kernel,iterations=niterations)
    img_vert = cv2.dilate(img_vert, vertical_kernel, iterations=2)
    img_vert = cv2.medianBlur(np.uint8(img_vert)*255, 9)
    img_vert = skeletonize(img_vert, method = "lee")
    return img_vert

def strings_extraction(canny_edges,kernel_lenght = 150, niterations=1):
    kernel_lenght = np.array(canny_edges).shape[1]//kernel_lenght
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_lenght,1))
    im_horizontal = cv2.erode(canny_edges, horizontal_kernel,iterations=niterations)
    im_horizontal = cv2.dilate(im_horizontal, horizontal_kernel,iterations=1)
    im_horizontal = cv2.morphologyEx(im_horizontal,cv2.MORPH_CLOSE,horizontal_kernel,iterations=3)
    return im_horizontal

def vertical_proj_peaks_v2(im_vert,alpha = 0.5):
    im_hor_ver = im_vert 
    vertical_proj = np.sum(im_hor_ver,0)
    peaks_vert, _ = find_peaks(vertical_proj, prominence=alpha*np.max(np.abs(vertical_proj))) 
    return peaks_vert

def mask_pts(im_horizontal ,peaks_vert, threshold = 400,vmin=0,vmax=-1):
    linesP = Hough_linesP(im_horizontal,threshold)
    lines = np.zeros([len(linesP),1])
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        lines[i]=l[1]
    l = linesP[np.argmax(lines)][0]
    l2 = linesP[np.argmin(lines)][0]
    m1 = (l[3]-l[1])/(l[2]-l[0])
    b1 = l[1] - m1*l[0]

    m2 = (l2[3]-l2[1])/(l2[2]-l2[0])
    b2 = l2[1] - m2*l2[0]
    p1 = np.int0([peaks_vert[vmin],peaks_vert[vmin]*m1+b1])
    p2 = np.int0([peaks_vert[vmin],peaks_vert[vmin]*m2+b2])
    p3 = np.int0([peaks_vert[vmax],peaks_vert[vmax]*m2+b2])
    p4 = np.int0([peaks_vert[vmax],peaks_vert[vmax]*m1+b1])
    pts = np.array([p1,p2,p3,p4])
    m = np.array([m1,m2])
    b = np.array([b1,b2])
    return pts,m,b

def mask_pts_v2(im_horizontal ,peaks_vert, threshold = 400,vmin=0,vmax=-1):
    linesP = Hough_linesP(im_horizontal,threshold)
    points = np.zeros((len(linesP),4))
    for i in range(len(linesP)):
        points[i] = linesP[i][0]
    points.reshape(2*len(linesP),2)

    th = (np.max(points[:,1]) + np.min(points[:,1]))/2
    pts_1 = []
    pts_2 = []
    for i in range(len(linesP)):
        if points[i,1] >= th:
            pts_1.append(points[i])
        else:
            pts_2.append(points[i])
    pts_1 = np.array(pts_1)
    pts_2 = np.array(pts_2)

    x1 = np.zeros([len(pts_1),1])
    x2 = np.zeros([len(pts_2),1])
    y1 = pts_1[:,1]
    y2 = pts_2[:,1]
    for i  in range(len(pts_1)):
        x1[i] = pts_1[i,0]
    
    for i in range(len(pts_2)):
        x2[i] = pts_2[i,0]
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

def fretboard_mask(img, pts, guitar_angle):
    image_shape = img.shape
    polygon = np.array([pts[0][::-1],pts[1][::-1],pts[2][::-1],pts[3][::-1]])
    mask = np.uint8(1*polygon2mask(image_shape, polygon))
    mask = rotate_image(mask, -guitar_angle)
    return mask

def write_masks(f_path, filename, mask):
    directory = os.path.join(f_path, filename)
    cv2.imwrite(directory, mask)

def fretbord_correction(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    a = []
    b = []
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            a.append(x)
            b.append(y)
            cv2.circle(image, (x, y), 2, (0, 0, 255), thickness=-5)
            cv2.putText(image, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (255, 0, 0), thickness=1)
            cv2.imshow("image", image)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    c = np.array([[a[0],b[0]],[a[2],b[2]],[a[3],b[3]],[a[1],b[1]]])
    img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    rows,columns = img.shape

    polygon = np.array([c[0][::-1],c[1][::-1],c[2][::-1],c[3][::-1]])

    mask = np.uint8(1*polygon2mask(img.shape, polygon))
    return mask