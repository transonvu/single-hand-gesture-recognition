import numpy as np
import cv2
import sys
import math
from svm import *

SZ = 20
bin_n = 16

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

svm = cv2.ml.SVM_load('svm_data.dat')
cap = cv2.VideoCapture(0)
if not cap.isOpened:
    print "Webcam not connected. \n"
    sys.exit()
    
def face_detection(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray_frame, minSize=(120, 120))
    max_area = -1
    max_area_index = None
    for (x, y, w, h) in faces:
        if w * h > max_area:
            max_area = w * h
            max_area_index = (x, y, w, h)
    if max_area_index is None:
        return None
    return list(max_area_index)

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed. 
        return img.copy()
    # Calculate skew based on central momemts. 
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness. 
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

while(True):
    ret, frame = cap.read()
    
    # Smoothing image
    blur = cv2.GaussianBlur(frame, (5, 5), 3.0, 3.0)

    # Covert hsv
    hsv_frame = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # Detect face
    face = face_detection(blur)
    min_skin = (0, 10, 60)
    max_skin = (20, 150, 255)

    rangeRes = cv2.inRange(hsv_frame, min_skin, max_skin)
    erode_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) 
    dilate_element = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6)) 
    rangeRes = cv2.erode(rangeRes, erode_element, iterations = 2)
    rangeRes = cv2.dilate(rangeRes, dilate_element, iterations = 2)    

    if not (face is None):
        rangeRes[face[1]:face[1] + face[3], face[0]:face[0] + face[2]] = 0

    im2, contours, hierarchy = cv2.findContours(rangeRes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_area = -1
    max_area_index = -1
    for i in range(len(contours)):
        contour_area = cv2.contourArea(contours[i])
        if contour_area > max_area:
            max_area = contour_area
            max_area_index = i
    if max_area >= 10000:
        x, y, w, h = cv2.boundingRect(contours[max_area_index])
        skin = cv2.bitwise_and(frame, frame, mask = rangeRes)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        crop_img = skin[y:y + h, x:x + w]
        gray_hand = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        gray_hand = cv2.resize(gray_hand,(500, 500), interpolation = cv2.INTER_CUBIC)
        cv2.imshow("cropped", gray_hand)
        hist = hog(gray_hand)
        feature_vector = np.float32(hist).reshape(-1,64)
        labels = svm.predict(feature_vector)
        print ('-------------', labels[1])

    cv2.imshow('rangeRes', rangeRes)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
