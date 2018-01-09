import numpy as np
import cv2
import sys
import math

SZ = 20
samples = []
num=1
trueValue=1
labels=[]
bin_n = 8

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
    bins = np.int32(bin_n*ang/(2*np.pi))
    bin_cells = bins[:8, :8], bins[:8, 8:16], bins[:8, 16:24], bins[:8, 24:], bins[8:16, :8], bins[8:16, 8:16], bins[8:16, 16:24], bins[8:16, 24:], bins[16:24, :8], bins[16:24, 8:16], bins[16:24, 16:24], bins[16:24, 24:], bins[24:, :8], bins[24:, 8:16], bins[24:, 16:24], bins[24:, 24:]
    mag_cells = mag[:8, :8], mag[:8, 8:16], mag[:8, 16:24], mag[:8, 24:], mag[8:16, :8], mag[8:16, 8:16], mag[8:16, 16:24], mag[8:16, 24:], mag[16:24, :8], mag[16:24, 8:16], mag[16:24, 16:24], mag[16:24, 24:], mag[24:, :8], mag[24:, 8:16], mag[24:, 16:24], mag[24:, 24:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    hist = hist / math.sqrt(sum(hist ** 2))
    return hist

while (num<=1000):
    MIN_H_SKIN = (0, 10, 60)
    MAX_H_SKIN = (20, 150, 255)

    if trueValue*200+1 == num:
        trueValue= trueValue+1

    name  = 'images/1001_'+str(trueValue)+'_'+str(num)+'.png'
    print (name)
    res = cv2.imread(name,0)
    im2, contours, hierarchy = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_area = -1
    max_area_index = -1
    for i in range(len(contours)):
        contour_area = cv2.contourArea(contours[i])
        if contour_area > max_area:
            max_area = contour_area
            max_area_index = i
    if max_area != -1:
        x, y, w, h = cv2.boundingRect(contours[max_area_index])
        crop_img = res[y:y+h, x:x+w]
        crop_img = cv2.resize(crop_img,(32, 32), interpolation = cv2.INTER_CUBIC)
        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)

    hist = hog(crop_img)
    samples.append(hist)
    labels.append(trueValue)
    num= num+1


samples = np.float32(samples)
labels = np.array(labels)

print ('samples',samples.size,labels.size)

svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)

svm.train(samples, cv2.ml.ROW_SAMPLE,labels)
svm.save('svm_data.dat')
