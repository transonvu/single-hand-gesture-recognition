import numpy as np
import cv2
import sys
from svmutil import *

samples = []
num=1
trueValue=1
labels=[]
bin_n = 16
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

while (num<=1000):
    MIN_H_SKIN = (0, 10, 60)
    MAX_H_SKIN = (20, 150, 255)

    stateSize = 6
    measSize = 4
    contrSize = 0
    type_init = np.float32
    type = cv2.CV_32F

    kf = cv2.KalmanFilter(stateSize, measSize, contrSize, type)
    state = np.zeros((stateSize, 1), type_init);
    meas = np.zeros((measSize, 1), type_init)
    cv2.setIdentity(kf.transitionMatrix)

    kf.measurementMatrix = np.zeros((measSize, stateSize), type_init)
    kf.measurementMatrix[0][0] = 1.0
    kf.measurementMatrix[1][1] = 1.0
    kf.measurementMatrix[2][4] = 1.0
    kf.measurementMatrix[3][5] = 1.0

    kf.processNoiseCov[0][0] = 1e-2
    kf.processNoiseCov[1][1] = 1e-2
    kf.processNoiseCov[2][2] = 2.0
    kf.processNoiseCov[3][3] = 1.0
    kf.processNoiseCov[4][5] = 1e-2
    kf.processNoiseCov[5][5] = 1e-2

    cv2.setIdentity(kf.measurementNoiseCov, (1e-1))

    found = False
    res = None
    blur = None
    frmHsv = None
    ticks = 0
    notFoundCount = 0
    if trueValue*200+1 == num:
        trueValue= trueValue+1
    name  = 'mytrain/1001_'+str(trueValue)+'_'+str(num)+'.png'
    print (name)
    res = cv2.imread(name,0)
    #cv2.imshow('image',res)
    precTick = ticks
    ticks = cv2.getTickCount()
    dT = (ticks - precTick) / cv2.getTickFrequency()
    res, contours, hierarchy = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hands = []
    handsBox = []

    x, y, w, h = cv2.boundingRect(contours[0])
    hands.append(contours[0])
    handsBox.append([x, y, w, h])
    #cv2.drawContours(im2, contours[i], -1, (0,255,0), 3)
    crop_img = res[y:y+h, x:x+w]
    crop_img = cv2.resize(crop_img,(500, 500), interpolation = cv2.INTER_CUBIC)
    thresh = 1
    crop_img = cv2.threshold(crop_img, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("cropped", crop_img)
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

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()

# When everything done, release the capture
cv2.destroyAllWindows()
