import numpy as np
import cv2
import sys
from svm import *
import os

MIN_H_SKIN = (0, 10, 60)
MAX_H_SKIN = (20, 150, 255)

svm=cv2.ml.SVM_load('svm_data.dat')

stateSize = 6
measSize = 4
contrSize = 0
type_init = np.float32
type = cv2.CV_32F
bin_n=16
#http://answers.opencv.org/question/150999/opencv-32-python-svm-problem/
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

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("rtsp://192.168.1.254/sjcam.mov")
if not cap.isOpened:
    print ("Webcam not connected. \n")
    sys.exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

found = False
res = None
blur = None
frmHsv = None
ticks = 0
notFoundCount = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame is None:
    	continue
    res = frame.copy()
    precTick = ticks
    ticks = cv2.getTickCount()
    dT = (ticks - precTick) / cv2.getTickFrequency()
    if found:
        kf.transitionMatrix[2] = dT
        kf.transitionMatrix[3] = dT

        # print ("dT: ", dT)

        state = kf.predict()
        # print ("State post: ")
        # print (state)

        cv2.circle(res, (int(state[0][0]), int(state[1][0])), 2, (255, 0, 0), -1);
        cv2.rectangle(res, (int(state[0][0] - state[4][0] / 2), int(state[1][0] - state[5][0] / 2)), (int(state[0][0] + state[4][0] / 2), int(state[1][0] + state[5][0] / 2)), (255, 0, 0), 2)

    blur = cv2.GaussianBlur(frame, (5, 5), 3.0, 3.0)
    frmHsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    rangeRes = np.zeros(frame.shape, np.uint8)
    rangeRes = cv2.inRange(frmHsv, MIN_H_SKIN, MAX_H_SKIN)
    rangeRes = cv2.erode(rangeRes, None, iterations = 2)
    rangeRes = cv2.dilate(rangeRes, None, iterations = 2)
    cv2.imshow("Threshold", rangeRes)

    im2, contours, hierarchy = cv2.findContours(rangeRes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hands = []
    handsBox = []
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        ratio = float(w) / h
        if ratio > 1.0:
            ratio = 1.0 / ratio

        if ratio > 0.75 and w * h >= 90000:
            hands.append(contours[i])
            handsBox.append([x, y, w, h])
            #cv2.drawContours(im2, contours[i], -1, (0,255,0), 3)
            crop_img = rangeRes[y:y+h, x:x+w]
            crop_img = cv2.resize(crop_img,(500, 500), interpolation = cv2.INTER_CUBIC)
            cv2.imshow("cropped", crop_img)
            test=hog(crop_img)
            test1 = np.float32(test).reshape(-1,64)
            y=svm.predict(test1)
            os.system('cls')
            print ('-------------',y[1])


    # print ("Hands found: ", len(handsBox))
    for i in range(len(hands)):
        cv2.drawContours(res, hands, i, (20,150,20), 1)
        cv2.rectangle(res, (handsBox[i][0], handsBox[i][1]), (handsBox[i][0] + handsBox[i][2], handsBox[i][1] + handsBox[i][3]), (0, 255, 0), 2)

        center = (int(handsBox[i][0] + handsBox[i][2] / 2.), int(handsBox[i][1] + handsBox[i][3] / 2.))
        cv2.circle(res, center, 2, (20, 150, 20), -1)

        sstr = "(" + str(center[0]) + ", " + str(center[1]) + ")"
        cv2.putText(res, sstr, (center[0] + 3, center[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 150, 20), 2)

    if len(hands) == 0:
        notFoundCount += 1
        # print ("notFoundCount: ", notFoundCount)
        if notFoundCount >= 10:
            found = False
        else:
            kf.statePost = state
    else:
        notFoundCount = 0

        max_area = 0
        maxHandsBox = None
        for i in range(len(handsBox)):
            if max_area < handsBox[i][2] * handsBox[i][3]:
                maxHandsBox = handsBox[i]
                max_area = handsBox[i][2] * handsBox[i][3]

        meas[0][0] = maxHandsBox[0] + maxHandsBox[2] / 2.
        meas[1][0] = maxHandsBox[1] + maxHandsBox[3] / 2.
        meas[2][0] = maxHandsBox[2]
        meas[3][0] = maxHandsBox[3]

        if not found:
            kf.errorCovPre[0][0] = 1.
            kf.errorCovPre[1][1] = 1.
            kf.errorCovPre[2][2] = 1.
            kf.errorCovPre[3][3] = 1.
            kf.errorCovPre[4][5] = 1.
            kf.errorCovPre[5][5] = 1.

            state[0][0] = meas[0][0]
            state[1][0] = meas[1][0]
            state[2][0] = 0.
            state[3][0] = 0.
            state[4][0] = meas[2][0]
            state[5][0] = meas[3][0]

            kf.statePost = state

            found = True
        else:
            kf.correct(meas)
        # print ("Measure matrix:")
        # print (meas)

    cv2.imshow("Result", res)

    # frameHLS = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    # cv2.imshow('frame', frameHLS)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
