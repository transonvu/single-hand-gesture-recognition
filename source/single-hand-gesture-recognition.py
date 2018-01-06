import numpy as np
import cv2
import sys
import math

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

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

# def distance_euclid(point1, point2):
#     return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

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
        img = rangeRes[x:x + w, y:y + h]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0))

        cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]
        # First half is trainData, remaining is testData
        train_cells = [ i[:50] for i in cells ]
        test_cells = [ i[50:] for i in cells]
        deskewed = [map(deskew,row) for row in train_cells]
        hogdata = [map(hog,row) for row in deskewed]
        trainData = np.float32(hogdata).reshape(-1,64)
        responses = np.float32(np.repeat(np.arange(10),250)[:,np.newaxis])
        svm = cv2.ml.SVM_create()
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setC(2.67)
        svm.setGamma(5.383)
        svm.train(trainData, cv2.ml.ROW_SAMPLE, responses)
        svm.save('svm_data.dat')
    
    cv2.imshow('rangeRes', rangeRes)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# defects = cv2.convexityDefects(contours[max_area_index],hull)
# bottommost = tuple(contours[max_area_index][contours[max_area_index][:,:,1].argmax()][0])
# cv2.circle(frame, bottommost, 10, [255,0,0], -1)
# for i in range(defects.shape[0]):
#     s,e,f,d = defects[i,0]
#     start = tuple(contours[max_area_index][s][0])
#     end = tuple(contours[max_area_index][e][0])
#     far = tuple(contours[max_area_index][f][0])
#     y = 0
#     if end[1] < start[1]:
#         y = start[1]
#     else:
#         y = end[1]

#     if bottommost[1] - y <= 100:
#         if (distance_euclid(start, end) >= 100):
#             cv2.line(frame, start, far, [0,255,0], 2)
#             cv2.line(frame, far, end, [0,255,0], 2)

    # cv2.line(frame, start, end, [0,255,0], 2)
    # cv2.line(frame, start, far, [0,255,0], 2)
    # cv2.line(frame, far, end, [0,255,0], 2)
    # cv2.circle(frame, far, 5, [0,0,255], -1)

# hull = cv2.convexHull(contours[max_area_index])
# x, y, w, h = cv2.boundingRect(hull)
# cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0))
# cv2.drawContours(frame, contours, max_area_index, (0, 255, 0), 3)
# cv2.drawContours(frame, hull, -1, (0, 0, 255), 3)
