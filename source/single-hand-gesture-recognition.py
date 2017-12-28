import numpy as np
import cv2
import sys

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

cap = cv2.VideoCapture(0)
if not cap.isOpened:
    print "Webcam not connected. \n"
    sys.exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

found = False
res = None
blur = None
frmHsv = None
ticks = 0
notFoundCount = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    res = frame.copy()

    precTick = ticks
    ticks = cv2.getTickCount()
    dT = (ticks - precTick) / cv2.getTickFrequency()
    if found:
        kf.transitionMatrix[2] = dT
        kf.transitionMatrix[3] = dT
 
        print "dT: ", dT
 
        state = kf.predict()
        print "State post: "
        print state            
        
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
 
        if ratio > 0.75 and w * h >= 100000:
            hands.append(contours[i])
            handsBox.append([x, y, w, h])

    print "Hands found: ", len(handsBox)
    for i in range(len(hands)):
        cv2.drawContours(res, hands, i, (20,150,20), 1)
        cv2.rectangle(res, (handsBox[i][0], handsBox[i][1]), (handsBox[i][0] + handsBox[i][2], handsBox[i][1] + handsBox[i][3]), (0, 255, 0), 2)
        
        center = (int(handsBox[i][0] + handsBox[i][2] / 2.), int(handsBox[i][1] + handsBox[i][3] / 2.))
        cv2.circle(res, center, 2, (20, 150, 20), -1)

        sstr = "(" + str(center[0]) + ", " + str(center[1]) + ")"
        cv2.putText(res, sstr, (center[0] + 3, center[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 150, 20), 2)

    if len(hands) == 0:
        notFoundCount += 1
        print "notFoundCount: ", notFoundCount
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
        print "Measure matrix:"
        print meas
    
    cv2.imshow("Result", res)

    # frameHLS = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    # cv2.imshow('frame', frameHLS)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

                function process_delivery_access_code() {
                    let temp = jQuery('.field-multiple-drag');

                    for (let i = 0; i < temp.length; ++i){
                        let field_access_code = jQuery('input[name="field_delivery_good[und][' + i + '][field_access_code][und][0][value]"]')
                        let field_package = jQuery('select[name="field_delivery_good[und][' + i + '][field_package][und]"]');
                        let field_goods = jQuery('select[name="field_delivery_good[und][' + i + '][field_goods][und]"]');
                        let field_color = jQuery('select[name="field_delivery_good[und][' + i + '][field_color][und]"]');
                        let field_specification = jQuery('select[name="field_delivery_good[und][' + i + '][field_specification][und]"]');
                        let field_net_ctn = jQuery('select[name="field_delivery_good[und][' + i + '][field_net_ctn][und]"]');
                        let field_size = jQuery('select[name="field_delivery_good[und][' + i + '][field_size][und]"]');
                        let field_glazing = jQuery('select[name="field_delivery_good[und][' + i + '][field_glazing][und]"]');
                        let field_color_wire = jQuery('input[name="field_delivery_good[und][' + i + '][field_color_wire][und][0][value]"]');
                        let field_lot_no = jQuery('input[name="field_delivery_good[und][' + i + '][field_lot_no][und][0][value]"]');
                        let field_contract = jQuery('input[name="field_delivery_good[und][' + i + '][field_contract][und][0][value]"]');

                        field_package.change(function (){
                                    let access_code = field_access_code.val();
                                    let temp = field_package.find('option:selected').text();
                                    access_code = temp.slice(temp.length - 2, temp.length - 1) + access_code.slice(1, access_code.length);
                                    console.log(access_code);
                                    field_access_code.val(access_code);
                                });
                        field_goods.change(function (){
                                    let access_code = field_access_code.val();
                                    let temp = field_goods.find('option:selected').text();
                                    access_code = access_code.slice(0, 1)+ temp.slice(temp.length - 3, temp.length - 1) + access_code.slice(3, access_code.length);
                                    console.log(access_code);
                                    field_access_code.val(access_code);
                                });
                        field_color.change(function (){
                                    let access_code = field_access_code.val();
                                    let temp = field_color.find('option:selected').text();
                                    access_code = access_code.slice(0, 3) + temp.slice(temp.length - 3, temp.length - 1) + access_code.slice(5, access_code.length);
                                    console.log(access_code);
                                    field_access_code.val(access_code);
                                });
                        field_specification.change(function (){
                                    let access_code = field_access_code.val();
                                    let temp = field_specification.find('option:selected').text();
                                    access_code = access_code.slice(0, 5) + temp.slice(temp.length - 2, temp.length - 1) + access_code.slice(6, access_code.length);
                                    console.log(access_code);
                                    field_access_code.val(access_code);
                                });
                        field_net_ctn.change(function (){
                                    let access_code = field_access_code.val();
                                    let temp = field_net_ctn.find('option:selected').text();
                                    access_code = access_code.slice(0, 6) + temp.slice(temp.length - 5, temp.length - 1) + access_code.slice(10, access_code.length);
                                    console.log(access_code);
                                    field_access_code.val(access_code);
                                });
                        field_size.change(function (){
                                    let access_code = field_access_code.val();
                                    let temp = field_size.find('option:selected').text();
                                    access_code = access_code.slice(0, 10) +  temp.slice(temp.length - 9, temp.length - 1) + access_code.slice(18, access_code.length);
                                    console.log(access_code);
                                    field_access_code.val(access_code);
                                });
                        field_glazing.change(function (){
                                    let access_code = field_access_code.val();
                                    let temp = field_glazing.find('option:selected').text();
                                    access_code = access_code.slice(0, 18) +  temp.slice(temp.length - 3, temp.length - 1) + access_code.slice(20, access_code.length);
                                    console.log(access_code);
                                    field_access_code.val(access_code);
                                });
                        field_color_wire.change(function (){
                                    let access_code = field_access_code.val();
                                    let temp = field_color_wire.val();
                                    access_code = access_code.slice(0, 20) +  temp.slice(0,2) + access_code.slice(22, access_code.length);
                                    console.log(access_code);
                                    field_access_code.val(access_code);
                                });
                        field_lot_no.change(function (){
                                    let access_code = field_access_code.val();
                                    let temp = field_lot_no.val();
                                    access_code = access_code.slice(0, 22) +  temp.slice(0,3) + access_code.slice(25, access_code.length);
                                    console.log(access_code);
                                    field_access_code.val(access_code);
                                });
                        field_contract.change(function (){
                                    let access_code = field_access_code.val();
                                    let temp = field_contract.val();
                                    access_code = access_code.slice(0, 25) +  temp.slice(0,6) + access_code.slice(31, access_code.length);
                                    console.log(access_code);
                                    field_access_code.val(access_code);
                                });

                        field_access_code.on('input',function() {
                        
                            let access_code = jQuery(this).val();
                            
                            field_package.find('option').prop('selected', false);
                            field_goods.find('option').prop('selected', false);
                            field_color.find('option').prop('selected', false);
                            field_specification.find('option').prop('selected', false);
                            field_net_ctn.find('option').prop('selected', false);
                            field_size.find('option').prop('selected', false);
                            field_glazing.find('option').prop('selected', false);
                            field_color_wire.val("");
                            field_lot_no.val(""); 
                            field_contract.val("");
                        
                            if (access_code.length > 0)
                            {
                                let package = 'Mã đóng thùng:  ' + access_code[0];
                                field_package.find('option:contains("' + package + '")').prop('selected', true);      
                            }

                            if (access_code.length > 2)
                            {
                                let name = 'Mã hàng:  ' +  access_code[1]+access_code[2];
                                        field_goods.find('option:contains("' + name + '")').prop('selected', true);
                            }
                            if (access_code.length > 4)
                            {
                                let color = 'Mã màu:  ' + access_code[3] + access_code[4];
                                        field_color.find('option:contains("' + color + '")').prop('selected', true);	
                            }
                            if (access_code.length > 5)
                            {
                                let specification = 'Mã quy cách cấp đông:  ' +  access_code[5];
                                        field_specification.find('option:contains("' + specification + '")').prop('selected', true);
                            }
                            if (access_code.length > 9)
                            {
                                let net = 'Mã net:  ' + access_code.slice(6,9);
                                        field_net_ctn.find('option:contains("' + net + '")').prop('selected', true);
                            }
                            if (access_code.length > 17)
                            {
                                let size = 'Mã size:  ' +  access_code.slice(10,17);
                                        field_size.find('option:contains("' + size + '")').prop('selected', true);
                            }
                            if (access_code.length > 19)
                            {
                                let glazing ='Mã mạ băng:  ' +  access_code.slice(18,20);
                                        field_glazing.find('option:contains("' + glazing + '")').prop('selected', true);
                            }
                            if (access_code.length > 21)
                            {
                                let color_wire = access_code.slice(20,22);
                                        field_color_wire.val(color_wire);
                            }	
                            if (access_code.length > 24)
                                {
                                        let lot_no = access_code.slice(22,25);
                                        field_lot_no.val(lot_no);
                                }
                            if (access_code.length > 30)
                                {
                                        let contract = access_code.slice(25,31);
                                        field_contract.val(contract);
                                }		
                        });
                    }
                }

                process_delivery_access_code();
                jQuery(document).ajaxComplete(function() {
                    process_delivery_access_code();
                });