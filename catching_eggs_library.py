import cv2
import numpy as np
import random

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    for x in range(0,rows):
        for y in range(0,len(imgArray[x])):
            if len(imgArray[x][y].shape)==2:
                imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
            if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0,0), fx=scale, fy=scale)
            else:
                imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), fx=scale, fy=scale)
            if y == 0:
                horiz = imgArray[x][y]
            else:
                horiz = np.hstack((horiz, imgArray[x][y]))
        if len(imgArray[x]) < cols and y < len(imgArray[x]):
            blank = np.zeros((imgArray[0][0].shape[0], imgArray[0][0].shape[1], 3), np.uint8)
            for i in range(0, cols - len(imgArray[x])):
                horiz = np.hstack((horiz, blank))
        if x == 0:
            vertic = horiz
        else:
            vertic = np.vstack((vertic, horiz))

    return vertic


def drawOnCanvas(imgRes, myPoints):
    for point in myPoints:
        cv2.circle(imgRes, (point[0], point[1]), 10, point[2], cv2.FILLED)


def generatePerches(img):
    imgW, imgH = img.shape[1], img.shape[0]
    lineOneA, lineOneB = (0, imgH // 4), (imgH // 4, 3 * imgH // 8)
    cv2.line(img, lineOneA, lineOneB, (255, 0, 255), 3)
    lineTwoA, lineTwoB = (0, imgH // 2), (imgH // 4, 5 * imgH // 8)
    cv2.line(img, lineTwoA, lineTwoB, (255, 0, 255), 3)
    lineThreeA, lineThreeB = (imgW, imgH // 4), (imgW - imgH // 4, 3 * imgH // 8)
    cv2.line(img, lineThreeA, lineThreeB, (255, 0, 255), 3)
    lineFourA, lineFourB = (imgW, imgH // 2), (imgW - imgH // 4, 5 * imgH // 8)
    cv2.line(img, lineFourA, lineFourB, (255, 0, 255), 3)


def generateEggs(img, eggType):
    imgW, imgH = img.shape[1], img.shape[0]
    colorEgg = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    if eggType == 1:
        egg = (15, imgH // 4 - 15, colorEgg, False)
    elif eggType == 2:
        egg = (15, imgH // 2 - 15, colorEgg, False)
    elif eggType == 3:
        egg = (imgW - 15, imgH // 4 - 15, colorEgg, True)
    elif eggType == 4:
        egg = (imgW - 15, imgH // 2 - 15, colorEgg, True)
    else:
        egg = None

    return egg


def moveEgg(img, egg):
    imgW, imgH = img.shape[1], img.shape[0]
    ratio = imgW / imgH
    (x, y, color, direction) = egg
    if direction:
        if y < imgH:
            if x > imgW  - (imgH // 4) - 50:
                egg = (int(x-10*ratio), int(y + 10/ratio), color, direction)
            else:
                egg = (x, int(y+10/ratio), color, direction)
    else:
        if y < imgH:
            if x < (imgH // 4) + 50:
                egg = (int(x+10*ratio), int(y+10/ratio), color, direction)
            else:
                egg = (x, int(y+10/ratio), color, direction)
    return egg


def eggGenerator(img, myPoints):
    imgH = img.shape[0]
    eggType = randomizer()
    if eggType != 0:
        egg = generateEggs(img, eggType)
        myPoints.append(egg)

    oldPoints = myPoints.copy()
    myPoints = []

    for egg in oldPoints:
        if egg[1] < imgH:
            egg = moveEgg(img, egg)
            myPoints.append(egg)

    if len(myPoints) != 0:
        drawOnCanvas(img, myPoints)

    return myPoints


def randomizer():
    if random.randint(0,100) > 90:
        return random.randint(1,4)
    else:
        return 0


def detectObject(outputs, img, classNames, confidThreshold, nmsThreshold, thing):
    ht, wt, ct = img.shape
    boundbox = []
    classIds = []
    confidVals = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confidThreshold:
                w, h = int(detection[2]*wt), int(detection[3]*ht)
                x, y = int((detection[0]*wt)-w/2), int((detection[1]*ht)-h/2)
                boundbox.append([x,y,w,h])
                classIds.append(classId)
                confidVals.append(float(confidence))

    x,y,w,h = 0, 0, 0, 0

    indices = cv2.dnn.NMSBoxes(boundbox, confidVals, confidThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        if classNames[classIds[i]] == thing:
            box = boundbox[i]
            x,y,w,h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 2)
            # cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confidVals[i]*100)}%',
            #             (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

    return (x,y,w,h)


def eggInBasket(egg, basket):
    (xe, ye, ce, dire) = egg
    (xb, yb, wb, hb) = basket
    ret = False
    if (xe > xb) and (xe < xb + wb) and (ye > yb) and (ye < yb + hb):
        ret = True
    return ret