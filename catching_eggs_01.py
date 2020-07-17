# Catching Eggs - virtual game
#   - version with YOLO detection

import cv2
from catching_eggs_library import stackImages, detectObject, generatePerches, eggGenerator, eggInBasket

def bananaYolo():
    cam = cv2.VideoCapture(0)
    frameWidth = 640
    frameHeight = 480
    cam.set(3, frameHeight)
    cam.set(4, frameWidth)
    cam.set(10, 80)
    wht = 320
    confThresh = 0.5
    nmsThresh = 0.3

# define type of basket, e.g. cup, banana, car,... anything from coco.names
    thingName = 'banana'

    classesFile = "coco.names"
    classNames = []
    with open(classesFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

# better detection, slower framerate
    modelConfiguration = 'yolov3-320.cfg'
    modelWeights = 'yolov3-320.weights'
# worse detection, faster framerate
#     modelConfiguration = 'yolov3-tiny.cfg'
#     modelWeights = 'yolov3-tiny.weights'

    net = cv2.dnn.readNet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    myEggs = []
    scorePoints = 0

    while True:
        success, imgCamFlip = cam.read()
        imgCam = cv2.flip(imgCamFlip, 1)
        generatePerches(imgCam)

        blob = cv2.dnn.blobFromImage(imgCam, 1/255, (wht, wht), [0,0,0], 1, crop=False)
        net.setInput(blob)
        layerNames = net.getLayerNames()
        outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)

        objBox = detectObject(outputs, imgCam, classNames, confThresh, nmsThresh, thingName)
        myEggs = eggGenerator(imgCam, myEggs)
        for egg in myEggs:
            if eggInBasket(egg, objBox):
                scorePoints += 1
                myEggs.remove(egg)

        cv2.putText(imgCam, 'Score: '+str(scorePoints), (frameWidth-150, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        imgStack = stackImages(1.2, [[imgCam]])
        cv2.imshow('Catching Eggs - YOLO version', imgStack)

        keyPressed = cv2.waitKey(1)
        if keyPressed == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    bananaYolo()