# Catching Eggs - virtual game
#   - version with color detection

import cv2
import numpy as np
from catching_eggs_library import stackImages, generatePerches, eggGenerator, getBoundingBox, eggInBasket

def bananaColor():
    frameWidth = 640
    frameHeight = 480
    brightness = 80

    # yellow color mask in HSV format
    lowerMask = np.array([12, 93, 0])
    upperMask = np.array([45, 255, 255])

    cam = cv2.VideoCapture(0)
    cam.set(3, frameWidth)
    cam.set(4, frameHeight)
    cam.set(10, brightness)

    myEggs = []
    scorePoints = 0

    while True:
        success, imgCamFlip = cam.read()
        imgCam = cv2.flip(imgCamFlip, 1)
        generatePerches(imgCam)

        myEggs = eggGenerator(imgCam, myEggs)
        basket = getBoundingBox(imgCam, lowerMask, upperMask)
        cv2.rectangle(imgCam, (basket[0], basket[1]), (basket[0]+basket[2], basket[1]+basket[3]), (255,0,255), 2)
        for egg in myEggs:
            if eggInBasket(egg, basket):
                scorePoints += 1
                myEggs.remove(egg)

        cv2.putText(imgCam, 'Score: '+str(scorePoints), (frameWidth-150, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        imgStack = stackImages(1.2, [[imgCam]])
        cv2.imshow('Catching Eggs - color detection', imgStack)
        keypressed = cv2.waitKey(1)
        if keypressed == ord('q'):
            break


if __name__ == '__main__':
    bananaColor()