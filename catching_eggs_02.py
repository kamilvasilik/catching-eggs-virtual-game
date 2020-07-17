# Catching Eggs - virtual game
#   - version with haarcascade

import cv2
from catching_eggs_library import stackImages, generatePerches, eggGenerator, eggInBasket

def bananaCascade():
    frameWidth = 640
    frameHeight = 480
    brightness = 80
    path = 'haarcascade/haarcascade_banana_03.xml'

    cam = cv2.VideoCapture(0)
    cam.set(3, frameWidth)
    cam.set(4, frameHeight)
    cam.set(10, brightness)

    myEggs = []
    scorePoints = 0

    cascade = cv2.CascadeClassifier(path)

    while True:
        success, imgCamFlip = cam.read()
        imgCam = cv2.flip(imgCamFlip, 1)
        imgGray = cv2.cvtColor(imgCam, cv2.COLOR_BGR2GRAY)
        generatePerches(imgCam)

        myEggs = eggGenerator(imgCam, myEggs)
        baskets = cascade.detectMultiScale(imgGray, 1.025, 16)
        for basket in baskets:
            (x,y,w,h) = basket
            area = w * h
            if area > 10000:
                cv2.rectangle(imgCam, (x,y), (x+w, y+h), (255, 0, 255), 2)

            for egg in myEggs:
                if eggInBasket(egg, basket):
                    scorePoints += 1
                    myEggs.remove(egg)

        cv2.putText(imgCam, 'Score: '+str(scorePoints), (frameWidth-150, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        imgStack = stackImages(1.2, [[imgCam]])
        cv2.imshow('Catching Eggs - cascade version', imgStack)
        keyPressed = cv2.waitKey(1)
        if keyPressed == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    bananaCascade()
