import os

import cvzone
import cv2
#import numpy as np
#import matplotlib.pyplot as plt
import mediapipe as mp
from time import time
#from mediapipe.framework.formats import landmark_pb2 as mediapipe_landmarks

from cvzone.PoseModule import PoseDetector


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS,10)
pTime = 0
detector = PoseDetector()
mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=False, min_detection_confidence=0.3)


shirtFolderPath = "Resources/Shirts"
listShirts = os.listdir(shirtFolderPath)
# print(listShirts)
fixedRatio = 262 / 190  # widthOfShirt/widthOfPoint11to12
shirtRatioHeightWidth = 581 / 440
imageNumber = 0
imgButtonRight = cv2.imread("Resources/button.png", cv2.IMREAD_UNCHANGED)
imgButtonLeft = cv2.flip(imgButtonRight, 1)
counterRight = 0
counterLeft = 0
selectionSpeed = 10


while True:
    success, img = cap.read()

    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

    img = cv2.flip(img, 1)
    cv2.flip(img, 1)
    img_height, img_width, _ = img.shape
    img = cv2.resize(img, (int(img_width * (640 / img_height)), 640))


    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    print(results.pose_landmarks)
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
    print(id, lm)
    cx, cy = int(lm.x * w), int(lm.y * h)
    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    cTime = time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    #cv2.waitKey(1)
    img = detector.findPose(img)
    #img = cv2.flip(img,1)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
    if lmList:
        # center = bboxInfo["center"]
        lm11 = lmList[11][1:3]
        lm12 = lmList[12][1:3]
        imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)

        widthOfShirt = int((lm11[0] - lm12[0]) * fixedRatio)
        print(widthOfShirt)
        imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))
        currentScale = (lm11[0] - lm12[0]) / 190
        offset = int(44 * currentScale), int(48 * currentScale)

        try:
            img = cvzone.overlayPNG(img, imgShirt, (lm12[0] - offset[0], lm12[1] - offset[1]))
        except:
            pass

#        img = cvzone.overlayPNG(img, imgButtonRight, (1074, 293))
        img = cvzone.overlayPNG(img, imgButtonLeft, (72, 293))

        if lmList[16][1] < 300:
            counterRight += 1
            cv2.ellipse(img, (139, 360), (66, 66), 0, 0,
                        counterRight * selectionSpeed, (0, 255, 0), 20)
            if counterRight * selectionSpeed > 360:
                counterRight = 0
                if imageNumber < len(listShirts) - 1:
                    imageNumber += 1
        elif lmList[15][1] > 900:
            counterLeft += 1
            cv2.ellipse(img, (1138, 360), (66, 66), 0, 0,
                        counterLeft * selectionSpeed, (0, 255, 0), 20)
            if counterLeft * selectionSpeed > 360:
                counterLeft = 0
                if imageNumber > 0:
                    imageNumber -= 1

        else:
            counterRight = 0
            counterLeft = 0

    cv2.imshow("Image", img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()