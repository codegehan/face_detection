import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time

path = 'imageAttendance'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    currentImg = cv2.imread(f'{path}/{cl}')
    images.append(currentImg)
    classNames.append(os.path.splitext(cl)[0])
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
#attendance-here
def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtString}')
            
encodeListKnown = findEncodings(images)
print('Encoding complete')

#image coming from webcam
cap = cv2.VideoCapture(0)
previousTime = 0
while True:
    success, img = cap.read()
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,30), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0), 2)
    
    imgSmall = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)    
    
    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodeCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDistance)
        matchIndex = np.argmin(faceDistance)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc 
            y1,x2,y2,x1 =  y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.rectangle(img, (x1,y2-20), (x2,y2), (0,0,255), cv2.FILLED)
            cv2.putText(img, name,(x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255), 1)
            markAttendance(name)
            # Make a post request to send data to database
            # print(f'Data to send: {name}')
        else:
            name = "UNKNOWN"
            y1,x2,y2,x1 = faceLoc 
            y1,x2,y2,x1 =  y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.rectangle(img, (x1,y2-20), (x2,y2), (0,0,255), cv2.FILLED)
            cv2.putText(img, name,(x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255), 1)
            markAttendance(name)

    cv2.imshow('Camera', img)
    cv2.waitKey(1)