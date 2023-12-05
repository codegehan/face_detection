import cv2
import numpy as np
import face_recognition

imageGehan = face_recognition.load_image_file('images/2x2.jpg')
imageGehan = cv2.cvtColor(imageGehan, cv2.COLOR_BGR2RGB)
imageLiza = face_recognition.load_image_file('images/Francine.jpg')
imageLiza = cv2.cvtColor(imageLiza, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imageGehan)[0]
encodeGehan = face_recognition.face_encodings(imageGehan)[0]
cv2.rectangle(imageGehan,(faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 0, 255), 2)

faceLoc2 = face_recognition.face_locations(imageLiza)[0]
encodeLiza = face_recognition.face_encodings(imageLiza)[0]
cv2.rectangle(imageLiza,(faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 0, 255), 2)

results = face_recognition.compare_faces([encodeGehan], encodeLiza)
faceDistance = face_recognition.face_distance([encodeGehan], encodeLiza)
print(results, faceDistance)
cv2.putText(imageLiza, f'{results[0]} {round(faceDistance[0]), 2}', (50, 50), cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,255),2)

cv2.imshow('Gehan', imageGehan)
cv2.imshow('Liza', imageLiza)
cv2.waitKey(0)