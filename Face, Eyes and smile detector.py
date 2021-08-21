# The program detect smiles

# import openCV
import cv2 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


def detect_smiles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    red, green, blue = ((255,0,0),(0,255,0),(0,0,255))
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, h, w) in faces:
        cv2.rectangle(frame, (x, y), (x+h, y+w), red)
        roi_gray = gray[x:x+h, y:y+w]
        roi_frame = frame[x:x+h, y:y+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (ex, ey, eh, ew) in eyes:
            cv2.rectangle(roi_frame, (ex, ey), (ex+eh, ey+ew), green, 3)
        for (sx, sy, sh, sw) in smiles:
            cv2.rectangle(roi_frame, (sx, sy), (sx+sh, sy+sw), blue, 3)
    return frame

video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    canvas = detect_smiles(frame)
    cv2.imshow('SMILE!', canvas)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()        