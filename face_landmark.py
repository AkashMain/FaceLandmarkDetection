import cv2 as cv  
import dlib  

cap = cv.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()  # alternative of haar_cascade

datfile = "C:\Code From VScode\Python\OpenCV\Shape predictor Face landmark data file\shape_predictor_68_face_landmarks.dat"

dlib_facelandmark = dlib.shape_predictor(datfile) # predict facelandmarks

while True:
    _,frame = cap.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)

    for face in faces:

        face_landmarks = dlib_facelandmark(gray,face)

        for n in range(0,68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            cv.circle(frame,(x,y),1,(0,255,0),1)

    cv.imshow('Face Landmarks',frame)
    key = cv.waitKey(1)
    if key ==27:
        break

cap.release()
cv.destroyAllWindows()
