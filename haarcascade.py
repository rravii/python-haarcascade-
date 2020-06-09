import cv2 as cv

cap = cv.VideoCapture(0)
cap.set(10,100) # adjust brightness

faceCascade = cv.CascadeClassifier("haar-cascade/haarcascade_frontalface_default.xml")
eyeCascade = cv.CascadeClassifier("haar-cascade/haarcascade_eye.xml")
clockCascade = cv.CascadeClassifier("haar-cascade/haarcascade_wallclock.xml")
# fullCascade = cv.CascadeClassifier("haar-cascade/haarcascade_fullbody.xml")

while(cap.isOpened()):
    ret, frame = cap.read()
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # to channel 1
    faces = faceCascade.detectMultiScale(grey, 1.1, 4)  # scaleFactor, minNeighbors
    clock = clockCascade.detectMultiScale(grey, 1.1, 4)
    # fullbody = fullCascade.detectMultiScale(grey, 1.3, 5)

    for (x, y, w, h) in faces:
        cv.putText(frame, 'Face', (x+w, y+h), cv.FONT_HERSHEY_PLAIN, 2.5, (255, 0, 0), 2)
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        face_grey = grey[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(face_grey)
        for(ex, ey, ew, eh) in eyes:
            cv.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 3)

    for (x, y, w, h) in clock:
        cv.putText(frame, 'clock', (x + w, y + h), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)

    # for (x, y, w, h) in fullbody:
    #     cv.putText(frame, 'Body', (x + w, y + h), cv.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 0), 2)
    #     cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 5)

    output = cv.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))
    cv.imshow("Video", output)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
