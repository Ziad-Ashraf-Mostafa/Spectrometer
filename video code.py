import cv2

url = "http://192.168.137.190:8080/video"
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # your processing here
    cv2.imshow("frame", gray)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
