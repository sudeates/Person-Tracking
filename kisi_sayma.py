import cv2
from ultralytics import YOLO

threshold=int(input("Esik sayisini giriniz: "))
model=YOLO("yolov8n.pt")
cap=cv2.VideoCapture(0)

while True:
    ok,frame=cap.read()
    if not ok:
        break
    classes=[0]
    results=model.track(frame,classes=classes,persist=True)
    count=len(results[0].boxes)
    annotated = results[0].plot()
    if count > threshold:
        print("UYARI: esik asildi!")
        cv2.putText(annotated, "UYARI!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        print("Kisi sayisi:", count)
    cv2.imshow("pencere adi", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()