import cv2
from ultralytics import YOLO

threshold=int(input("Esik sayisini giriniz: "))
model=YOLO("yolov8n.pt")
cap=cv2.VideoCapture(0)
ok,first_frame=cap.read()
secilen_roi=cv2.selectROI("ROI seciniz",first_frame,False)
cv2.destroyWindow("ROI seciniz")
x, y, w, h = secilen_roi
roi_x1=x
roi_y1=y
roi_x2=x+w
roi_y2=y+h
while True:
    ok,frame=cap.read()
    if not ok:
        break
    classes=[0]
    results=model.track(frame,classes=classes,persist=True)
    count=len(results[0].boxes)
    annotated = results[0].plot()
    roi_count=0
    for box in results[0].boxes.xyxy:
        x1,y1,x2,y2=box[0],box[1],box[2],box[3]
        merkez_x=(x1+x2)/2
        merkez_y=(y1+y2)/2
        print("merkez_x:", merkez_x, "merkez_y:", merkez_y, "| roi_x1:", roi_x1, "roi_x2:", roi_x2, "roi_y1:", roi_y1, "roi_y2:", roi_y2)
        if roi_x1<merkez_x<roi_x2 and roi_y1<merkez_y<roi_y2:
            roi_count+=1
    if roi_count > threshold:
        renk = (0, 0, 255)   # kırmızı (BGR sırası: mavi=0, yeşil=0, kırmızı=255)
    else:
        renk = (0, 255, 0)   # yeşil
    cv2.rectangle(annotated,(roi_x1,roi_y1),(roi_x2,roi_y2),renk,2)
    if roi_count > threshold:
        print("UYARI: esik asildi!")
        cv2.putText(annotated, "UYARI!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        print("Kisi sayisi:", roi_count)
    cv2.imshow("pencere adi", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()