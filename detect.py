from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\karti\runs\detect\runs\road_hazard_v2\weights\best.pt")

cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.30)

    annotated = results[0].plot()

    cv2.imshow("Road Guardian", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()