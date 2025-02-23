from ultralytics import YOLO
import cv2
import os

detector = YOLO(r'best_detect.pt')

def detect(image):
    result = detector.predict(image, verbose=False)[0]
    boxes = result.boxes
    output = None
    highest_conf = 0.0

    if not boxes:
        return output
    for box in boxes.data:
        x1, y1, x2, y2 = map(int, box[:4])
        conf = float(box[4])
        if conf > highest_conf:
            highest_conf = conf
            output = [x1, y1, x2, y2, conf]
    return output

tien=1

output_folder = f'Data/train/{tien}'
os.makedirs(output_folder, exist_ok=True)

cap =cv2.VideoCapture(0)
count=0
while True:
    ret, frame=cap.read()
    frame=cv2.flip(frame,1)
    output=detect(frame)
    if output is not None:
        x1, y1, x2, y2, d_conf = output
        if d_conf>0.9:
            cash = frame[y1:y2, x1:x2]
            count+=1
            frame_path = os.path.join(output_folder, f"{tien}_frame_{count}.jpg")
            cv2.imwrite(frame_path, cash)
            print(f"Đã lưu frame {count} " )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("frame",frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
