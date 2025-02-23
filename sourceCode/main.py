from ultralytics import YOLO
import cv2
import speech_recognition as sr
from gtts import gTTS
import os
from pygame import mixer
import threading

detector = YOLO(r'best_detect.pt')
classifier = YOLO(r'best3.pt')

add=False 
total=0
            
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


def classify(image):
    result = classifier.predict(image, verbose = False)[0]
    idx = result.probs.top1
    conf = result.probs.top1conf.item()
    return classifier.names[idx], conf

def speak(txt):
    try:
        tts = gTTS(text=txt, lang='vi', slow=False)
        tts.save('noi.mp3')
        mixer.init()
        current_dir = os.path.dirname(os.path.realpath(__file__))  # Lấy đường dẫn của thư mục chứa mã nguồn
        mixer.music.load(os.path.join(current_dir, 'noi.mp3'))
        mixer.music.play()
        while mixer.music.get_busy():
            continue
        mixer.quit()
        os.remove("noi.mp3")
    except Exception as e:
        pass


def record():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Đang lắng nghe...")
        audio_data = recognizer.listen(source,phrase_time_limit=2)
        try:
            text = recognizer.recognize_google(audio_data, language="vi-VN")
            print("Bạn đã nói: " + text)
            return text
        except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
            print("Bạn đã không nói gì")
            return  record()
        
        
def main():
    count = 0
    prevTien = "None"
    tien = 0
    global total
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            break

        output = detect(frame)

        # Kiểm tra xem output có phải là None hay không
        if output is not None:
            x1, y1, x2, y2, d_conf = output
            
            if d_conf > 0.85:
                cash = frame[y1:y2, x1:x2]
                name, c_conf = classify(cash)

                if c_conf > 0.92:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{name} ({c_conf:.2f})"
                    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    tien = name

            else:
                tien=0
        else:
            tien = 0

        if prevTien == tien:
            count += 1
        else:
            count = 0
            prevTien = tien
        if count == 10 and tien != 0:
            if add:
                threading.Thread(target=speak, args=(f"{tien} nghìn,   ,cộng",)).start()
                total=total+int(tien)
            else:
                threading.Thread(target=speak, args=(f"Đây là {tien} nghìn",)).start()
        cv2.imshow("frame", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def listen_thread():
    global add, total
    while True:
        text=record()
        if "cộng" in text:
            speak(f"Bắt đầu cộng tiền")
            add=True
            
        
        if "dừng" in text:
            add=False
            total=total*1000
            speak(f"Tổng cộng tiền là {total} ")
            total=0


if __name__=="__main__":
    listen_thread = threading.Thread(target=listen_thread)
    listen_thread.daemon = True
    listen_thread.start()
    main()