
import cv2
import numpy as np
import os
from datetime import datetime

face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

age_file = cv2.dnn.readNetFromCaffe(
    "age_deploy.prototxt",#import age_deploy.prototx file address
    "age_net.caffemodel"  #import age_net.caffemodel file address
)
AGE_LIST = ['(0-2)', '(4-6)', '(8-11)', '(15-20)', '(28-32)', '(38-43)', '(48-55)', '(60-80)', '(85-100)']

if not os.path.exists("faces"):
    os.makedirs("faces")

def load_existing_faces():
    face_encodings = []
    for fname in os.listdir("faces"):
        path = os.path.join("faces", fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        resized = cv2.resize(img, (100, 100))
        face_encodings.append(resized)
    return face_encodings

def is_new_face(face_img, existing_faces, threshold=2000):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (100, 100))
    for existing in existing_faces:
        diff = np.mean((resized.astype("float") - existing.astype("float")) ** 2)
        if diff < threshold:
            return False 
    return True 

cap = cv2.VideoCapture(0)
existing_faces = load_existing_faces()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        if is_new_face(face_img, existing_faces):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            face_filename = f"faces/face_{timestamp}.jpg"
            cv2.imwrite(face_filename, face_img)
            existing_faces.append(cv2.resize(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY), (100, 100)))

        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                     (78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False)
        age_file.setInput(blob)
        age_preds = age_file.forward()
        age_label = AGE_LIST[age_preds[0].argmax()]

        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile.detectMultiScale(roi_gray, 1.7, 20)
        smile_text = "smile" if len(smiles) > 0 else "without smile"

        label = f"Age: {age_label} | {smile_text}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 127), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 127), 2)

    cv2.imshow("power face recognition - exit with enter", frame)
    if cv2.waitKey(1) & 0xFF == ord('\r'):
        break
cap.release()
cv2.destroyAllWindows()

#i-power
