import tkinter as tk
from tkinter import Label, Button
import cv2
import threading
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model and labels
model = load_model("mask_detector_mobilenetv2.h5")
labels = ["with_mask", "without_mask", "mask_weared_incorrect"]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# GUI Class
class MaskDetectionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Face Mask Detection")
        self.video = cv2.VideoCapture(0)
        self.running = False

        self.label = Label(window)
        self.label.pack()

        self.start_btn = Button(window, text="Start", command=self.start)
        self.start_btn.pack()

        self.stop_btn = Button(window, text="Stop", command=self.stop)
        self.stop_btn.pack()

    def start(self):
        self.running = True
        self.update_frame()

    def stop(self):
        self.running = False
        self.video.release()
        self.window.quit()

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.video.read()
        if not ret:
            return

        # Face detection
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (224, 224))
            face_array = img_to_array(face_resized)
            face_array = preprocess_input(face_array)
            face_array = np.expand_dims(face_array, axis=0)

            pred = model.predict(face_array)[0]
            label_idx = np.argmax(pred)
            label = labels[label_idx]
            confidence = pred[label_idx]

            color = (0, 255, 0) if label == "with_mask" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Convert to ImageTk
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)

        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)
        self.window.after(10, self.update_frame)

# Run GUI
root = tk.Tk()
app = MaskDetectionApp(root)
root.mainloop()
