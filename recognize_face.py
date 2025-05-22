import sys
import os
import pickle
import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
import csv
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QComboBox, QVBoxLayout, QHBoxLayout,
    QGroupBox, QFormLayout, QMessageBox
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtMultimedia import QCameraInfo

class RecognitionWindow(QWidget):
    def __init__(self, return_callback=None):
        super().__init__()
        self.return_callback = return_callback
        self.setWindowTitle("Real-time Face Recognition")
        self.setGeometry(120, 120, 900, 700)
        self._init_ui()
        self._init_logic()

    def _init_ui(self):
        # Setup panel
        input_group = QGroupBox("Setup")
        form = QFormLayout()

        self.camera_box = QComboBox()
        cameras = QCameraInfo.availableCameras()
        for idx, cam in enumerate(cameras):
            desc = cam.description()
            self.camera_box.addItem(f"{idx}: {desc}", idx)

        form.addRow("Camera:", self.camera_box)
        input_group.setLayout(form)

        # --- CSV logging setup ---
        self.csv_path = "recognitions.csv"
        # check if file exists to know whether to write header
        write_header = not os.path.exists(self.csv_path)
        self.csv_file = open(self.csv_path, 'a', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        if write_header:
            self.csv_writer.writerow(["timestamp", "name", "confidence_pct", "sunglasses"])


        # Control buttons
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Recognition")
        self.start_btn.clicked.connect(self.start_recognition)
        self.stop_btn = QPushButton("Stop Recognition")
        self.stop_btn.clicked.connect(self.stop_recognition)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)

        # Video display
        self.video_label = QLabel()
        self.video_label.setFixedSize(800, 600)
        self.video_label.setStyleSheet("background-color: #000; border: 1px solid #ccc;")
        self.video_label.setAlignment(Qt.AlignCenter)

        # Status label
        self.status_label = QLabel("Select camera and press 'Start Recognition'.")
        self.status_label.setFont(QFont("Arial", 10))
        self.status_label.setAlignment(Qt.AlignCenter)

        # Back to menu button
        self.back_btn = QPushButton("Back to Menu")
        self.back_btn.clicked.connect(self._go_back)

        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(input_group)
        layout.addLayout(btn_layout)
        layout.addWidget(self.video_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.back_btn)
        self.setLayout(layout)

    def _init_logic(self):
        # Timer for grabbing frames
        self.timer = QTimer()
        self.timer.timeout.connect(self._process_frame)
        self.cap = None
        # Load known faces
        pickle_file = "known_faces.pickle"
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
        else:
            data = {}
        self.known_encodings = []
        self.known_names = []
        for name, encs in data.items():
            for e in encs:
                self.known_encodings.append(e)
                self.known_names.append(name)
        # Sunglasses cascade
        self.glasses_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
        )
        # Constants
        self.BRIGHTNESS_THRESHOLD = 75
        self.CASCADE_NEIGHBORS = 4
        self.CASCADE_MIN_SIZE = (40, 40)

    def start_recognition(self):
        idx = self.camera_box.currentData()
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Camera Error", f"Cannot open camera index {idx}.")
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.camera_box.setEnabled(False)
        self.status_label.setText("Recognition started...")
        self.timer.start(30)

    def stop_recognition(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.clear()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.camera_box.setEnabled(True)
        self.status_label.setText("Recognition stopped.")
        if hasattr(self, 'csv_file') and not self.csv_file.closed:
            self.csv_file.close()

    def _process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.setText("Failed to grab frame.")
            return
        # Upscale frame
        frame = cv2.resize(frame, (0,0), fx=1.25, fy=1.25)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        inv = 1/0.8  
        table = np.array([((i/255.0)**inv)*255 for i in range(256)]).astype('uint8')
        frame = cv2.LUT(frame, table)

        # # —— CLAHE preprocessing—— 
        # lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        # l, a, b = cv2.split(lab)
        # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        # cl = clahe.apply(l)
        # lab_cl = cv2.merge((cl, a, b))
        # frame = cv2.cvtColor(lab_cl, cv2.COLOR_LAB2BGR)
        # # —— end CLAHE —— 
        # Face locations
        locs = face_recognition.face_locations(rgb, model='hog')
        try:
            encodings = face_recognition.face_encodings(rgb, locs)
        except Exception:
            return
        landmarks_list = face_recognition.face_landmarks(rgb, locs)

        for (top, right, bottom, left), encoding, landmarks in zip(locs, encodings, landmarks_list):
            wearing_sunglasses = False
            eye_pts = landmarks.get('left_eye', []) + landmarks.get('right_eye', [])
            if eye_pts:
                xs, ys = zip(*eye_pts)
                x1, x2 = max(min(xs)-10,0), min(max(xs)+10, frame.shape[1])
                y1, y2 = max(min(ys)-10,0), min(max(ys)+10, frame.shape[0])
                eye_roi = frame[y1:y2, x1:x2]
                if eye_roi.size:
                    gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
                    mean_b = np.mean(gray)
                    glasses = self.glasses_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=self.CASCADE_NEIGHBORS,
                        minSize=self.CASCADE_MIN_SIZE
                    )
                    
                    if len(glasses) > 0 or mean_b < self.BRIGHTNESS_THRESHOLD:
                        wearing_sunglasses = True
            # Conditional encoding
            if wearing_sunglasses and eye_pts:
                pil_img = Image.fromarray(rgb[top:bottom, left:right])
                draw = ImageDraw.Draw(pil_img)
                eye_local = [(x-left, y-top) for x,y in eye_pts]
                draw.polygon(eye_local, fill=(0,0,0))
                enc_list = face_recognition.face_encodings(np.array(pil_img))
                threshold = 0.85
            else:
                enc_list = [encoding]
                threshold = 0.8
            if enc_list:
                enc = enc_list[0]
                name = 'Unknown'
                confidence = 0.0
                if self.known_encodings:
                    dists = face_recognition.face_distance(self.known_encodings, enc)
                    best = np.argmin(dists)
                    if dists[best] < threshold:
                        name = self.known_names[best]
                        confidence = (1 - dists[best]/threshold)*100
                        if wearing_sunglasses:
                            confidence *= 0.85
            else:
                name, confidence = 'Unknown', 0.0

            # Draw box and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            label = f"{name}: {confidence:.0f}%"
            if wearing_sunglasses:
                label += " (sunglasses)"
            cv2.putText(frame, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            if wearing_sunglasses:
                msg = 'Please take off your sunglasses'
                (w,h),_ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (left, bottom+5), (left+w, bottom+5+h), (0,0,255), cv2.FILLED)
                cv2.putText(frame, msg, (left, bottom+5+h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            # --- log to CSV ---
            timestamp = datetime.now().isoformat()
            self.csv_writer.writerow([
                timestamp,
                name,
                f"{confidence:.1f}",
                1 if wearing_sunglasses else 0
            ])
            # ensure it's written out immediately
            self.csv_file.flush()


        # Display frame in Qt label
        rgb_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb_disp.shape
        bytes_pl = ch*w
        qt_img = QImage(rgb_disp.data, w, h, bytes_pl, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def _go_back(self):
        self.stop_recognition()
        self.hide()
        if self.return_callback:
            self.return_callback()

    def closeEvent(self, event):
        self.stop_recognition()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = RecognitionWindow()
    win.show()
    sys.exit(app.exec_())
