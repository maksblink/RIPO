import sys
import os
import pickle
import cv2
import face_recognition
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QComboBox, QLineEdit, QVBoxLayout, QHBoxLayout,
    QMessageBox, QGroupBox, QFormLayout
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtMultimedia import QCameraInfo

class FaceRememberingApp(QWidget):
    def __init__(self, return_callback=None):
        super().__init__()
        self.setWindowTitle("Face Remembering")
        self.setGeometry(100, 100, 800, 750)
        self.return_callback = return_callback
        self._init_ui()
        self._init_logic()

    def _init_ui(self):
        # Input group with camera index and names
        input_group = QGroupBox("Setup")
        form = QFormLayout()

        self.camera_box = QComboBox()
        # List available cameras with index and description
        cameras = QCameraInfo.availableCameras()
        for idx, cam_info in enumerate(cameras):
            desc = cam_info.description()
            self.camera_box.addItem(f"{idx}: {desc}", idx)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter person name")

        form.addRow("Camera:", self.camera_box)
        form.addRow("Name:", self.name_edit)
        input_group.setLayout(form)

        # Buttons
        btn_layout = QHBoxLayout()
        self.open_btn = QPushButton("Open Camera")
        self.open_btn.clicked.connect(self.start_capture)
        self.stop_btn = QPushButton("Stop Camera")
        self.stop_btn.clicked.connect(self.stop_capture)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.open_btn)
        btn_layout.addWidget(self.stop_btn)

        # Video display
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: #000; border: 1px solid #ccc;")
        self.video_label.setAlignment(Qt.AlignCenter)

        # Status
        self.status_label = QLabel("Select camera index and enter name, then click 'Open Camera'.")
        self.status_label.setFont(QFont("Arial", 10))
        self.status_label.setAlignment(Qt.AlignCenter)

        # Back to menu button
        self.back_btn = QPushButton("Back to Menu")
        self.back_btn.clicked.connect(self._go_back)

        # Overall layout
        layout = QVBoxLayout()
        layout.addWidget(input_group)
        layout.addLayout(btn_layout)
        layout.addWidget(self.video_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.back_btn)
        self.setLayout(layout)

    def _init_logic(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self._process_frame)
        self.cap = None
        # Backend settings
        self.sampling_interval = 10
        self.min_face_size = 100
        self.frame_count = 0
        self.output_folder = "extracted_faces"
        os.makedirs(self.output_folder, exist_ok=True)
        self.pickle_file = "known_faces.pickle"
        if os.path.exists(self.pickle_file):
            with open(self.pickle_file, 'rb') as f:
                self.known_faces = pickle.load(f)
        else:
            self.known_faces = {}

    def start_capture(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Input Error", "Please enter a name.")
            return
        self.person_name = name
        if self.person_name not in self.known_faces:
            self.known_faces[self.person_name] = []

        
        device_index = self.camera_box.currentData()
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Camera Error", f"Cannot open camera index {device_index}.")
            return

        # UI state
        self.open_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.camera_box.setEnabled(False)
        self.name_edit.setEnabled(False)
        self.status_label.setText(f"Capturing for '{self.person_name}' on camera {device_index}...")

        # Start
        self.timer.start(30)

    def stop_capture(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.clear()
        self.open_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.camera_box.setEnabled(True)
        self.name_edit.setEnabled(True)
        self.status_label.setText("Camera stopped.")
        # Save persistent data
        with open(self.pickle_file, 'wb') as f:
            pickle.dump(self.known_faces, f)

    def _process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.setText("Failed to grab frame.")
            return
        inv = 1/0.8  
        table = np.array([((i/255.0)**inv)*255 for i in range(256)]).astype('uint8')
        frame = cv2.LUT(frame, table)

        # # —— CLAHE preprocessing, exactly as in RecognitionWindow —— 
        # lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        # l, a, b = cv2.split(lab)
        # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        # cl = clahe.apply(l)
        # lab_cl = cv2.merge((cl, a, b))
        # frame = cv2.cvtColor(lab_cl, cv2.COLOR_LAB2BGR)
        # # —— end CLAHE —— 
        self.frame_count += 1
        display = frame.copy()

        if self.frame_count % self.sampling_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            saved = 0
            for idx, (top, right, bottom, left) in enumerate(locs):
                w, h = right-left, bottom-top
                if w < self.min_face_size or h < self.min_face_size:
                    continue
                crop = frame[top:bottom, left:right]
                fname = f"{self.person_name}_{self.frame_count}_{idx}.jpg"
                path = os.path.join(self.output_folder, fname)
                cv2.imwrite(path, crop)

                face_rgb = np.ascontiguousarray(rgb[top:bottom, left:right])
                landmarks = face_recognition.face_landmarks(face_rgb)
                if not landmarks:
                    continue
                encs = face_recognition.face_encodings(face_rgb)
                if encs:
                    self.known_faces[self.person_name].append(encs[0])
                    saved += 1
                cv2.rectangle(display, (left, top), (right, bottom), (0,255,0), 2)

            self.status_label.setText(
                f"Frame {self.frame_count}: found {len(locs)} face(s), saved {saved} encoding(s)."
            )

        # Show
        rgb_disp = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_disp.shape
        bytes_pl = ch * w
        qt_img = QImage(rgb_disp.data, w, h, bytes_pl, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def _go_back(self):
        # stop camera & timers
        self.stop_capture()
        # just hide this window
        self.hide()
        # show the main window again
        if self.return_callback:
            self.return_callback()


    def closeEvent(self, event):
        self.stop_capture()
        super().closeEvent(event)



