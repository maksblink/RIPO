import sys
import pickle

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QWidget, QVBoxLayout,
    QMessageBox, QLabel, QFrame, QHBoxLayout
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from FaceRemembering import FaceRememberingApp 
from recognize_face import RecognitionWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition Frontend")
        self.setGeometry(100, 100, 740, 440)
        self.center_on_screen()
        self.init_ui()
        self.setStyleSheet("""
            QMainWindow { background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                                     stop:0 #e3f0ff, stop:1 #c1d3fe);}
        """ )

    def center_on_screen(self):
        screen = QApplication.primaryScreen().availableGeometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )

    def init_ui(self):
        main_vbox = QVBoxLayout()
        main_vbox.setSpacing(24)
        main_vbox.setContentsMargins(60, 40, 60, 40)

        title = QLabel("Face Recognition System")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 30, QFont.Bold))
        title.setStyleSheet("color: #30548c; margin-bottom: 18px; letter-spacing: 1px;")
        main_vbox.addWidget(title)

        frame = QFrame()
        frame.setObjectName("menuCard")
        frame.setStyleSheet("""
            QFrame#menuCard {
                background: rgba(255,255,255,0.96);
                border-radius: 14px;
                border: 2px solid #aac7ff;
                padding: 36px 50px;
                min-width: 400px;
                max-width: 600px;
            }
        """ )

        vbox = QVBoxLayout(frame)
        vbox.setSpacing(22)
        vbox.setContentsMargins(20, 10, 20, 10)

        btn_style = """
            QPushButton {
                background-color: #4975e6;
                color: white;
                font-size: 19px;
                font-weight: 600;
                border-radius: 10px;
                padding: 15px 10px;
                margin-bottom: 5px;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background-color: #3562c8;
            }
        """

        btn_recognize = QPushButton("Start Face Recognition")
        btn_recognize.setStyleSheet(btn_style)
        btn_recognize.clicked.connect(self.open_recognition_window)
        vbox.addWidget(btn_recognize)

        btn_remember = QPushButton("Remember New Face")
        btn_remember.setStyleSheet(btn_style)
        btn_remember.clicked.connect(self.open_remembering_window)
        vbox.addWidget(btn_remember)

        btn_view = QPushButton("View Known Faces")
        btn_view.setStyleSheet(btn_style)
        btn_view.clicked.connect(self.show_known_faces)
        vbox.addWidget(btn_view)

        btn_exit = QPushButton("Exit")
        btn_exit.setStyleSheet("""
            QPushButton {
                background-color: #fa6868;
                color: white;
                font-size: 18px;
                font-weight: 600;
                border-radius: 10px;
                padding: 14px;
                margin-top: 16px;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background-color: #d64040;
            }
        """ )
        btn_exit.clicked.connect(self.close)
        vbox.addWidget(btn_exit)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(frame, stretch=2)
        hbox.addStretch(1)

        main_vbox.addLayout(hbox)

        container = QWidget()
        container.setLayout(main_vbox)
        self.setCentralWidget(container)

    def open_recognition_window(self):
        if not hasattr(self, 'recognition_window') or self.recognition_window is None:
            self.recognition_window = RecognitionWindow(return_callback=self.show)
        self.recognition_window.show()
        self.hide()

    def open_remembering_window(self):
        
        self.remembering_window = FaceRememberingApp(return_callback=self.show)
        self.remembering_window.show()
        self.hide()
    def show_known_faces(self):
        try:
            with open("known_faces.pickle", "rb") as f:
                known_faces = pickle.load(f)
            if known_faces:
                names = "\n".join(known_faces.keys())
            else:
                names = "No faces found."
            QMessageBox.information(self, "Known Faces", f"People:\n{names}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not load known faces:\n{e}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
