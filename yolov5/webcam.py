import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import cv2
from pathlib import Path

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

class YOLOv5GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.device = select_device('')
        self.model = DetectMultiBackend(weights=Path("yolov5s.pt"), device=self.device)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.model.warmup(imgsz=(1, 3, 640, 640))  # warmup
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30 ms per frame
        self.camera = cv2.VideoCapture(0)  # Initialize camera

    def initUI(self):
        self.setWindowTitle('YOLOv5 Object Detection')
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.video_label = QLabel(self)
        self.video_label.setGeometry(10, 10, 640, 480)

        self.start_button = QPushButton('Start Detection', self)
        self.start_button.setGeometry(660, 10, 120, 30)
        self.start_button.clicked.connect(self.start_detection)

    def start_detection(self):
        # Open file dialog to select an image or video
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Select Image/Video', '', 'Image Files (*.jpg *.png);;Video Files (*.mp4 *.avi)')
        if file_path:
            # Process the selected file
            self.process_file(file_path)

    def process_file(self, file_path):
        cap = cv2.VideoCapture(file_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Process frame for object detection
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
                self.display_image(frame)
            else:
                break
        cap.release()

    def update_frame(self):
        # Capture frame from camera or video stream for real-time detection
        ret, frame = self.camera.read()  # Read frame from camera
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            self.display_image(frame)

    def display_image(self, img):
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap)
        self.video_label.setScaledContents(True)

    def closeEvent(self, event):
        # Clean up resources when closing the application
        self.timer.stop()
        self.camera.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = YOLOv5GUI()
    window.show()
    sys.exit(app.exec_())
