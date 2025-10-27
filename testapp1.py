import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QTextEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap
import cv2
from ultralytics import YOLO

# --- IMPORTANT: UPDATE THIS PATH ---
# This should be the exact path to your trained model file.
MODEL_PATH = 'runs/detect/pothole_detector_yolov8s/weights/best.pt'
# ------------------------------------

# This class handles signals from the worker thread to the main UI thread
class WorkerSignals(QObject):
    new_frame = pyqtSignal(QImage)
    status_update = pyqtSignal(str, str) # Message and color
    processing_finished = pyqtSignal(list)

# This is the worker thread that runs the YOLO model so the UI doesn't freeze
class VideoWorker(QThread):
    def __init__(self, video_path, model_path):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self.signals = WorkerSignals()
        self._is_running = True

    def run(self):
        if not os.path.exists(self.model_path):
            self.signals.status_update.emit(f"Error: Model not found at {self.model_path}", "red")
            return
            
        model = YOLO(self.model_path)
        cap = cv2.VideoCapture(self.video_path)
        
        detected_timestamps = []

        while cap.isOpened() and self._is_running:
            success, frame = cap.read()
            if not success:
                break

            # Run YOLO detection
            results = model(frame, verbose=False) # verbose=False keeps console clean
            
            pothole_found_this_frame = False
            # Check if any detected object is a 'pothole'
            for result in results:
                for box in result.boxes:
                    class_name = model.names[int(box.cls[0])]
                    if class_name == 'pothole':
                        pothole_found_this_frame = True
                        break
                if pothole_found_this_frame:
                    break

            # Update status and record timestamp
            if pothole_found_this_frame:
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp_sec = timestamp_ms / 1000.0
                detected_timestamps.append(f"{timestamp_sec:.2f}")
                self.signals.status_update.emit("Pothole Detected!", "red")
            else:
                self.signals.status_update.emit("Scanning...", "white")

            # Convert frame for UI display (BGR to RGB)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.signals.new_frame.emit(qt_image.scaledToWidth(800, Qt.TransformationMode.SmoothTransformation))

        cap.release()
        self.signals.processing_finished.emit(list(dict.fromkeys(detected_timestamps))) # Remove duplicates

    def stop(self):
        self._is_running = False

# The main application window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pothole Detector")
        self.setGeometry(100, 100, 840, 720)

        # --- UI Styling (CSS-like) ---
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2c3e50;
            }
            QLabel {
                color: #ecf0f1;
                font-size: 16px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                font-size: 16px;
                padding: 10px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QTextEdit {
                background-color: #34495e;
                color: #ecf0f1;
                border: 1px solid #2c3e50;
                font-family: Consolas, Courier New, monospace;
                font-size: 14px;
            }
        """)

        # --- Widgets ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.video_label = QLabel("Upload a video to begin testing")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFixedSize(800, 450)
        self.video_label.setStyleSheet("background-color: black; border-radius: 5px;")

        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFixedHeight(30)

        self.upload_button = QPushButton("Upload Video")
        self.upload_button.clicked.connect(self.open_video_file)

        self.results_box = QTextEdit()
        self.results_box.setReadOnly(True)
        self.results_box.setPlaceholderText("Detection timestamps will appear here...")
        
        # --- Layout ---
        self.layout.addWidget(self.video_label)
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.upload_button)
        self.layout.addWidget(self.results_box)

        self.worker_thread = None

    def open_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.results_box.clear()
            self.results_box.setPlaceholderText("Processing video...")
            self.status_label.setText("Status: Starting...")
            self.upload_button.setEnabled(False)
            
            self.worker_thread = VideoWorker(video_path=file_path, model_path=MODEL_PATH)
            self.worker_thread.signals.new_frame.connect(self.update_video_frame)
            self.worker_thread.signals.status_update.connect(self.update_status)
            self.worker_thread.signals.processing_finished.connect(self.on_processing_finished)
            self.worker_thread.start()

    def update_video_frame(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def update_status(self, message, color):
        self.status_label.setText(f"Status: {message}")
        self.status_label.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")

    def on_processing_finished(self, timestamps):
        self.upload_button.setEnabled(True)
        self.status_label.setText("Status: Finished")
        self.status_label.setStyleSheet("color: #2ecc71; font-size: 16px; font-weight: bold;")
        
        if not timestamps:
            self.results_box.setText("No potholes were detected for the whole video.")
        else:
            header = "Potholes Detected at the following timestamps (seconds):\n" + "="*55
            self.results_box.setText(header + "\n" + "\n".join(timestamps))

    def closeEvent(self, event):
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())