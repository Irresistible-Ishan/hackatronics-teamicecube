import sys
import os
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QScrollArea)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap
import cv2
from ultralytics import YOLO

# --- IMPORTANT: UPDATE THIS PATH ---
MODEL_PATH = 'runs/detect/pothole_detector_yolov8s/weights/best.pt'
# ------------------------------------

# --- Worker Signals ---
class WorkerSignals(QObject):
    new_frame = pyqtSignal(QImage)
    new_snapshot = pyqtSignal(QImage) # Signal for sending a snapshot with bounding box
    status_update = pyqtSignal(str, str)
    processing_finished = pyqtSignal()

# --- Video Processing Worker Thread ---
class VideoWorker(QThread):
    def __init__(self, video_path, model_path):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self.signals = WorkerSignals()
        self._is_running = True
        self.last_snapshot_time = 0

    def run(self):
        if not os.path.exists(self.model_path):
            self.signals.status_update.emit(f"Error: Model not found at {self.model_path}", "red")
            return
            
        model = YOLO(self.model_path)
        cap = cv2.VideoCapture(self.video_path)
        
        while cap.isOpened() and self._is_running:
            success, frame = cap.read()
            if not success:
                break

            current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            
            # Run YOLO detection
            results = model(frame, verbose=False)
            
            pothole_found_this_frame = False
            # Check for potholes and prepare a snapshot if needed
            snapshot_frame = None
            for result in results:
                if len(result.boxes) > 0:
                    snapshot_frame = frame.copy() # Make a copy for drawing
                    for box in result.boxes:
                        class_name = model.names[int(box.cls[0])]
                        if class_name == 'pothole':
                            pothole_found_this_frame = True
                            # Draw bounding box ON THE COPY
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = box.conf[0]
                            label = f"Pothole {confidence:.2f}"
                            cv2.rectangle(snapshot_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(snapshot_frame, label, (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Update status and send snapshot if a pothole was found
            if pothole_found_this_frame:
                self.signals.status_update.emit("Pothole Detected!", "red")
                # Throttle snapshots to one every 2 seconds to avoid flooding the UI
                if current_time_ms - self.last_snapshot_time > 2000:
                    self.last_snapshot_time = current_time_ms
                    # Convert the snapshot frame (with box) to QImage and emit
                    rgb_snapshot = cv2.cvtColor(snapshot_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_snapshot.shape
                    qt_snapshot = QImage(rgb_snapshot.data, w, h, ch * w, QImage.Format.Format_RGB888)
                    self.signals.new_snapshot.emit(qt_snapshot)
            else:
                self.signals.status_update.emit("Scanning...", "white")

            # Convert the ORIGINAL clean frame for UI display
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.signals.new_frame.emit(qt_image)

        cap.release()
        self.signals.processing_finished.emit()

    def stop(self):
        self._is_running = False

# --- Main Application Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pothole Detector")
        self.setGeometry(100, 100, 1200, 720) # Wider window
        self.apply_styles()

        # --- Main Layout (Horizontal) ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # --- Left Panel (Video Player) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.video_label = QLabel("Upload a video to begin testing")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFixedSize(800, 450)
        self.video_label.setStyleSheet("background-color: black; border-radius: 5px;")
        
        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.upload_button = QPushButton("Upload Video")
        self.upload_button.clicked.connect(self.open_video_file)

        left_layout.addWidget(self.video_label)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(self.upload_button)

        # --- Right Panel (Snapshots) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        title_label = QLabel("Detected Potholes")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.snapshot_layout = QVBoxLayout(self.scroll_content)
        self.snapshot_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.scroll_content)

        right_layout.addWidget(title_label)
        right_layout.addWidget(self.scroll_area)
        
        # --- Assemble Main Layout ---
        self.main_layout.addWidget(left_panel, 7) # 70% width
        self.main_layout.addWidget(right_panel, 3) # 30% width

        self.worker_thread = None

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget { background-color: #2c3e50; color: #ecf0f1; }
            QLabel { font-size: 16px; }
            QPushButton {
                background-color: #3498db; font-size: 16px; padding: 10px;
                border-radius: 5px; border: none;
            }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton:disabled { background-color: #555; }
            QScrollArea { border: 1px solid #34495e; }
        """)

    def open_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            self.clear_snapshots()
            self.upload_button.setEnabled(False)
            self.status_label.setText("Status: Starting...")
            
            self.worker_thread = VideoWorker(video_path=file_path, model_path=MODEL_PATH)
            self.worker_thread.signals.new_frame.connect(self.update_video_frame)
            self.worker_thread.signals.new_snapshot.connect(self.add_snapshot)
            self.worker_thread.signals.status_update.connect(self.update_status)
            self.worker_thread.signals.processing_finished.connect(self.on_processing_finished)
            self.worker_thread.start()

    def update_video_frame(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image).scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

    def update_status(self, message, color):
        self.status_label.setText(f"Status: {message}")
        self.status_label.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")

    def add_snapshot(self, image):
        snapshot_label = QLabel()
        snapshot_label.setPixmap(QPixmap.fromImage(image).scaledToWidth(300, Qt.TransformationMode.SmoothTransformation))
        snapshot_label.setStyleSheet("border: 1px solid #3498db; border-radius: 5px; margin-bottom: 5px;")
        self.snapshot_layout.addWidget(snapshot_label)
        
    def clear_snapshots(self):
        while self.snapshot_layout.count():
            item = self.snapshot_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def on_processing_finished(self):
        self.upload_button.setEnabled(True)
        self.status_label.setText("Status: Finished")
        self.status_label.setStyleSheet("color: #2ecc71; font-size: 16px; font-weight: bold;")
        if self.snapshot_layout.count() == 0:
            info_label = QLabel("No potholes were detected.")
            self.snapshot_layout.addWidget(info_label)

    def closeEvent(self, event):
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())