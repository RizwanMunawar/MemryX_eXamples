"""
Fire and Smoke Detection GUI
PyQt6 interface for real-time fire/smoke detection monitoring
"""

import time
from threading import Thread

import cv2
import numpy as np
from memryx import mxapi
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QSlider, QGroupBox,
                             QLineEdit)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap


class StreamSignals(QObject):
    """Signals for thread-safe GUI updates"""
    frame_ready = pyqtSignal(np.ndarray)
    detection_update = pyqtSignal(int, int)  # fire_count, smoke_count
    fps_update = pyqtSignal(float)
    alert = pyqtSignal(str, str, float)  # class_name, timestamp, score


class FireSmokeGUI(QMainWindow):
    """Main GUI window for Fire/Smoke Detection"""
    
    def __init__(self, app, dfp, post_model):
        super().__init__()
        self.setWindowTitle("Fire & Smoke Detection - MemryX MXA")
        self.setGeometry(100, 100, 1200, 800)
        self._is_closing = False
        
        # Set dark theme with fire accent colors
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
            QLabel {
                color: #eaeaea;
                font-size: 12px;
            }
            QPushButton {
                background-color: #e94560;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff6b6b;
            }
            QPushButton:pressed {
                background-color: #c73e54;
            }
            QGroupBox {
                border: 2px solid #3d3d5c;
                border-radius: 8px;
                margin-top: 12px;
                font-weight: bold;
                color: #eaeaea;
                padding: 10px;
                background-color: #16213e;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
                color: #e94560;
            }
            QLineEdit {
                background-color: #0f3460;
                color: #eaeaea;
                border: 1px solid #3d3d5c;
                padding: 6px;
                border-radius: 4px;
            }
            QSlider::groove:horizontal {
                background: #3d3d5c;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #e94560;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #ff6b6b;
            }
        """)
        
        # Signals for thread-safe updates
        self.signals = StreamSignals()
        self.signals.frame_ready.connect(self.update_frame)
        self.signals.detection_update.connect(self.update_detection_counts)
        self.signals.fps_update.connect(self.update_fps)
        self.signals.alert.connect(self.on_alert)
        
        self.app = app
        self.app.gui_callback = self.signals
        self.dfp = dfp
        self.post_model = post_model
        
        # Detection status with hold time
        self.fire_detected = False
        self.smoke_detected = False
        self.fire_last_detected_time = 0
        self.smoke_last_detected_time = 0
        self.status_hold_duration = 2.0  # Hold status for 2 seconds
        
        # Setup UI
        self.setup_ui()
        
        # Start accelerator thread
        self.accl_thread = Thread(target=self.run_accelerator, daemon=True)
        self.accl_thread.start()
        
    def setup_ui(self):
        """Create the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Left side - Video display
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        
        # Video group
        # Set title based on video source type
        feed_title = "Live Feed" if self.app.src_is_cam else "Video Feed"
        video_group = QGroupBox(feed_title)
        video_group_layout = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setScaledContents(False)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: #000000; 
            border: 3px solid #3d3d5c;
            border-radius: 8px;
        """)
        video_group_layout.addWidget(self.video_label)
        
        # Status bar below video
        status_bar = QHBoxLayout()
        
        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setStyleSheet("color: #00ff88; font-weight: bold; font-size: 16px;")
        
        self.fps_label = QLabel("Video FPS: 0.0")
        self.fps_label.setStyleSheet("color: #00aaff; font-weight: bold; font-size: 14px;")
        
        status_bar.addWidget(self.status_label)
        status_bar.addStretch()
        status_bar.addWidget(self.fps_label)
        
        video_group_layout.addLayout(status_bar)
        video_group.setLayout(video_group_layout)
        video_layout.addWidget(video_group)
        
        main_layout.addWidget(video_container, stretch=3)
        
        # Right side - Control panel
        control_panel = QWidget()
        control_panel.setMaximumWidth(320)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(15)
        
        # Title
        title = QLabel("🔥 Fire & Smoke\nDetection")
        title.setStyleSheet("""
            font-size: 24px; 
            font-weight: bold; 
            color: #e94560; 
            padding: 10px;
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(title)
        
        # Detection status
        status_group = QGroupBox("Detection Status")
        status_group_layout = QVBoxLayout()
        
        # Fire status
        fire_layout = QHBoxLayout()
        fire_icon = QLabel("🔥")
        fire_icon.setStyleSheet("font-size: 24px;")
        self.fire_status_label = QLabel("No Fire")
        self.fire_status_label.setStyleSheet("color: #888888; font-size: 18px; font-weight: bold;")
        fire_layout.addWidget(fire_icon)
        fire_layout.addWidget(self.fire_status_label)
        fire_layout.addStretch()
        status_group_layout.addLayout(fire_layout)
        
        # Smoke status
        smoke_layout = QHBoxLayout()
        smoke_icon = QLabel("💨")
        smoke_icon.setStyleSheet("font-size: 24px;")
        self.smoke_status_label = QLabel("No Smoke")
        self.smoke_status_label.setStyleSheet("color: #888888; font-size: 18px; font-weight: bold;")
        smoke_layout.addWidget(smoke_icon)
        smoke_layout.addWidget(self.smoke_status_label)
        smoke_layout.addStretch()
        status_group_layout.addLayout(smoke_layout)
        
        status_group.setLayout(status_group_layout)
        control_layout.addWidget(status_group)
        
        # Settings
        settings_group = QGroupBox("Detection Settings")
        settings_layout = QVBoxLayout()
        
        # Confidence slider
        conf_label_layout = QHBoxLayout()
        conf_label_layout.addWidget(QLabel("Confidence Threshold:"))
        self.conf_value = QLineEdit(f"{self.app.confidence_thres:.2f}")
        self.conf_value.setMaximumWidth(60)
        self.conf_value.setReadOnly(True)
        conf_label_layout.addWidget(self.conf_value)
        settings_layout.addLayout(conf_label_layout)
        
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setMinimum(5)
        self.conf_slider.setMaximum(95)
        self.conf_slider.setValue(int(self.app.confidence_thres * 100))
        self.conf_slider.valueChanged.connect(self.on_conf_change)
        settings_layout.addWidget(self.conf_slider)
        
        settings_group.setLayout(settings_layout)
        control_layout.addWidget(settings_group)
        
        # Alerts log
        alert_group = QGroupBox("Recent Alerts")
        alert_layout = QVBoxLayout()
        
        self.alert_log = QLabel("No alerts")
        self.alert_log.setWordWrap(True)
        self.alert_log.setStyleSheet("""
            color: #888888; 
            font-size: 11px; 
            padding: 10px;
            background-color: #0f3460;
            border-radius: 4px;
        """)
        self.alert_log.setMinimumHeight(120)
        self.alert_log.setAlignment(Qt.AlignmentFlag.AlignTop)
        alert_layout.addWidget(self.alert_log)
        
        alert_group.setLayout(alert_layout)
        control_layout.addWidget(alert_group)
        
        # Spacer
        control_layout.addStretch()
        
        # Exit button
        exit_btn = QPushButton("✕ Exit Application")
        exit_btn.setStyleSheet("""
            QPushButton {
                background-color: #cc3333;
                padding: 12px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #dd4444;
            }
        """)
        exit_btn.clicked.connect(self.close_application)
        control_layout.addWidget(exit_btn)
        
        main_layout.addWidget(control_panel, stretch=1)
    
    def on_conf_change(self, value):
        """Update confidence threshold"""
        self.app.confidence_thres = value / 100.0
        self.app.model.confidence_thres = value / 100.0
        self.conf_value.setText(f"{self.app.confidence_thres:.2f}")
    
    def update_frame(self, frame):
        """Update video display"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
    
    def update_fps(self, fps):
        """Update FPS display"""
        self.fps_label.setText(f"Video FPS: {fps:.1f}")
    
    def update_detection_counts(self, fire_count, smoke_count):
        """Update detection status with 2-second hold"""
        current_time = time.time()
        
        # Update fire status
        if fire_count > 0:
            self.fire_detected = True
            self.fire_last_detected_time = current_time
            self.fire_status_label.setText("Fire Detected!")
            self.fire_status_label.setStyleSheet("color: #ff4444; font-size: 18px; font-weight: bold;")
        elif current_time - self.fire_last_detected_time >= self.status_hold_duration:
            # Only clear status after hold duration has passed
            self.fire_detected = False
            self.fire_status_label.setText("No Fire")
            self.fire_status_label.setStyleSheet("color: #888888; font-size: 18px; font-weight: bold;")
        
        # Update smoke status
        if smoke_count > 0:
            self.smoke_detected = True
            self.smoke_last_detected_time = current_time
            self.smoke_status_label.setText("Smoke Detected!")
            self.smoke_status_label.setStyleSheet("color: #ffaa00; font-size: 18px; font-weight: bold;")
        elif current_time - self.smoke_last_detected_time >= self.status_hold_duration:
            # Only clear status after hold duration has passed
            self.smoke_detected = False
            self.smoke_status_label.setText("No Smoke")
            self.smoke_status_label.setStyleSheet("color: #888888; font-size: 18px; font-weight: bold;")
        
        # Update main status based on current detection state
        if self.fire_detected:
            self.status_label.setText("FIRE DETECTED!")
            self.status_label.setStyleSheet("color: #ff4444; font-weight: bold; font-size: 16px;")
        elif self.smoke_detected:
            self.status_label.setText("SMOKE DETECTED!")
            self.status_label.setStyleSheet("color: #ffaa00; font-weight: bold; font-size: 16px;")
        else:
            self.status_label.setText("Status: Monitoring...")
            self.status_label.setStyleSheet("color: #00ff88; font-weight: bold; font-size: 16px;")
    
    def on_alert(self, class_name, timestamp, score):
        """Handle detection alerts"""
        alert_text = f"[{timestamp}] {class_name.upper()}: {score:.2f}"
        
        current_log = self.alert_log.text()
        if current_log == "No alerts":
            self.alert_log.setText(alert_text)
        else:
            logs = current_log.split('\n')
            logs.insert(0, alert_text)
            self.alert_log.setText('\n'.join(logs[:8]))  # Keep last 8 alerts
        
        # Flash alert style
        if class_name == 'fire':
            self.alert_log.setStyleSheet("""
                color: #ff4444; 
                font-size: 11px; 
                font-weight: bold; 
                padding: 10px;
                background-color: #3a1010;
                border-radius: 4px;
            """)
        else:
            self.alert_log.setStyleSheet("""
                color: #ffaa00; 
                font-size: 11px; 
                font-weight: bold; 
                padding: 10px;
                background-color: #3a3010;
                border-radius: 4px;
            """)
    
    def run_accelerator(self):
        """Run the MXA accelerator in background thread"""
        try:
            if self.app.done:
                return

            if not self.app.display_thread.is_alive():
                self.app.display_thread.start()

            if self.app.done:
                return

            self.app.accl = mxapi.MxAccl(
                dfp_path=self.dfp,
                local_mode=True,
                use_model_shape=[False, False],
            )

            self.app.accl.connect_stream(
                self.app.capture_and_preprocess,
                self.app.postprocess,
                stream_id=0,
                model_id=0,
            )
            self.app.accl.start()
            self.app.accl.wait()
        except Exception as e:
            if not self.app.done:
                print(f"Accelerator error: {e}")
        finally:
            self.app.done = True

    def shutdown_app(self):
        """Stop background processing and wait briefly for workers to exit."""
        if self._is_closing:
            return
        self._is_closing = True

        self.app.gui_callback = None
        self.app.request_stop()

        if self.app.display_thread.is_alive():
            self.app.display_thread.join(timeout=1.0)
        if self.accl_thread.is_alive():
            self.accl_thread.join(timeout=1.0)
    
    def close_application(self):
        """Clean shutdown"""
        self.close()
    
    def closeEvent(self, event):
        """Handle window close"""
        self.shutdown_app()
        event.accept()
