"""
Compositor for drawing overlays on video frames
Adapted from realtime_multiface_recognition/src/modules/compositor.py
"""
import queue
import numpy as np
import cv2
from PySide6.QtCore import QTimer, QObject, Signal, Qt
from PySide6.QtWidgets import QCheckBox, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from .utils import Framerate, SliderWithLabel
from .roi_manager import ROIManager
from .face_tracker import TrackedFace


class MxColors: 
    LightBlue = (198, 234, 242)
    Blue = (53, 169, 188)
    DarkBlue = (20, 53, 85) 
    Teal = (60, 187, 187)
    Green = (0, 255, 0)
    Red = (0, 0, 255)
    Yellow = (0, 255, 255)


class Compositor(QObject):
    frame_ready = Signal(np.ndarray)

    def __init__(self, roi_manager: ROIManager, parent=None):
        super().__init__(parent)
        self.roi_manager = roi_manager
        self.framerate = Framerate()
        self.mouse_position = (-1, -1)
        self.paused = False
        self.tracked_faces: list[TrackedFace] = []
        self.alert_active = False

        # Create config widgets.
        self.bbox_checkbox = QCheckBox("Draw Boxes")
        self.bbox_checkbox.setChecked(True)
        self.label_checkbox = QCheckBox("Show Labels")
        self.label_checkbox.setChecked(True)
        self.roi_checkbox = QCheckBox("Show ROI")
        self.roi_checkbox.setChecked(True)

        self.label_scale_slider = SliderWithLabel("Font Scale:", 
                                                  minimum=50, 
                                                  maximum=300, 
                                                  initial=85, 
                                                  step=5, 
                                                  multiplier=0.01)
        self.label_thickness_slider = SliderWithLabel("Font Thickness:", 
                                                      minimum=1,
                                                      maximum=10, 
                                                      initial=2,
                                                      step=1, 
                                                      multiplier=1)
        self.line_thickness_slider = SliderWithLabel("Line Thickness:", 
                                                     minimum=1,
                                                     maximum=10, 
                                                     initial=2,
                                                     step=1, 
                                                     multiplier=1)

    def set_paused(self, paused: bool):
        self.paused = paused

    def update_mouse_pos(self, pos):
        self.mouse_position = pos

    def set_tracked_faces(self, faces: list[TrackedFace]):
        """Update tracked faces for drawing"""
        self.tracked_faces = faces

    def set_alert_active(self, active: bool):
        """Set alert state"""
        self.alert_active = active

    def draw_roi(self, frame):
        """Draw all ROI overlays"""
        if not self.roi_checkbox.isChecked():
            return frame
            
        if not self.roi_manager.has_roi():
            return frame
        
        # Convert frame to BGR for OpenCV drawing operations
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Draw all ROIs
        for roi in self.roi_manager.get_all_rois():
            if not roi.enabled:
                continue
                
            x1, y1, x2, y2 = roi.coordinates
            
            # Use red color if alert is active, otherwise use ROI's color
            # OpenCV drawing functions expect BGR colors: (Blue, Green, Red)
            if roi.alert_active:
                # Red in BGR format: (0, 0, 255) means B=0, G=0, R=255 = RED
                color = (0, 0, 255)  # Explicit red in BGR
                # Draw thicker border for alert
                border_thickness = 4
                # No overlay when alert is active - only border
                draw_overlay = False
            else:
                color = roi.color  # ROI's assigned color (already in BGR)
                border_thickness = 2
                draw_overlay = True
            
            # Draw semi-transparent overlay only when not in alert mode
            if draw_overlay:
                overlay = frame_bgr.copy()
                alpha = 0.3
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                frame_bgr = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)
            
            # Draw ROI border
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, border_thickness)
            
            # Draw ROI label or intrusion alert
            if roi.alert_active:
                label = f"{roi.name}: Intrusion Detected"
                font_scale = 0.8
                thickness = 3
            else:
                label = roi.name
                font_scale = 0.7
                thickness = 2
            
            cv2.putText(frame_bgr, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # Convert back to RGB
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                       
        return frame

    def draw_faces(self, frame):
        """Draw tracked faces with authorization status"""
        if not self.bbox_checkbox.isChecked():
            return frame
            
        for face in self.tracked_faces:
            if not face.activated:
                continue
                
            x1, y1, x2, y2 = face.bbox
            
            # Choose color and label based on ROI and authorization status
            if face.in_roi:
                # Inside ROI - check authorization status
                if face.authorized and face.name != "PERSON":
                    color = (0,255,0)
                    label = f"AUTHORIZED: {face.name} (ID: {face.track_id})"
                else:
                    # Unauthorized (name == "PERSON")
                    color = (255, 0, 0)
                    label = f"UNAUTHORIZED (ID: {face.track_id})"
            else:
                # Outside ROI - show as "PERSON"
                color = MxColors.Blue
                label = f"PERSON (ID: {face.track_id})"
                
            # Draw bounding box
            line_thickness = int(self.line_thickness_slider.value())
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)
            
            # Draw label
            if self.label_checkbox.isChecked():
                font_scale = self.label_scale_slider.value()
                thickness = int(self.label_thickness_slider.value())
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
                           
        return frame

    def draw_alert(self, frame):
        """Draw intrusion alert - now handled in draw_roi"""
        # Alert display is now integrated into ROI drawing
        return frame

    def draw(self, frame):
        """Main draw function"""
        self.framerate.update()
        frame = np.copy(frame)
        
        # Draw ROI
        frame = self.draw_roi(frame)
        
        # Draw tracked faces
        frame = self.draw_faces(frame)
        
        # Draw alert
        frame = self.draw_alert(frame)

        # If paused, dim and overlay "Paused: Click to Play"
        if self.paused:
            h, w = frame.shape[:2]
            # Dim the frame
            darkened = np.zeros_like(frame)
            frame = cv2.addWeighted(frame, 0.4, darkened, 0.6, 0)

            # Prepare text
            text = "Paused: Click to Play"
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Scale text relative to width
            font_scale = 1.5
            thickness = 3
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            text_x = (w - text_width) // 2
            text_y = (h + text_height) // 2

            # Draw outline for readability
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, MxColors.LightBlue, thickness, lineType=cv2.LINE_AA)

        self.frame_ready.emit(frame)


class CompositorConfigPopup(QDialog):
    def __init__(self, compositor, parent=None):
        super().__init__(parent)
        self.compositor = compositor
        self.setWindowTitle("Compositor Configuration")
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Add (and reparent) the compositor's configuration widgets.
        layout.addWidget(self.compositor.bbox_checkbox)
        layout.addWidget(self.compositor.label_checkbox)
        layout.addWidget(self.compositor.roi_checkbox)
        layout.addWidget(self.compositor.label_scale_slider)
        layout.addWidget(self.compositor.label_thickness_slider)
        layout.addWidget(self.compositor.line_thickness_slider)

        # Buttons for closing the popup.
        button_layout = QHBoxLayout()
        self.close_button = QPushButton("Close")
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)

        self.close_button.clicked.connect(self.close)

