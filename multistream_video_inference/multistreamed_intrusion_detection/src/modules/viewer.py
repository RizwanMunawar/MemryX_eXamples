import cv2
from PySide6.QtWidgets import (QLabel, QWidget, QVBoxLayout)
from PySide6.QtGui import QImage, QPixmap, QMouseEvent
from PySide6.QtCore import Signal

# Viewer for video frames with ROI drawing capability
class FrameViewer(QWidget):
    mouse_move = Signal(tuple)
    mouse_click = Signal(tuple)
    roi_drawing_started = Signal(tuple)  # (x, y)
    roi_drawing_updated = Signal(tuple, tuple)  # (start, current)
    roi_drawing_finished = Signal(tuple, tuple)  # (start, end)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Viewer")

        # Create and configure the video display label
        self.video_label = VideoLabel(self)
        self.video_label.setMouseTracking(True)

        # Set a layout and add the video label to the widget
        layout = QVBoxLayout(self)
        layout.addWidget(self.video_label)

        self.current_frame = None
        self.roi_drawing = False
        self.roi_start = None
        self.roi_current = None

    def start_roi_drawing(self):
        """Enable ROI drawing mode"""
        self.roi_drawing = True
        self.roi_start = None
        self.roi_current = None

    def stop_roi_drawing(self):
        """Disable ROI drawing mode"""
        self.roi_drawing = False
        self.roi_start = None
        self.roi_current = None

    def update_frame(self, frame):
        self.current_frame = frame

        # Resize the frame to fit the available area for the video viewer while preserving the aspect ratio
        video_label_width = self.video_label.width()
        video_label_height = self.video_label.height()
        frame_height, frame_width, _ = frame.shape

        aspect_ratio = frame_width / frame_height
        if video_label_width / video_label_height > aspect_ratio:
            new_height = video_label_height
            new_width = int(aspect_ratio * new_height)
        else:
            new_width = video_label_width
            new_height = int(new_width / aspect_ratio)

        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        self.video_label.setMinimumSize(1, 1)

        # Draw ROI if drawing
        if self.roi_drawing and self.roi_start is not None and self.roi_current is not None:
            # Convert ROI coordinates to display coordinates
            display_start = self._frame_to_display_coords(self.roi_start, frame_width, frame_height, new_width, new_height)
            display_current = self._frame_to_display_coords(self.roi_current, frame_width, frame_height, new_width, new_height)
            if display_start and display_current:
                cv2.rectangle(frame, display_start, display_current, (0, 255, 0), 2)

        ## Get image information
        height, width, channels = frame.shape
        bytes_per_line = channels * width

        # Create QImage and display it
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def _frame_to_display_coords(self, frame_coords, frame_w, frame_h, display_w, display_h):
        """Convert frame coordinates to display coordinates"""
        if frame_coords is None:
            return None
        fx, fy = frame_coords
        scale_x = display_w / frame_w
        scale_y = display_h / frame_h
        return (int(fx * scale_x), int(fy * scale_y))

class VideoLabel(QLabel):
    def __init__(self, parent: FrameViewer):
        super().__init__(parent)
        self.setMouseTracking(True)
    
    def mousePressEvent(self, ev: QMouseEvent):
        viewer = self.parent()
        if viewer.current_frame is not None:
            pos = self._translate_mouse_coords(ev, viewer)
            if pos:
                if viewer.roi_drawing:
                    # Start ROI drawing
                    viewer.roi_start = pos
                    viewer.roi_current = pos
                    viewer.roi_drawing_started.emit(pos)
                else:
                    viewer.mouse_click.emit(pos)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent):
        viewer = self.parent()
        if viewer.current_frame is not None:
            pos = self._translate_mouse_coords(ev, viewer)
            if pos:
                if viewer.roi_drawing and viewer.roi_start is not None:
                    # Update ROI drawing
                    viewer.roi_current = pos
                    viewer.roi_drawing_updated.emit(viewer.roi_start, pos)
                else:
                    viewer.mouse_move.emit(pos)

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QMouseEvent):
        viewer = self.parent()
        if viewer.roi_drawing and viewer.roi_start is not None:
            pos = self._translate_mouse_coords(ev, viewer)
            if pos:
                viewer.roi_current = pos
                viewer.roi_drawing_finished.emit(viewer.roi_start, pos)
        super().mouseReleaseEvent(ev)

    def _translate_mouse_coords(self, ev, viewer):
        # Get the position of the mouse relative to the QLabel
        mouse_x = ev.position().x()
        mouse_y = ev.position().y()

        # Calculate the scaling factors
        video_label_width = self.width()
        video_label_height = self.height()
        frame_height, frame_width, _ = viewer.current_frame.shape

        aspect_ratio = frame_width / frame_height
        if video_label_width / video_label_height > aspect_ratio:
            new_height = video_label_height
            new_width = int(aspect_ratio * new_height)
        else:
            new_width = video_label_width
            new_height = int(new_width / aspect_ratio)

        x_offset = 0
        y_offset = (video_label_height - new_height) // 2

        # Adjust mouse position to account for the resized video and offset
        if x_offset <= mouse_x <= x_offset + new_width and y_offset <= mouse_y <= y_offset + new_height:
            adjusted_x = int((mouse_x - x_offset) * (frame_width / new_width))
            adjusted_y = int((mouse_y - y_offset) * (frame_height / new_height))
            mouse_position = (adjusted_x, adjusted_y)
        else:
            mouse_position = None

        return mouse_position

