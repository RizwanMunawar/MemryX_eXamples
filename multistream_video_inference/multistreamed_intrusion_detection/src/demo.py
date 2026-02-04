import os
import sys
import glob
import argparse
import cv2
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QWidget,
                               QVBoxLayout, QLineEdit, QPushButton,
                               QHBoxLayout, QSplitter, QCheckBox, QFrame,
                               QTreeWidget, QTreeWidgetItem, QInputDialog, QDialog,
                               QMessageBox, QFileDialog, QListWidget, QListWidgetItem,
                               QMenu)
from PySide6.QtGui import QImage, QPixmap, QMouseEvent, QKeyEvent
from PySide6.QtCore import QTimer, Qt, QThread, Signal, QMutex

import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.capture import CaptureConfigDialog, VIDEO_CONFIG
from modules.compositor import CompositorConfigPopup
from modules.grid_viewer import GridViewer
from modules.authorization_db import AuthorizationDatabase, DatabaseViewerWidget
from modules.multistream_roi_manager import MultiStreamROIManager
from modules.multistream_face_recognizer import MultiStreamFaceRecognizer
from modules.multistream_face_tracker import MultiStreamFaceTracker
from modules.multistream_intrusion_detector import MultiStreamIntrusionDetector
from modules.multistream_capture import MultiStreamCapture
from modules.multistream_compositor import MultiStreamCompositor


class Demo(QMainWindow):
    def __init__(self, video_paths, video_config=None):
        super().__init__()
        self.setWindowTitle("Enhanced Intrusion Detection System - Multi-Stream")
        
        self.num_streams = len(video_paths)
        
        # Create the grid video display
        self.viewer = GridViewer(self.num_streams)

        # Set up video capture and processing
        self.capture_thread = MultiStreamCapture(video_paths, video_config)
        self.auth_database = AuthorizationDatabase()
        self.database_viewer = DatabaseViewerWidget(self.auth_database)
        self.roi_manager = MultiStreamROIManager(self.num_streams)
        
        # Initialize models (paths should be set correctly)
        models_dir = Path(__file__).parent.parent / 'models'
        face_recognizer = MultiStreamFaceRecognizer(models_dir, self.num_streams)
        
        # Create face tracker with ROI manager and auth database
        face_tracker = MultiStreamFaceTracker(face_recognizer, self.auth_database, self.roi_manager)
        face_tracker.start()
        
        self.intrusion_detector = MultiStreamIntrusionDetector(
            face_tracker,
            self.roi_manager,
            self.auth_database
        )
        
        # Store face tracker for access in other methods
        self.face_tracker = face_tracker
        
        self.compositor = MultiStreamCompositor(self.roi_manager)
        self.compositor.set_paused(self.capture_thread.pause, stream_idx=None)

        # Create buttons
        self.config_popup_button = QPushButton("Compositor Config", self)
        self.config_popup_button.clicked.connect(self.open_compositor_config)

        self.capture_control_button = QPushButton("Capture Config", self)
        self.capture_control_button.clicked.connect(self.open_capture_config)
        
        self.roi_draw_button = QPushButton("Draw ROI", self)
        self.roi_draw_button.clicked.connect(self.toggle_roi_drawing)
        self.roi_drawing_mode = False
        self.active_stream_idx = None  # Track which stream is active for ROI drawing
        
        self.roi_clear_button = QPushButton("Clear Selected ROI", self)
        self.roi_clear_button.clicked.connect(self.clear_selected_roi)
        
        self.roi_clear_all_button = QPushButton("Clear All ROIs", self)
        self.roi_clear_all_button.clicked.connect(self.clear_all_rois)
        
        # ROI list widget
        self.roi_list = QListWidget(self)
        self.roi_list.itemDoubleClicked.connect(self.rename_roi)
        self.roi_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.roi_list.customContextMenuRequested.connect(self.show_roi_context_menu)
        self.roi_list.itemSelectionChanged.connect(self.on_roi_selection_changed)
        self.selected_roi_info = None  # (stream_idx, roi_id)

        # Wire signals
        self.capture_thread.frame_ready.connect(self.process_frame)
        self.compositor.frame_ready.connect(self.viewer.update_frame)
        self.viewer.mouse_move.connect(self.compositor.update_mouse_pos)
        self.viewer.mouse_click.connect(self.handle_viewer_mouse_click)
        self.viewer.roi_drawing_finished.connect(self.handle_roi_finished)
        
        # Connect face tracker frame ready signal
        self.face_tracker.frame_ready.connect(self.process_face_detection)
        
        # Add click-to-pause functionality
        self.viewer.mouse_click.connect(self.toggle_capture_pause)

        self.setup_layout()

        self.capture_thread.start()
        # Create a persistent instance for the config popup
        self.config_popup = None
        
        # Timer for processing frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.poll_framerates)
        self.timer.start(33)  # ~30 fps
        
        # Initialize ROI list
        self.update_roi_list()

    def setup_layout(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setMinimumSize(300, 200)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Splitter to separate control panel and video viewer
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)

        # Left control panel
        self.control_panel = QWidget()
        self.control_panel.setFixedWidth(300)
        self.control_layout = QVBoxLayout(self.control_panel)
        self.splitter.addWidget(self.control_panel)

        # Add buttons
        self.control_layout.addWidget(self.capture_control_button)
        self.control_layout.addWidget(self.config_popup_button)
        self.control_layout.addWidget(self.roi_draw_button)
        self.control_layout.addWidget(self.roi_clear_button)
        self.control_layout.addWidget(self.roi_clear_all_button)
        
        # Add ROI list
        roi_label = QLabel("ROIs:")
        self.control_layout.addWidget(roi_label)
        self.control_layout.addWidget(self.roi_list)
        
        self.control_layout.addWidget(self.database_viewer)

        # Right: Grid viewer
        self.splitter.addWidget(self.viewer)
        self.splitter.setStretchFactor(1, 1)

    def open_compositor_config(self):
        # For multi-stream, we'll use the first stream's compositor for config
        # (all compositors share the same config widgets)
        if self.config_popup is None:
            compositor = self.compositor.get_compositor(0)
            if compositor:
                self.config_popup = CompositorConfigPopup(compositor, self)
        if self.config_popup:
            self.config_popup.show()
            self.config_popup.raise_()

    def open_capture_config(self):
        # Capture config is global for all streams
        current_resolution = "2k"
        if self.capture_thread.video_config:
            for res in ["1080p", "2k", "4k"]:
                if VIDEO_CONFIG.get(res) == self.capture_thread.video_config:
                    current_resolution = res
                    break

        # Use first video path for dialog
        current_video_path = self.capture_thread.video_paths[0] if self.capture_thread.video_paths else '/dev/video0'
        dialog = CaptureConfigDialog(current_video_path, current_resolution, self)
        if dialog.exec() == QDialog.Accepted:
            new_video_path, new_resolution = dialog.get_configuration()
            print(f"Note: Capture config change applies to all streams. New config: {new_resolution}")
            # Note: For simplicity, we're not changing video paths at runtime
            # This would require stopping/restarting all capture threads

    def toggle_roi_drawing(self):
        """Toggle ROI drawing mode"""
        self.roi_drawing_mode = not self.roi_drawing_mode
        if self.roi_drawing_mode:
            self.viewer.start_roi_drawing()  # Enable on all streams
            self.roi_draw_button.setText("Stop Drawing ROI")
        else:
            self.viewer.stop_roi_drawing()
            self.roi_draw_button.setText("Draw ROI")
            self.active_stream_idx = None

    def clear_selected_roi(self):
        """Clear the selected ROI"""
        if self.selected_roi_info is not None:
            stream_idx, roi_id = self.selected_roi_info
            self.roi_manager.remove_roi(stream_idx, roi_id)
            self.update_roi_list()
            self.selected_roi_info = None
            print(f"ROI {roi_id} from stream {stream_idx} cleared")
        else:
            QMessageBox.information(self, "No Selection", "Please select an ROI from the list to clear.")

    def clear_all_rois(self):
        """Clear all ROIs"""
        reply = QMessageBox.question(self, "Clear All ROIs", 
                                     "Are you sure you want to clear all ROIs from all streams?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.roi_manager.clear_all_rois()
            self.update_roi_list()
            self.selected_roi_info = None
            print("All ROIs cleared")

    def handle_roi_finished(self, stream_idx, start, end):
        """Handle ROI drawing completion"""
        x1, y1 = start
        x2, y2 = end
        
        # Show dialog to name the ROI
        name, ok = QInputDialog.getText(self, 'Name ROI', f'Enter name for ROI on Stream {stream_idx}:')
        if not ok or not name.strip():
            name = None  # Will use default name
        
        # Add new ROI to the stream where it was drawn
        roi_id = self.roi_manager.add_roi(stream_idx, x1, y1, x2, y2, name)
        self.update_roi_list()
        print(f"ROI added to Stream {stream_idx}: ID={roi_id}, name={name or 'default'}, ({x1}, {y1}) to ({x2}, {y2})")
        
        self.roi_drawing_mode = False
        self.viewer.stop_roi_drawing()
        self.roi_draw_button.setText("Draw ROI")
        self.active_stream_idx = None
    
    def update_roi_list(self):
        """Update the ROI list widget with stream:ROI format"""
        self.roi_list.clear()
        for stream_idx in range(self.num_streams):
            rois = self.roi_manager.get_all_rois(stream_idx)
            for roi in rois:
                item = QListWidgetItem(f"Stream {stream_idx}: {roi.name}")
                item.setData(Qt.UserRole, (stream_idx, roi.roi_id))
                self.roi_list.addItem(item)
    
    def rename_roi(self, item):
        """Rename an ROI (double-click)"""
        stream_idx, roi_id = item.data(Qt.UserRole)
        roi = self.roi_manager.get_roi(stream_idx, roi_id)
        if roi:
            new_name, ok = QInputDialog.getText(self, 'Rename ROI', 
                                                f'Enter new name for Stream {stream_idx}: {roi.name}:',
                                                text=roi.name)
            if ok and new_name.strip():
                roi.name = new_name.strip()
                self.update_roi_list()
                print(f"ROI {roi_id} on Stream {stream_idx} renamed to {new_name}")
    
    def show_roi_context_menu(self, position):
        """Show context menu for ROI list"""
        item = self.roi_list.itemAt(position)
        if item is None:
            return
        
        stream_idx, roi_id = item.data(Qt.UserRole)
        menu = QMenu(self)
        
        rename_action = menu.addAction("Rename")
        delete_action = menu.addAction("Delete")
        
        action = menu.exec_(self.roi_list.mapToGlobal(position))
        
        if action == rename_action:
            self.rename_roi(item)
        elif action == delete_action:
            self.delete_roi(stream_idx, roi_id)
    
    def delete_roi(self, stream_idx, roi_id):
        """Delete an ROI"""
        roi = self.roi_manager.get_roi(stream_idx, roi_id)
        if roi:
            reply = QMessageBox.question(self, "Delete ROI", 
                                        f"Are you sure you want to delete 'Stream {stream_idx}: {roi.name}'?",
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.roi_manager.remove_roi(stream_idx, roi_id)
                self.update_roi_list()
                if self.selected_roi_info == (stream_idx, roi_id):
                    self.selected_roi_info = None
                print(f"ROI {roi_id} ({roi.name}) from Stream {stream_idx} deleted")
    
    def on_roi_selection_changed(self):
        """Handle ROI list selection change"""
        current_item = self.roi_list.currentItem()
        if current_item:
            self.selected_roi_info = current_item.data(Qt.UserRole)
        else:
            self.selected_roi_info = None

    def handle_viewer_mouse_click(self, stream_idx, mouse_pos):
        """Handle mouse click on viewer - prioritize face clicks over ROI selection"""
        if mouse_pos is None:
            return
        
        if self.roi_drawing_mode:
            self.active_stream_idx = stream_idx
            return  # ROI drawing is handled separately
        
        mouse_x, mouse_y = mouse_pos
        
        # PRIORITY 1: Check for face clicks first (including faces inside ROIs)
        tracked_faces = self.face_tracker.get_activated_tracker_objects(stream_idx)
        for face in tracked_faces:
            (left, top, right, bottom) = face.bbox
            if left <= mouse_x <= right and top <= mouse_y <= bottom:
                # Clicked on a face - show authorization dialog
                self.show_authorization_dialog(stream_idx, face)
                return  # Face click takes priority
        
        # PRIORITY 2: If no face clicked, check for ROI selection
        clicked_roi = self.roi_manager.get_roi_at_point(stream_idx, mouse_x, mouse_y)
        if clicked_roi:
            # Select this ROI in the list
            for i in range(self.roi_list.count()):
                item = self.roi_list.item(i)
                item_stream_idx, item_roi_id = item.data(Qt.UserRole)
                if item_stream_idx == stream_idx and item_roi_id == clicked_roi.roi_id:
                    self.roi_list.setCurrentItem(item)
                    break
            return

    def show_authorization_dialog(self, stream_idx, face):
        """Show dialog to select Authorized or Unauthorized"""
        options = ["Authorized", "Unauthorized"]
        
        # Get current index
        current_index = 0 if face.authorized else 1
        
        choice, ok = QInputDialog.getItem(
            self, 
            'Set Authorization Status',
            f'Select authorization status for Face ID {face.track_id} (Stream {stream_idx}):',
            options,
            current_index,
            False
        )
        
        if not ok:
            return
            
        if choice == "Authorized":
            self.authorize_face(stream_idx, face)
        else:
            self.unauthorize_face(stream_idx, face)

    def authorize_face(self, stream_idx, face):
        """Authorize a face by capturing their face embedding"""
        # Get the current frame from viewer
        viewer = self.viewer.get_viewer(stream_idx)
        if viewer is None or viewer.current_frame is None:
            return
            
        frame = np.copy(viewer.current_frame)
        (left, top, right, bottom) = face.bbox
        
        # Extract face region from bounding box
        face_region = frame[top:bottom, left:right]
        
        if face_region.size == 0:
            QMessageBox.warning(self, 'Error', 'Could not extract face region.')
            return
            
        # Get profile path
        profile_path = self.database_viewer.get_selected_directory()
        if not profile_path:
            # Create new profile
            profile_name, ok = QInputDialog.getText(self, 'Authorize Face', 'Enter name for authorized person:')
            if not ok or not profile_name:
                return
            profile_path = os.path.join(self.database_viewer.db_path, profile_name)
            os.makedirs(profile_path, exist_ok=True)
            self.database_viewer.load_profiles()
        else:
            profile_name = os.path.basename(profile_path)
        
        # Save face image
        i = 0
        while os.path.exists(os.path.join(profile_path, f"{i}.jpg")):
            i += 1
        filename = os.path.join(profile_path, f"{i}.jpg")
        cv2.imwrite(filename, cv2.cvtColor(face_region, cv2.COLOR_RGB2BGR))
        
        # Use the face's existing embedding if available, otherwise generate one
        embedding = face.embedding if face.embedding is not None and face.embedding.size > 0 else None
        
        if embedding is None:
            # Try to get face embedding using face recognizer
            try:
                # Put frame for detection
                try:
                    self.face_tracker.face_recognizer.detect_put(stream_idx, face_region, block=False, timeout=0.01)
                    annotated_frame = self.face_tracker.face_recognizer.detect_get(stream_idx, block=False, timeout=0.2)
                    
                    if annotated_frame and annotated_frame.num_detected_faces > 0:
                        bbox = annotated_frame.boxes[0]
                        keypoints = annotated_frame.keypoints[0] if annotated_frame.keypoints else []
                        
                        if len(keypoints) >= 2:
                            eyes = (keypoints[0], keypoints[1])
                        else:
                            x, y, w, h = bbox
                            eyes = ((x + w//4, y + h//3), (x + 3*w//4, y + h//3))
                        
                        x, y, w, h = bbox
                        xyxy = (x, y, x + w, y + h)
                        
                        self.face_tracker.face_recognizer.recognize_put(stream_idx, (face.track_id, face_region, xyxy, eyes), block=False, timeout=0.01)
                        _, embedding = self.face_tracker.face_recognizer.recognize_get(stream_idx, block=False, timeout=0.2)
                except Exception as e:
                    print(f"Could not generate embedding: {e}")
            except Exception as e:
                print(f"Face recognition error: {e}")
        
        # Update face status in tracker
        with self.face_tracker.tracker_dict_locks[stream_idx]:
            if face.track_id in self.face_tracker.tracker_dicts[stream_idx]:
                tracked_face = self.face_tracker.tracker_dicts[stream_idx][face.track_id]
                tracked_face.name = profile_name
                tracked_face.authorized = True
                if embedding is not None:
                    tracked_face.embedding = embedding
        
        # If we have an embedding, add to database
        if embedding is not None:
            self.auth_database.add_to_database(embedding, filename)
        
        print(f'Authorized face: {profile_name}, saved to {filename}')
        self.database_viewer.load_profiles()

    def unauthorize_face(self, stream_idx, face):
        """Mark a face as unauthorized"""
        # Update face status in tracker
        with self.face_tracker.tracker_dict_locks[stream_idx]:
            if face.track_id in self.face_tracker.tracker_dicts[stream_idx]:
                tracked_face = self.face_tracker.tracker_dicts[stream_idx][face.track_id]
                tracked_face.name = "PERSON"
                tracked_face.authorized = False
        
        print(f'Marked face ID {face.track_id} on Stream {stream_idx} as unauthorized')

    def toggle_capture_pause(self, stream_idx, pos):
        """Toggle play/pause of the capture thread"""
        if pos is None:
            return

        # If ROI drawing, don't toggle pause
        if self.roi_drawing_mode:
            return

        mouse_x, mouse_y = pos
        
        # If click is on an ROI, don't toggle pause (handled by ROI selection)
        clicked_roi = self.roi_manager.get_roi_at_point(stream_idx, mouse_x, mouse_y)
        if clicked_roi:
            return  # clicked on an ROI; handled by ROI selection

        # If click is on any tracked face, don't toggle pause (handled by authorize)
        tracked_faces = self.face_tracker.get_activated_tracker_objects(stream_idx)
        for face in tracked_faces:
            left, top, right, bottom = face.bbox
            if left <= mouse_x <= right and top <= mouse_y <= bottom:
                return  # clicked on a face; handled by authorize

        # Click was not on ROI or person: toggle pause/play (global for all streams)
        self.capture_thread.toggle_play()
        state = "paused" if self.capture_thread.pause else "running"
        print(f"Capture threads {state}")
        self.compositor.set_paused(self.capture_thread.pause, stream_idx=None)

    def process_frame(self, stream_idx, frame):
        """Process a frame - send to face tracker for detection"""
        # Send frame to face tracker for detection
        self.face_tracker.detect(stream_idx, frame)
    
    def process_face_detection(self, stream_idx, frame):
        """Process face detection results"""
        # Don't draw if paused - let timer handle paused redraws to avoid flickering
        if self.capture_thread.pause:
            return
        
        # Process frame through intrusion detector
        tracked_faces = self.intrusion_detector.process_frame(stream_idx, frame)
        
        # Update compositor for this stream
        self.compositor.set_tracked_faces(stream_idx, tracked_faces)
        self.compositor.set_alert_active(stream_idx, self.intrusion_detector.alert_active[stream_idx])
        self.intrusion_detector.update_alert(stream_idx)
        
        # Draw frame
        self.compositor.draw(stream_idx, frame)

    def poll_framerates(self):
        """Poll for frame updates when paused"""
        # If paused, redraw the last known frame so the overlay shows
        if self.capture_thread.pause:
            for stream_idx in range(self.num_streams):
                if stream_idx in self.face_tracker.current_frames:
                    frame = np.copy(self.face_tracker.current_frames[stream_idx].image)
                    # Only redraw if frame has changed or first time to avoid flickering
                    if not hasattr(self, f'_last_paused_frame_{stream_idx}') or not np.array_equal(frame, getattr(self, f'_last_paused_frame_{stream_idx}', None)):
                        self.compositor.draw(stream_idx, frame)
                        setattr(self, f'_last_paused_frame_{stream_idx}', frame)

    def closeEvent(self, event):
        # Stop timer first to prevent further frame polling
        self.timer.stop()
        
        self.capture_thread.stop()
        self.capture_thread.wait()
        self.face_tracker.stop()
        if hasattr(self.face_tracker, 'face_recognizer'):
            self.face_tracker.face_recognizer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Intrusion Detection System - Multi-Stream')
    parser.add_argument('--video_paths', default = ['/dev/video0'], nargs='+', help='Video source paths (e.g., /dev/video0 /dev/video1)')
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    
    # Use command-line arguments (defaults to ['/dev/video0'] if not provided)
    video_paths = args.video_paths
    
    # Validate video sources
    valid_video_paths = []
    for video_path in video_paths:
        if "/dev/video" in video_path:
            # Try to verify it's actually a capture device
            test_cap = cv2.VideoCapture(video_path)
            if not test_cap.isOpened():
                print(f"Warning: {video_path} cannot be opened, skipping")
                test_cap.release()
                continue
            # Try to read a frame to verify it's a capture device
            ret, frame = test_cap.read()
            test_cap.release()
            if not ret or frame is None:
                print(f"Warning: {video_path} appears to be a control device, skipping")
                continue
        # Only reach here if device is valid
        valid_video_paths.append(video_path)
    
    if not valid_video_paths:
        print("Error: Video Source not Found")
        sys.exit(1)
    
    print(f"Using {len(valid_video_paths)} video stream(s): {valid_video_paths}")
    player = Demo(valid_video_paths, VIDEO_CONFIG['1080p'])
    player.resize(1600, 900)
    player.show()
    sys.exit(app.exec())
