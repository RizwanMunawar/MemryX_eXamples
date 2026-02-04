"""
Multi-Stream Intrusion Detection Logic
Handles intrusion detection per stream with independent alert states
"""
import numpy as np
import cv2
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from .multistream_face_tracker import MultiStreamFaceTracker, TrackedFace
from .multistream_roi_manager import MultiStreamROIManager
from .authorization_db import AuthorizationDatabase


@dataclass
class IntrusionEvent:
    """Represents an intrusion detection event"""
    stream_idx: int
    track_id: int
    timestamp: float
    face_image_path: str
    bbox: Tuple[int, int, int, int]


class MultiStreamIntrusionDetector:
    """Multi-stream intrusion detection system using face detection only"""
    
    def __init__(self, 
                 face_tracker: MultiStreamFaceTracker,
                 roi_manager: MultiStreamROIManager,
                 auth_database: AuthorizationDatabase,
                 intrusions_dir: str = '../assets/intrusions'):
        """
        Initialize multi-stream intrusion detector
        
        Args:
            face_tracker: Multi-stream face tracking module
            roi_manager: Multi-stream ROI management module
            auth_database: Authorization database
            intrusions_dir: Directory to save intrusion images
        """
        self.face_tracker = face_tracker
        self.roi_manager = roi_manager
        self.auth_database = auth_database
        self.intrusions_dir = Path(intrusions_dir)
        self.intrusions_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_streams = face_tracker.num_streams
        
        self.frame_ids: Dict[int, int] = {i: 0 for i in range(self.num_streams)}
        self.intrusion_events: List[IntrusionEvent] = []
        
        # Per-stream alert states
        self.alert_active: Dict[int, bool] = {i: False for i in range(self.num_streams)}
        self.alert_counters: Dict[int, Dict[int, int]] = {
            i: {} for i in range(self.num_streams)
        }
        self.ALERT_NUM_FRAMES = 60
        
    def process_frame(self, stream_idx: int, frame: np.ndarray) -> List[TrackedFace]:
        """
        Process a frame for intrusion detection on a specific stream
        
        Args:
            stream_idx: Stream index
            frame: Input frame (RGB format)
            
        Returns:
            List of tracked faces with authorization status
        """
        self.frame_ids[stream_idx] += 1
        
        # Face detection and tracking is handled by MultiStreamFaceTracker
        # We just need to check for intrusions in tracked faces
        tracked_faces = self.face_tracker.get_activated_tracker_objects(stream_idx)
        
        # Check for unauthorized faces in ROIs and trigger alerts
        for face in tracked_faces:
            # Check all ROIs this face is in
            for roi_id in face.in_roi_ids:
                if not face.authorized:
                    # Unauthorized face in ROI - trigger intrusion alert
                    if self.frame_ids[stream_idx] % 30 == 0:  # Check every 30 frames to avoid spam
                        self._handle_intrusion(stream_idx, face, frame, roi_id)
        
        return tracked_faces
            
    def _handle_intrusion(self, stream_idx: int, face: TrackedFace, frame: np.ndarray, roi_id: int):
        """
        Handle intrusion detection - save face image and log event
        
        Args:
            stream_idx: Stream index where intrusion was detected
            face: Tracked face that triggered intrusion
            frame: Full frame
            roi_id: ID of the ROI where intrusion was detected
        """
        # Extract face region from bounding box
        x1, y1, x2, y2 = face.bbox
        h, w = frame.shape[:2]
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        face_region = frame[y1:y2, x1:x2]
        
        if face_region.size == 0:
            return
        
        # Generate filename with timestamp and stream index
        timestamp = time.time()
        filename = f"{int(timestamp)}_{stream_idx}_{face.track_id}_roi{roi_id}.jpg"
        filepath = self.intrusions_dir / filename
        
        # Save face image
        cv2.imwrite(str(filepath), cv2.cvtColor(face_region, cv2.COLOR_RGB2BGR))
        
        # Create intrusion event
        event = IntrusionEvent(
            stream_idx=stream_idx,
            track_id=face.track_id,
            timestamp=timestamp,
            face_image_path=str(filepath),
            bbox=face.bbox
        )
        self.intrusion_events.append(event)
        
        # Set alert for this specific ROI in this specific stream
        self.alert_counters[stream_idx][roi_id] = self.ALERT_NUM_FRAMES
        roi_manager = self.roi_manager.get_roi_manager(stream_idx)
        if roi_manager:
            roi_manager.set_roi_alert(roi_id, True)
        
        # Update alert state for this stream
        if roi_manager:
            self.alert_active[stream_idx] = any(
                roi.alert_active for roi in roi_manager.get_all_rois()
            )
        else:
            self.alert_active[stream_idx] = len(self.alert_counters[stream_idx]) > 0
        
        roi = roi_manager.get_roi(roi_id) if roi_manager else None
        roi_name = roi.name if roi else f"ROI {roi_id}"
        print(f"INTRUSION DETECTED in Stream {stream_idx}, {roi_name}! Track ID: {face.track_id}, Saved to: {filepath}")
        
    def update_alert(self, stream_idx: int):
        """Update alert state for a specific stream"""
        # Decrement counters for all ROIs in this stream
        roi_ids_to_remove = []
        for roi_id, counter in self.alert_counters[stream_idx].items():
            if counter > 0:
                self.alert_counters[stream_idx][roi_id] = counter - 1
            else:
                # Counter expired, clear alert for this ROI
                roi_manager = self.roi_manager.get_roi_manager(stream_idx)
                if roi_manager:
                    roi_manager.set_roi_alert(roi_id, False)
                roi_ids_to_remove.append(roi_id)
        
        # Remove expired counters
        for roi_id in roi_ids_to_remove:
            del self.alert_counters[stream_idx][roi_id]
        
        # Update alert state for this stream
        roi_manager = self.roi_manager.get_roi_manager(stream_idx)
        if roi_manager:
            self.alert_active[stream_idx] = any(
                roi.alert_active for roi in roi_manager.get_all_rois()
            )
        else:
            self.alert_active[stream_idx] = len(self.alert_counters[stream_idx]) > 0
            
    def get_intrusion_events(self, stream_idx: int = None) -> List[IntrusionEvent]:
        """
        Get list of intrusion events
        
        Args:
            stream_idx: Optional stream index to filter events. If None, returns all events.
            
        Returns:
            List of intrusion events
        """
        if stream_idx is not None:
            return [e for e in self.intrusion_events if e.stream_idx == stream_idx]
        return self.intrusion_events.copy()
    
    def get_tracked_faces(self, stream_idx: int) -> List[TrackedFace]:
        """Get all tracked faces for a specific stream"""
        return self.face_tracker.get_activated_tracker_objects(stream_idx)

