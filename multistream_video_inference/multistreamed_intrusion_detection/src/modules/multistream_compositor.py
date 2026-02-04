"""
Multi-Stream Compositor
Manages compositing for multiple video streams with per-stream ROI managers and alert states
"""
from PySide6.QtCore import QObject, Signal
import numpy as np
from typing import Dict

from .compositor import Compositor
from .multistream_roi_manager import MultiStreamROIManager
from .face_tracker import TrackedFace


class MultiStreamCompositor(QObject):
    """Manages compositing for multiple streams"""
    frame_ready = Signal(int, np.ndarray)  # (stream_idx, frame)

    def __init__(self, roi_manager: MultiStreamROIManager):
        """
        Initialize multi-stream compositor
        
        Args:
            roi_manager: Multi-stream ROI manager
        """
        super().__init__()
        self.roi_manager = roi_manager
        self.num_streams = roi_manager.num_streams
        
        # Create compositor for each stream
        self.compositors: Dict[int, Compositor] = {}
        for stream_idx in range(self.num_streams):
            stream_roi_manager = roi_manager.get_roi_manager(stream_idx)
            compositor = Compositor(stream_roi_manager)
            # Connect compositor's frame_ready signal to our handler
            compositor.frame_ready.connect(lambda frame, idx=stream_idx: self._on_frame_ready(idx, frame))
            self.compositors[stream_idx] = compositor

    def _on_frame_ready(self, stream_idx: int, frame: np.ndarray):
        """Handle frame from a specific compositor and emit with stream index"""
        self.frame_ready.emit(stream_idx, frame)

    def set_paused(self, paused: bool, stream_idx: int = None):
        """
        Set paused state for compositor(s)
        
        Args:
            paused: Pause state
            stream_idx: Stream index. If None, sets for all streams.
        """
        if stream_idx is not None:
            if stream_idx in self.compositors:
                self.compositors[stream_idx].set_paused(paused)
        else:
            for compositor in self.compositors.values():
                compositor.set_paused(paused)

    def update_mouse_pos(self, stream_idx: int, pos: tuple):
        """
        Update mouse position for a specific stream
        
        Args:
            stream_idx: Stream index
            pos: Mouse position (x, y)
        """
        if stream_idx in self.compositors:
            self.compositors[stream_idx].update_mouse_pos(pos)

    def set_tracked_faces(self, stream_idx: int, faces: list[TrackedFace]):
        """
        Update tracked faces for a specific stream
        
        Args:
            stream_idx: Stream index
            faces: List of tracked faces
        """
        if stream_idx in self.compositors:
            self.compositors[stream_idx].set_tracked_faces(faces)

    def set_alert_active(self, stream_idx: int, active: bool):
        """
        Set alert state for a specific stream
        
        Args:
            stream_idx: Stream index
            active: Alert state
        """
        if stream_idx in self.compositors:
            self.compositors[stream_idx].set_alert_active(active)

    def draw(self, stream_idx: int, frame: np.ndarray):
        """
        Draw frame for a specific stream
        
        Args:
            stream_idx: Stream index
            frame: Frame to draw
        """
        if stream_idx in self.compositors:
            self.compositors[stream_idx].draw(frame)

    def get_compositor(self, stream_idx: int) -> Compositor:
        """Get compositor for a specific stream"""
        return self.compositors.get(stream_idx)

