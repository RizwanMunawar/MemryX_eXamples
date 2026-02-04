"""
Grid Viewer Widget
Displays multiple video streams in a grid layout
"""
from PySide6.QtWidgets import QWidget, QGridLayout
from PySide6.QtCore import Signal, Qt
import numpy as np
import math

from .viewer import FrameViewer


class GridViewer(QWidget):
    """Widget that displays multiple FrameViewer instances in a grid layout"""
    mouse_move = Signal(int, tuple)  # (stream_idx, pos)
    mouse_click = Signal(int, tuple)  # (stream_idx, pos)
    roi_drawing_started = Signal(int, tuple)  # (stream_idx, (x, y))
    roi_drawing_updated = Signal(int, tuple, tuple)  # (stream_idx, start, current)
    roi_drawing_finished = Signal(int, tuple, tuple)  # (stream_idx, start, end)

    def __init__(self, num_streams: int):
        """
        Initialize grid viewer
        
        Args:
            num_streams: Number of video streams to display
        """
        super().__init__()
        self.num_streams = num_streams
        
        # Calculate grid dimensions
        cols = math.ceil(math.sqrt(num_streams))
        rows = math.ceil(num_streams / cols)
        self.grid_cols = cols
        self.grid_rows = rows
        
        # Create layout
        self.grid_layout = QGridLayout(self)
        self.grid_layout.setSpacing(2)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create FrameViewer for each stream
        self.viewers: list[FrameViewer] = []
        for stream_idx in range(num_streams):
            viewer = FrameViewer()
            self.viewers.append(viewer)
            
            # Calculate grid position
            row = stream_idx // cols
            col = stream_idx % cols
            
            # Connect viewer signals to our signals with stream index
            viewer.mouse_move.connect(lambda pos, idx=stream_idx: self.mouse_move.emit(idx, pos))
            viewer.mouse_click.connect(lambda pos, idx=stream_idx: self.mouse_click.emit(idx, pos))
            viewer.roi_drawing_started.connect(lambda pos, idx=stream_idx: self.roi_drawing_started.emit(idx, pos))
            viewer.roi_drawing_updated.connect(lambda start, current, idx=stream_idx: self.roi_drawing_updated.emit(idx, start, current))
            viewer.roi_drawing_finished.connect(lambda start, end, idx=stream_idx: self.roi_drawing_finished.emit(idx, start, end))
            
            # Add to grid
            self.grid_layout.addWidget(viewer, row, col)
        
        # Set equal stretch for all cells
        for i in range(rows):
            self.grid_layout.setRowStretch(i, 1)
        for i in range(cols):
            self.grid_layout.setColumnStretch(i, 1)

    def update_frame(self, stream_idx: int, frame: np.ndarray):
        """
        Update frame for a specific stream
        
        Args:
            stream_idx: Stream index
            frame: Frame to display
        """
        if 0 <= stream_idx < len(self.viewers):
            self.viewers[stream_idx].update_frame(frame)

    def start_roi_drawing(self, stream_idx: int = None):
        """
        Start ROI drawing mode
        
        Args:
            stream_idx: Stream index to enable drawing on. If None, enables on all streams.
        """
        if stream_idx is not None:
            if 0 <= stream_idx < len(self.viewers):
                self.viewers[stream_idx].start_roi_drawing()
        else:
            for viewer in self.viewers:
                viewer.start_roi_drawing()

    def stop_roi_drawing(self, stream_idx: int = None):
        """
        Stop ROI drawing mode
        
        Args:
            stream_idx: Stream index to disable drawing on. If None, disables on all streams.
        """
        if stream_idx is not None:
            if 0 <= stream_idx < len(self.viewers):
                self.viewers[stream_idx].stop_roi_drawing()
        else:
            for viewer in self.viewers:
                viewer.stop_roi_drawing()

    def get_viewer(self, stream_idx: int) -> FrameViewer:
        """Get FrameViewer for a specific stream"""
        if 0 <= stream_idx < len(self.viewers):
            return self.viewers[stream_idx]
        return None

