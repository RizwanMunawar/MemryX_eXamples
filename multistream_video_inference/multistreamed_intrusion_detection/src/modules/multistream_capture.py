"""
Multi-Stream Video Capture Module
Manages multiple CaptureThread instances for different video sources
"""
from PySide6.QtCore import QObject, Signal
import numpy as np
from typing import List, Optional

from .capture import CaptureThread, VideoConfig


class MultiStreamCapture(QObject):
    """Manages multiple video capture threads, one per stream"""
    frame_ready = Signal(int, np.ndarray)  # (stream_idx, frame)

    def __init__(self, video_paths: List[str], video_config: Optional[VideoConfig] = None):
        """
        Initialize multi-stream capture
        
        Args:
            video_paths: List of video source paths (one per stream)
            video_config: Video configuration to use for all streams
        """
        super().__init__()
        self.video_paths = video_paths
        self.video_config = video_config
        self.num_streams = len(video_paths)
        
        # Create capture threads for each stream
        self.capture_threads: List[CaptureThread] = []
        for stream_idx, video_path in enumerate(video_paths):
            thread = CaptureThread(video_path, video_config)
            # Connect each thread's frame_ready signal to our handler
            thread.frame_ready.connect(lambda frame, idx=stream_idx: self._on_frame_ready(idx, frame))
            self.capture_threads.append(thread)

    def _on_frame_ready(self, stream_idx: int, frame: np.ndarray):
        """Handle frame from a specific stream and emit with stream index"""
        self.frame_ready.emit(stream_idx, frame)

    def start(self):
        """Start all capture threads"""
        for thread in self.capture_threads:
            thread.start()

    def stop(self):
        """Stop all capture threads"""
        for thread in self.capture_threads:
            thread.stop()

    def wait(self):
        """Wait for all capture threads to finish"""
        for thread in self.capture_threads:
            thread.wait()

    def toggle_play(self):
        """Toggle play/pause for all streams"""
        for thread in self.capture_threads:
            thread.toggle_play()

    @property
    def pause(self):
        """Get pause state (all streams share the same state)"""
        if self.capture_threads:
            return self.capture_threads[0].pause
        return False

