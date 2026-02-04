"""
Multi-Stream ROI Manager
Manages ROIs per stream, with each stream having its own independent ROI set
"""
from typing import Dict, List, Optional, Tuple
from .roi_manager import ROIManager, ROI


class MultiStreamROIManager:
    """Manages ROIs for multiple streams, with independent ROI sets per stream"""
    
    def __init__(self, num_streams: int):
        """
        Initialize multi-stream ROI manager
        
        Args:
            num_streams: Number of streams to manage ROIs for
        """
        self.num_streams = num_streams
        self.roi_managers: Dict[int, ROIManager] = {
            i: ROIManager() for i in range(num_streams)
        }

    def get_roi_manager(self, stream_idx: int) -> ROIManager:
        """Get ROI manager for a specific stream"""
        if stream_idx not in self.roi_managers:
            raise ValueError(f"Invalid stream index: {stream_idx}")
        return self.roi_managers[stream_idx]

    def add_roi(self, stream_idx: int, x1: int, y1: int, x2: int, y2: int, name: str = None) -> int:
        """
        Add a new ROI for a specific stream
        
        Args:
            stream_idx: Stream index
            x1, y1, x2, y2: ROI coordinates
            name: Optional name for the ROI
            
        Returns:
            ROI ID of the newly created ROI
        """
        roi_manager = self.get_roi_manager(stream_idx)
        return roi_manager.add_roi(x1, y1, x2, y2, name)

    def remove_roi(self, stream_idx: int, roi_id: int) -> bool:
        """
        Remove a ROI from a specific stream
        
        Args:
            stream_idx: Stream index
            roi_id: ROI ID to remove
            
        Returns:
            True if ROI was found and removed, False otherwise
        """
        roi_manager = self.get_roi_manager(stream_idx)
        return roi_manager.remove_roi(roi_id)

    def get_roi(self, stream_idx: int, roi_id: int = None) -> Optional[ROI]:
        """
        Get ROI by ID for a specific stream
        
        Args:
            stream_idx: Stream index
            roi_id: Optional ROI ID. If None, returns first ROI (backward compatibility)
            
        Returns:
            ROI object or None
        """
        roi_manager = self.get_roi_manager(stream_idx)
        return roi_manager.get_roi(roi_id)

    def get_all_rois(self, stream_idx: int) -> List[ROI]:
        """
        Get all ROIs for a specific stream
        
        Args:
            stream_idx: Stream index
            
        Returns:
            List of ROI objects
        """
        roi_manager = self.get_roi_manager(stream_idx)
        return roi_manager.get_all_rois()

    def get_roi_at_point(self, stream_idx: int, x: int, y: int) -> Optional[ROI]:
        """
        Get ROI that contains the given point for a specific stream
        
        Args:
            stream_idx: Stream index
            x, y: Point coordinates
            
        Returns:
            ROI object if point is inside a ROI, None otherwise
        """
        roi_manager = self.get_roi_manager(stream_idx)
        return roi_manager.get_roi_at_point(x, y)

    def has_roi(self, stream_idx: int) -> bool:
        """
        Check if any ROI is defined for a specific stream
        
        Args:
            stream_idx: Stream index
            
        Returns:
            True if stream has any ROIs, False otherwise
        """
        roi_manager = self.get_roi_manager(stream_idx)
        return roi_manager.has_roi()

    def in_roi(self, stream_idx: int, xywh: Tuple[int, int, int, int]) -> bool:
        """
        Check if the centroid of a bounding box is in any ROI for a specific stream
        
        Args:
            stream_idx: Stream index
            xywh: Bounding box as (x, y, width, height)
            
        Returns:
            True if centroid is in any ROI, False otherwise
        """
        roi_manager = self.get_roi_manager(stream_idx)
        return roi_manager.in_roi(xywh)

    def in_roi_xyxy(self, stream_idx: int, xyxy: Tuple[int, int, int, int]) -> List[int]:
        """
        Check which ROI(s) contain the centroid of a bounding box for a specific stream
        
        Args:
            stream_idx: Stream index
            xyxy: Bounding box as (x1, y1, x2, y2)
            
        Returns:
            List of ROI IDs that contain the centroid
        """
        roi_manager = self.get_roi_manager(stream_idx)
        return roi_manager.in_roi_xyxy(xyxy)

    def clear_all_rois(self, stream_idx: int = None):
        """
        Clear all ROIs for a specific stream or all streams
        
        Args:
            stream_idx: Stream index. If None, clears all streams
        """
        if stream_idx is not None:
            roi_manager = self.get_roi_manager(stream_idx)
            roi_manager.clear_all_rois()
        else:
            for roi_manager in self.roi_managers.values():
                roi_manager.clear_all_rois()

    def set_roi_alert(self, stream_idx: int, roi_id: int, active: bool):
        """
        Set alert state for a specific ROI in a specific stream
        
        Args:
            stream_idx: Stream index
            roi_id: ROI ID
            active: True to activate alert, False to deactivate
        """
        roi_manager = self.get_roi_manager(stream_idx)
        roi_manager.set_roi_alert(roi_id, active)

    def get_all_stream_rois(self) -> Dict[int, List[ROI]]:
        """
        Get all ROIs for all streams
        
        Returns:
            Dictionary mapping stream_idx to list of ROIs
        """
        return {
            stream_idx: manager.get_all_rois()
            for stream_idx, manager in self.roi_managers.items()
        }

