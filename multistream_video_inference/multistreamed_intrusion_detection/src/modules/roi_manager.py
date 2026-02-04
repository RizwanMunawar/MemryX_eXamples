"""
ROI Manager for handling Region of Interest definition and checking
"""
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field


@dataclass
class ROI:
    """Represents a single Region of Interest"""
    roi_id: int
    name: str
    coordinates: List[int]  # [x1, y1, x2, y2]
    color: Tuple[int, int, int] = (0, 255, 0)  # Green default (BGR format)
    alert_active: bool = False
    enabled: bool = True


class ROIManager:
    """Manages multiple ROI definitions and checking"""
    
    # Color palette for ROIs (BGR format)
    COLOR_PALETTE = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (0, 128, 255),  # Light Blue
        (128, 255, 0),  # Lime
    ]
    
    def __init__(self):
        self.rois: List[ROI] = []
        self.next_roi_id = 1
        # Keep for backward compatibility
        self.roi_color = (0, 255, 0)  # Green for normal, red for alert
        
    def add_roi(self, x1: int, y1: int, x2: int, y2: int, name: str = None) -> int:
        """
        Add a new ROI
        
        Args:
            x1, y1, x2, y2: ROI coordinates
            name: Optional name for the ROI. If None, generates default name
            
        Returns:
            ROI ID of the newly created ROI
        """
        # Ensure x1 < x2 and y1 < y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        roi_id = self.next_roi_id
        self.next_roi_id += 1
        
        if name is None:
            name = f"ROI {roi_id}"
        
        # Assign color from palette (cycle through if more ROIs than colors)
        color_idx = (roi_id - 1) % len(self.COLOR_PALETTE)
        color = self.COLOR_PALETTE[color_idx]
        
        roi = ROI(
            roi_id=roi_id,
            name=name,
            coordinates=[x1, y1, x2, y2],
            color=color,
            alert_active=False,
            enabled=True
        )
        
        self.rois.append(roi)
        return roi_id
    
    def remove_roi(self, roi_id: int) -> bool:
        """
        Remove a ROI by ID
        
        Returns:
            True if ROI was found and removed, False otherwise
        """
        for i, roi in enumerate(self.rois):
            if roi.roi_id == roi_id:
                self.rois.pop(i)
                return True
        return False
    
    def get_roi(self, roi_id: int = None) -> Optional[ROI]:
        """
        Get ROI by ID, or first ROI if no ID provided (backward compatibility)
        
        Args:
            roi_id: Optional ROI ID. If None, returns first ROI
            
        Returns:
            ROI object or None
        """
        if roi_id is None:
            # Backward compatibility: return first ROI coordinates as list
            if self.rois:
                return self.rois[0].coordinates
            return None
        
        for roi in self.rois:
            if roi.roi_id == roi_id:
                return roi
        return None
    
    def get_all_rois(self) -> List[ROI]:
        """Get all ROIs"""
        return self.rois.copy()
    
    def get_roi_at_point(self, x: int, y: int) -> Optional[ROI]:
        """
        Get ROI that contains the given point
        
        Args:
            x, y: Point coordinates
            
        Returns:
            ROI object if point is inside a ROI, None otherwise
        """
        # Check ROIs in reverse order (last drawn first)
        for roi in reversed(self.rois):
            if not roi.enabled:
                continue
            x1, y1, x2, y2 = roi.coordinates
            if x1 <= x <= x2 and y1 <= y <= y2:
                return roi
        return None
    
    def has_roi(self) -> bool:
        """Check if any ROI is defined (backward compatibility)"""
        return len(self.rois) > 0
        
    def in_roi(self, xywh: Tuple[int, int, int, int]) -> bool:
        """
        Check if the centroid of a bounding box is in any ROI (backward compatibility)
        
        Args:
            xywh: Bounding box as (x, y, width, height)
            
        Returns:
            True if centroid is in any ROI, False otherwise
        """
        if not self.has_roi():
            return False
            
        x1, y1, w, h = xywh
        x2 = x1 + w
        y2 = y1 + h
        centroid_box = [(x1 + x2) / 2, (y1 + y2) / 2]
        
        for roi in self.rois:
            if not roi.enabled:
                continue
            rx1, ry1, rx2, ry2 = roi.coordinates
            if (centroid_box[0] > rx1 and centroid_box[0] < rx2 and
                centroid_box[1] > ry1 and centroid_box[1] < ry2):
                return True
        return False
        
    def in_roi_xyxy(self, xyxy: Tuple[int, int, int, int]) -> List[int]:
        """
        Check which ROI(s) contain the centroid of a bounding box (xyxy format)
        
        Args:
            xyxy: Bounding box as (x1, y1, x2, y2)
            
        Returns:
            List of ROI IDs that contain the centroid
        """
        if not self.has_roi():
            return []
            
        x1, y1, x2, y2 = xyxy
        centroid_box = [(x1 + x2) / 2, (y1 + y2) / 2]
        
        roi_ids = []
        for roi in self.rois:
            if not roi.enabled:
                continue
            rx1, ry1, rx2, ry2 = roi.coordinates
            if (centroid_box[0] > rx1 and centroid_box[0] < rx2 and
                centroid_box[1] > ry1 and centroid_box[1] < ry2):
                roi_ids.append(roi.roi_id)
        
        return roi_ids
        
    def clear_roi(self):
        """Clear all ROIs (backward compatibility)"""
        self.rois.clear()
    
    def clear_all_rois(self):
        """Clear all ROIs"""
        self.rois.clear()
    
    def set_roi_alert(self, roi_id: int, active: bool):
        """
        Set alert state for a specific ROI
        
        Args:
            roi_id: ROI ID
            active: True to activate alert, False to deactivate
        """
        roi = self.get_roi(roi_id)
        if roi:
            roi.alert_active = active
    
    def set_alert_color(self):
        """Set all ROI colors to red for alert (backward compatibility)"""
        for roi in self.rois:
            roi.alert_active = True
        
    def set_normal_color(self):
        """Set all ROI colors to normal (backward compatibility)"""
        for roi in self.rois:
            roi.alert_active = False
    
    # Backward compatibility: maintain old set_roi method
    def set_roi(self, x1: int, y1: int, x2: int, y2: int):
        """Set ROI coordinates (backward compatibility - replaces all ROIs with single ROI)"""
        self.clear_all_rois()
        self.add_roi(x1, y1, x2, y2, "ROI 1")

