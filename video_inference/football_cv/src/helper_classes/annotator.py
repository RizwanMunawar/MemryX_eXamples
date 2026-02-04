import cv2
import numpy as np
from .utilities import get_bbox_wdth, get_center_of_bbox, get_foot_position 


# ======================================
# BoxScaler (input_size → original size)
# ======================================
class BoxScaler:
    """Converts model predictions (input_size x input_size) back to the original video frame size."""

    def __init__(self, input_size=640):
        self.input_size = input_size

    def scale(self, boxes, orig_w, orig_h):
        """Return a scaled *copy* of the boxes (never mutate input)."""
        if boxes.size == 0:
            return boxes

        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size

        scaled = boxes.copy()
        scaled[:, [0, 2]] *= scale_x
        scaled[:, [1, 3]] *= scale_y

        return scaled


# ======================================
# Annotator (draws annotations)
# ======================================
class Annotator:
    """
    Handles drawing annotations on video frames.
    """

    def __init__(self, input_size=None, class_names=None):
        self.class_names = class_names
        self.scaler = BoxScaler(input_size) if input_size else None
        self.input_size = input_size

        # Set expected class IDs
        self.ball_class_id = 0 
        self.goalkeeper_class_id = 1
        self.player_class_id = 2
        self.referee_class_id = 3

        
    def _get_jersey_color(self, frame, bbox):
        """
        Samples the jersey area and finds the dominant BGR color 
        by calculating the average color (equivalent to k=1 KMeans).
        """
        x1, y1, x2, y2 = bbox.astype(int)
        h = y2 - y1
        
        # Define Sample region
        x_crop = x1 + (x2 - x1) // 4
        w_crop = (x2 - x1) // 2
        y_crop = y1 + int(h * 0.15) 
        h_crop = int(h * 0.25)
        
        # Ensure coordinates are within frame bounds
        x_crop = max(0, x_crop)
        y_crop = max(0, y_crop)
        w_crop = min(frame.shape[1] - x_crop, w_crop)
        h_crop = min(frame.shape[0] - y_crop, h_crop)
        
        if w_crop <= 0 or h_crop <= 0:
            return (0, 255, 0) # Green fallback

        # 1. Extract region
        jersey_region = frame[y_crop:y_crop + h_crop, x_crop:x_crop + w_crop]
        
        # 2. VECTORIZED Calculation of the Mean Color
        # Use np.mean(axis=(0, 1)) to calculate the mean B, G, and R across all pixels
        try:
            dominant_bgr = np.mean(jersey_region, axis=(0, 1)).astype(int)
            return tuple(dominant_bgr.tolist())
        except Exception:
            # Fallback if region is somehow invalid
            return (128, 128, 128)
   

    
    def draw_annotations(self, frame, results_by_class):
        """
        Draws annotations on the frame using results split by class.
        
        Args:
            frame (np.ndarray): The current video frame.
            results_by_class (dict): The class-split dictionary from the tracker.
            
        Returns:
            np.ndarray: The frame with drawn annotations.
        """

        orig_h, orig_w = frame.shape[:2]
        
        # Iterate through each class group
        for cls, det_data in results_by_class.items():
            
            # Get the arrays for the current class
            boxes = det_data["boxes"]
            track_ids = det_data.get("id", np.array([]))
            
            # Speed and Distance are only guaranteed for Player class, but we extract them all
            speeds = det_data.get("speed", np.zeros(len(boxes)))
            distances = det_data.get("distance", np.zeros(len(boxes)))

            # Scale boxes once for this class group
            if self.scaler:
                scaled_boxes = self.scaler.scale(boxes, orig_w, orig_h)
            else:
                scaled_boxes = boxes

            # Iterate through each object within the class
            for i in range(len(scaled_boxes)):
                bbox = scaled_boxes[i]
                
                # Safely get track_id, speed, and distance for the current object
                track_id = int(track_ids[i]) if len(track_ids) > i else -1
                speed = speeds[i]
                distance = distances[i]

                
                color = (255, 255, 0) # Default Cyan
                
                # --- COLOR DETERMINATION AND DRAWING ---
                
                if cls == self.player_class_id: # class_id == 2
                    color = self._get_jersey_color(frame, bbox)
                    
                    # Drawing functions use the object's data (bbox, speed, distance)
                    self.draw_ellipse(frame, bbox, color)
                    self.draw_single_speed_and_distance(frame, bbox=bbox, speed=speed, distance=distance)
                    
                elif cls == self.goalkeeper_class_id: # class_id == 1
                    color = self._get_jersey_color(frame, bbox)
                    
                    self.draw_ellipse(frame, bbox, color)
                
                elif cls == self.ball_class_id: # class_id == 0
                    color = (0, 0, 255) # Red
                    self.draw_triangle(frame, bbox, color)
                
                elif cls == self.referee_class_id: # class_id == 3
                    color = (0, 255, 255) # yellow
                    self.draw_ellipse(frame, bbox, color)

        return frame

    def draw_single_speed_and_distance(self, frame, bbox, speed, distance):
        """
        Reused from:
        https://github.com/abdullahtarek/football_analysis
        Original Author: Abdullah Tarek
        License: MIT

        Modified for:
        - MemryX MXA inference pipeline
        - Frame queue-based processing

        Draws speed and distance for a single object (player) on the frame.

        Args:
            frame (np.ndarray): The current video frame.
            bbox (np.ndarray or list): A single bounding box [x1, y1, x2, y2].
            speed (float): The speed in km/h.
            distance (float): The total distance in meters.

        Returns:
            np.ndarray: The frame with the new annotation drawn.
        """
        
        # Filter: Ensure data is valid (speed/distance should be positive)
        # Using a small threshold > 0 to account for floating point numbers near zero
        if speed <= 0.01 or distance <= 0.0:
            return frame
        
        # Convert bbox coordinates to integers
        bbox = [int(c) for c in bbox]
        
        # Determine text position (using the utility function)
        position = get_foot_position(bbox)
        position = list(position)
        position[1] += 40 # Offset the text # pixels under the foot position
        
        position = tuple(map(int, position))

        # Draw Speed Text
        cv2.putText(
            frame, 
            f"{speed:.2f} km/h",
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0), # Black text
            2
        )
        
        # Draw Distance Text (20 pixels below the speed text)
        cv2.putText(
            frame, 
            f"{distance:.2f} m",
            (position[0], position[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0), # Black text
            2
        )
            
        return frame
        
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Reused from:
        https://github.com/abdullahtarek/football_analysis
        Original Author: Abdullah Tarek
        License: MIT    
        """
        y2 = int(bbox[3])
        
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_wdth(bbox)
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)


            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )    

        return frame

    def draw_triangle(self, frame, bbox, color):
        """
        Reused from:
        https://github.com/abdullahtarek/football_analysis
        Original Author: Abdullah Tarek
        License: MIT    
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20]
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame