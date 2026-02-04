from .utilities import measure_distance 
import numpy as np

class RealTimeSpeedDistance_Estimator():
    def __init__(self, frame_window=5, frame_rate=24):
        # Number of frames over which to calculate speed/distance
        self.frame_window = frame_window
        # Frame rate of the video source
        self.frame_rate = frame_rate
        
        # State: Stores the accumulated distance for each track ID
        # Format: {'players': {1: 15.5, 2: 10.2}, ...}
        self.total_distance = {}
        
        # State: Stores a history of recent track information for speed calculation
        # Format: {'players': {1: [frame_0_track_info, frame_1_track_info, ...]}, ...}
        self.track_history = {}
        
        # State: Stores the current frame number being processed
        self.current_frame_num = 0
    
    def _update_speed_and_distance(self, object, track_id, current_track_info):
        """
        Internal function to calculate and update speed/distance for a single track ID.
        """
        
        if object not in self.track_history:
            self.track_history[object] = {}
        
        if track_id not in self.track_history[object]:
            self.track_history[object][track_id] = []
        
        # Add the current track info to the history (maintaining a fixed size)
        self.track_history[object][track_id].append({
            'frame_num': self.current_frame_num,
            'position_transformed': current_track_info.get('position_transformed'),
            # Include 'bbox' if needed for the draw function
            # 'bbox': current_track_info.get('bbox') 
        })
        
        # Only keep the last 'frame_window' + 1 entries
        if len(self.track_history[object][track_id]) > self.frame_window + 1:
            self.track_history[object][track_id].pop(0)

        # Proceed only if we have enough data points (start and end of window)
        if len(self.track_history[object][track_id]) < self.frame_window + 1:
            return 
        
        # Define the start and end points for calculation
        start_track_info = self.track_history[object][track_id][0] # frame_num ago
        end_track_info = self.track_history[object][track_id][-1]  # current frame
        
        start_position = start_track_info.get('position_transformed')
        end_position = end_track_info.get('position_transformed')

        if start_position is None or end_position is None:
            return
        
        # --- Speed and Distance Calculation ---
        
        # This is the distance *covered in the last frame_window frames*
        distance_covered = measure_distance(start_position, end_position)
        
        # Calculate time elapsed
        time_elapsed = (end_track_info['frame_num'] - start_track_info['frame_num']) / self.frame_rate
        
        # Calculate speed for the window
        speed_meteres_per_second = distance_covered / time_elapsed
        speed_km_per_hour = speed_meteres_per_second * 3.6

        # --- Update Total Distance State ---
        if object not in self.total_distance:
            self.total_distance[object] = {}
        if track_id not in self.total_distance[object]:
            # Initialize with accumulated distance up to the start point (if any)
            self.total_distance[object][track_id] = 0
        
        # Distance covered in the *last single frame*
        # (This is an approximation using the window average, but more accurate to track total distance)
        # Assuming speed is constant for the window, we add the distance proportional to the time between the 
        # second-to-last frame and the current frame.
        
        # Simpler approach for real-time: We only add the *new* distance covered by the last step
        
        # Distance covered since the previous calculation frame
        if len(self.track_history[object][track_id]) >= 2:
            prev_track_info = self.track_history[object][track_id][-2]
            prev_position = prev_track_info.get('position_transformed')
            
            if prev_position is not None:
                distance_increment = measure_distance(prev_position, end_position)
                self.total_distance[object][track_id] += distance_increment

        # --- Update Current Track Info ---
        current_track_info['speed'] = speed_km_per_hour
        current_track_info['distance'] = self.total_distance[object][track_id]

    def process_frame_tracks(self, current_frame_tracks):
        """
        Processes tracks for a single frame, updates speed/distance states, and increments frame counter.
        
        Args:
            current_frame_tracks (dict): Tracks for the current frame, e.g., tracks['players'][frame_num]
            
        Returns:
            dict: The updated tracks for the current frame with 'speed' and 'distance' keys added.
        """
        
        # Iterate through all objects (players, etc.) in the current frame
        for object, object_tracks in current_frame_tracks.items():
            if object == 0 or object == 3:
                continue 
            
            # Iterate through all tracks (track_id, track_info) in the object
            for track_id, track_info in object_tracks.items():
                # This function updates the states and adds 'speed'/'distance' to track_info
                self._update_speed_and_distance(object, track_id, track_info)
                
        self.current_frame_num += 1
        return current_frame_tracks

class SpeedDistanceEstimatorWorker:
    """
    Worker class that wraps RealTimeSpeedDistance_Estimator.
    Handles:
      - Estimator instantiation
      - Filtering tracks by class IDs
      - Preparing input for estimator and injecting results
    """

    def __init__(self, frame_rate=24, frame_window=5, excluded_class_ids=None, class_names=["ball", "goalkeeper", "player", "referee"]):
        """
        Args:
            frame_rate (int): Frame rate of the video.
            frame_window (int): Number of frames over which speed/distance is calculated.
            excluded_class_ids (list[int]): Class IDs to skip. All others are included by default.
        """
        self.CLASS_NAMES = class_names

        self.estimator = RealTimeSpeedDistance_Estimator(frame_window=frame_window, frame_rate=frame_rate)
        self.excluded_class_ids = set(excluded_class_ids or [])
        # self.orignal_result = None

    def process(self, final_result):
        """
        Main method to process tracking results and add speed/distance.

        Args:
            final_result (dict): Must contain 'boxes', 'cls', 'id'

        Returns:
            dict: final_result with 'speed' and 'distance' added
        """

        frame_tracks = self._prepare_tracks(final_result)
        updated_tracks = self.estimator.process_frame_tracks(frame_tracks)
        self._inject_tracks(final_result, updated_tracks)

        return final_result

    def _get_transformed_position(self, bbox):
        return ((bbox[0] + bbox[2]) / 2 * 0.1, bbox[3] * 0.1) 

    def _prepare_tracks(self, final_result):
        self.orignal_result = final_result
        frame_tracks = {}
        for box, cls_id, track_id in zip(final_result['boxes'], final_result['cls'], final_result['id']):
            class_name = self.CLASS_NAMES[int(cls_id)]
            if class_name in ["ball", "referee"]: continue 
            if class_name not in frame_tracks: frame_tracks[class_name] = {}
            frame_tracks[class_name][track_id] = {'bbox': box, 'position_transformed': self._get_transformed_position(box)}
        return frame_tracks

    def _inject_tracks(self, final_result, updated_tracks):
        speed_map = {}
        for obj_tracks in updated_tracks.values():
            for tid, info in obj_tracks.items():
                if 'speed' in info: speed_map[tid] = info
        
        final_result['speed'] = np.zeros(len(final_result['id']), dtype=np.float32)
        final_result['distance'] = np.zeros(len(final_result['id']), dtype=np.float32)
        for i, tid in enumerate(final_result['id']):
            if tid in speed_map:
                final_result['speed'][i] = speed_map[tid]['speed']
                final_result['distance'][i] = speed_map[tid]['distance']