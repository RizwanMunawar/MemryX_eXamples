"""
Traffic Analysis Application with GUI
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.pop("QT_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

import argparse
import cv2
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""
import numpy as np
from collections import defaultdict
from MXObb import MXObb
import time
import threading
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from gui import TrafficAnalysisGUI



# --- ASYNC FRAME FEEDER ---
class FrameFeeder(threading.Thread):
    def __init__(self, cap, mx_obb):
        super().__init__()
        self.cap = cap
        self.mx_obb = mx_obb
        self.stopped = False
        self.daemon = True

    def run(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.mx_obb.input_q.put(None)
                break
            self.mx_obb.put(frame)


# --- KALMAN FILTER & SORT TRACKING ---
class KalmanBoxTracker(object):
    count = 0

    def __init__(self, bbox):
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.5
        self.kf.errorCovPost = np.eye(7, dtype=np.float32)
        self.kf.statePost = np.array(
            [bbox[0], bbox[1], bbox[2] * bbox[3], bbox[2] / bbox[3], 0, 0, 0],
            np.float32
        ).reshape((7, 1))
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        s = bbox[2] * bbox[3]
        r = bbox[2] / bbox[3]
        self.kf.correct(np.array([bbox[0], bbox[1], s, r], np.float32).reshape((4, 1)))

    def predict(self):
        if (self.kf.statePost[6] + self.kf.statePost[2]) <= 0:
            self.kf.statePost[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.get_state())
        return self.history[-1]

    def get_state(self):
        s = self.kf.statePost[2, 0]
        r = self.kf.statePost[3, 0]
        if s < 0:
            s = 0
        if r <= 0:
            r = 1.0
        w = np.sqrt(s * r)
        h = np.sqrt(s / r)
        return [self.kf.statePost[0, 0], self.kf.statePost[1, 0], w, h]


class SortTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.15):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(detections, trks)
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(detections[d, :][0])
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        ret = []
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def associate_detections_to_trackers(self, detections, trackers):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        iou_matrix = self.iou_batch(detections, trackers)

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = np.array([[x, y] for x, y in enumerate(np.argmax(iou_matrix, axis=1))
                                            if iou_matrix[x, y] >= self.iou_threshold])
        else:
            matched_indices = np.empty((0, 2))

        unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
        unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def iou_batch(self, bb_test, bb_gt):
        bb_test = np.expand_dims(bb_test[:, :4], 1)
        bb_gt = np.expand_dims(bb_gt[:, :4], 0)

        xx1 = np.maximum(bb_test[..., 0] - bb_test[..., 2] / 2, bb_gt[..., 0] - bb_gt[..., 2] / 2)
        yy1 = np.maximum(bb_test[..., 1] - bb_test[..., 3] / 2, bb_gt[..., 1] - bb_gt[..., 3] / 2)
        xx2 = np.minimum(bb_test[..., 0] + bb_test[..., 2] / 2, bb_gt[..., 0] + bb_gt[..., 2] / 2)
        yy2 = np.minimum(bb_test[..., 1] + bb_test[..., 3] / 2, bb_gt[..., 1] + bb_gt[..., 3] / 2)

        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h

        area_test = bb_test[..., 2] * bb_test[..., 3]
        area_gt = bb_gt[..., 2] * bb_gt[..., 3]

        union = area_test + area_gt - wh
        union[union <= 0] = 1e-6
        return wh / union


# --- INTERSECTION ZONE ---
class IntersectionZone:
    """Zone class for traffic analysis"""

    def __init__(self, name, rect, color):
        self.name = name
        self.rect = rect  # (x, y, w, h)
        self.color = color  # BGR tuple

        x, y, w, h = rect
        self.polygon = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.int32).reshape((-1, 1, 2))

        self.incoming_counts = defaultdict(int)
        self.source_colors = {}

    def contains_point(self, point):
        return cv2.pointPolygonTest(self.polygon, point, False) >= 0

    @classmethod
    def from_gui_zone(cls, gui_zone):
        """Create from GUI ZoneData"""
        rect = (gui_zone.rect.x(), gui_zone.rect.y(),
                gui_zone.rect.width(), gui_zone.rect.height())
        # Convert QColor to BGR
        color = (gui_zone.color.blue(), gui_zone.color.green(), gui_zone.color.red())
        return cls(gui_zone.name, rect, color)


class TrafficAnalyzer:
    """Traffic analysis engine that integrates with PyQt GUI"""

    def __init__(self, video_path, models_dir, gui):
        self.video_path = video_path
        self.models_dir = models_dir
        self.gui = gui

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.tracker = SortTracker(max_age=20, min_hits=3, iou_threshold=0.15)
        self.track_history = defaultdict(list)

        self.vehicle_colors = {}
        self.object_zone_history = {}
        self.vehicle_classes = {'large vehicle', 'small vehicle'}

        self.zones = []
        self.zone_overlay = None

        # State
        self.feeder = None
        self.mx_obb = None
        self.running = False
        self.shutting_down = False
        self.prev_time = 0
        self.curr_fps = 0

        # Show first frame for zone setup
        self.show_setup_frame()

    def show_setup_frame(self):
        """Display first frame for zone setup"""
        ret, frame = self.cap.read()
        if ret:
            self.gui.update_frame(frame)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def convert_gui_zones(self, preserve_stats: bool = True):
        """Convert GUI zones to analysis zones.

        If preserve_stats is True, keeps incoming_counts/source_colors for zones with the same name.
        """
        old_by_name = {z.name: z for z in getattr(self, "zones", [])} if preserve_stats else {}

        new_zones = []
        for gui_zone in self.gui.get_zones():
            zone = IntersectionZone.from_gui_zone(gui_zone)
            if preserve_stats and zone.name in old_by_name:
                prev = old_by_name[zone.name]
                zone.incoming_counts = prev.incoming_counts
                zone.source_colors = prev.source_colors
            new_zones.append(zone)

        self.zones = new_zones

        # If a zone was removed, drop any object->zone bindings to it
        zone_names = {z.name for z in self.zones}
        for obj_id, zname in list(self.object_zone_history.items()):
            if zname not in zone_names:
                self.object_zone_history.pop(obj_id, None)
                self.vehicle_colors[obj_id] = (200, 200, 200)

        self.zone_overlay = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        self._draw_static_zones(self.zone_overlay)

    def on_zones_changed(self, _zones):
        """Called when GUI zones are edited while analysis is running."""
        self.convert_gui_zones(preserve_stats=True)

    def _draw_static_zones(self, canvas):
        for zone in self.zones:
            cv2.fillPoly(canvas, [zone.polygon], zone.color)
            cv2.polylines(canvas, [zone.polygon], True, zone.color, 1, cv2.LINE_AA)

    def update_logic(self, object_id, centroid):
        current_zone_obj = None
        for zone in self.zones:
            if zone.contains_point(centroid):
                current_zone_obj = zone
                break

        current_zone_name = current_zone_obj.name if current_zone_obj else None

        if object_id in self.object_zone_history:
            last_zone_name = self.object_zone_history[object_id]
            if current_zone_name and last_zone_name and current_zone_name != last_zone_name:
                last_color = self.vehicle_colors.get(object_id)
                if last_color and last_color != (200, 200, 200):
                    current_zone_obj.incoming_counts[last_zone_name] += 1
                    current_zone_obj.source_colors[last_zone_name] = last_color

        if current_zone_obj:
            self.vehicle_colors[object_id] = current_zone_obj.color
        elif object_id not in self.vehicle_colors:
            self.vehicle_colors[object_id] = (200, 200, 200)

        if current_zone_name:
            self.object_zone_history[object_id] = current_zone_name

    def draw_visuals(self, frame, annotated_frame, active_tracks):
        curr_time = time.time()
        if curr_time - self.prev_time > 0:
            self.curr_fps = 1.0 / (curr_time - self.prev_time)
        self.prev_time = curr_time

        if self.zone_overlay is not None:
            cv2.addWeighted(frame, 1.0, self.zone_overlay, 0.3, 0, frame)
            for zone in self.zones:
                cv2.polylines(frame, [zone.polygon], True, zone.color, 2, cv2.LINE_AA)

        for zone in self.zones:
            if not zone.incoming_counts:
                continue

            x, y, w, h = zone.rect
            cx, cy = int(x + w / 2), int(y + h / 2)

            items_to_draw = []
            for src_name, count in zone.incoming_counts.items():
                bg_color = zone.source_colors.get(src_name, (50, 50, 50))
                items_to_draw.append((str(count), bg_color))

            if not items_to_draw:
                continue

            box_w, box_h = 35, 25
            spacing = 5
            is_vertical = h > w

            if is_vertical:
                total_h = len(items_to_draw) * (box_h + spacing) - spacing
                start_y = cy - total_h // 2

                for i, (text, color) in enumerate(items_to_draw):
                    curr_y = start_y + i * (box_h + spacing)
                    cv2.rectangle(frame, (cx - box_w // 2, curr_y),
                                  (cx + box_w // 2, curr_y + box_h), color, -1)
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    tx = cx - text_size[0] // 2
                    ty = curr_y + box_h // 2 + text_size[1] // 2
                    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                total_w = len(items_to_draw) * (box_w + spacing) - spacing
                start_x = cx - total_w // 2

                for i, (text, color) in enumerate(items_to_draw):
                    curr_x = start_x + i * (box_w + spacing)
                    cv2.rectangle(frame, (curr_x, cy - box_h // 2),
                                  (curr_x + box_w, cy + box_h // 2), color, -1)
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    tx = curr_x + box_w // 2 - text_size[0] // 2
                    ty = cy + text_size[1] // 2
                    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 2, cv2.LINE_AA)

        det_map = []
        for det in annotated_frame.detections:
            if det.cls in self.vehicle_classes:
                det_map.append(det)

        for track in active_tracks:
            cx, cy, w, h, obj_id = track
            cx, cy = int(cx), int(cy)
            obj_id = int(obj_id)

            self.track_history[obj_id].append((cx, cy))
            if len(self.track_history[obj_id]) > 60:
                self.track_history[obj_id].pop(0)

            color = self.vehicle_colors.get(obj_id, (200, 200, 200))

            if len(self.track_history[obj_id]) > 2:
                points = np.array(self.track_history[obj_id], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], False, color, thickness=1, lineType=cv2.LINE_AA)

            matched_det = None
            min_dist = 10000
            for det in det_map:
                d_cx, d_cy = det.bbox[0], det.bbox[1]
                dist = np.hypot(cx - d_cx, cy - d_cy)
                if dist < 50 and dist < min_dist:
                    min_dist = dist
                    matched_det = det

            box_points = None
            if matched_det:
                x_c, y_c, w_det, h_det = matched_det.bbox
                angle = np.degrees(matched_det.rot)
                box_points = cv2.boxPoints(((x_c, y_c), (w_det, h_det), angle)).astype(int)
            else:
                top_left = (int(cx - w / 2), int(cy - h / 2))
                bottom_right = (int(cx + w / 2), int(cy + h / 2))
                box_points = np.array([top_left, (bottom_right[0], top_left[1]),
                                       bottom_right, (top_left[0], bottom_right[1])])

            cv2.drawContours(frame, [box_points], 0, color, 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 2, color, -1, cv2.LINE_AA)

            label_text = f"{obj_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

            lx = cx - text_w // 2
            ly = cy - int(h / 2) - 10

            pad = 4
            cv2.rectangle(frame, (lx - pad, ly - text_h - pad), (lx + text_w + pad, ly + pad), color, -1)
            cv2.putText(frame, label_text, (lx, ly), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        return frame

    def start_analysis(self):
        """Initialize and start the analysis"""
        if self.running:
            return

        print("Loading AI Model...")
        self.mx_obb = MXObb(self.models_dir)

        # Convert GUI zones to analysis zones
        self.convert_gui_zones()

        # Reset video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Start frame feeder
        self.feeder = FrameFeeder(self.cap, self.mx_obb)
        self.feeder.start()

        self.running = True
        self.prev_time = time.time()

        print(f"Processing: {self.video_path}")

    def process_frame(self):
        """Process a single frame - called by timer"""
        if not self.running or self.shutting_down:
            return False

        if not self.gui.is_analysis_running():
            return True  # Still alive but paused

        try:
            # Non-blocking get with timeout
            annotated_frame = self.mx_obb.get(block=True, timeout=0.05)

            if annotated_frame is None or annotated_frame.image is None:
                self.running = False
                return False

            frame = annotated_frame.image

            dets_to_track = []
            for det in annotated_frame.detections:
                if det.cls in self.vehicle_classes:
                    d = [det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3], det.conf]
                    dets_to_track.append(d)

            dets_to_track = np.array(dets_to_track)
            if len(dets_to_track) == 0:
                dets_to_track = np.empty((0, 5))

            active_tracks = self.tracker.update(dets_to_track)
            for track in active_tracks:
                self.update_logic(int(track[4]), (int(track[0]), int(track[1])))

            display_frame = self.draw_visuals(frame, annotated_frame, active_tracks)

            # Update GUI
            self.gui.update_frame(display_frame)
            self.gui.update_fps(self.curr_fps)

            # Update zone stats in GUI
            zones_data = []
            for zone in self.zones:
                zones_data.append({
                    'name': zone.name,
                    'color': zone.color,  # BGR tuple
                    'incoming_counts': dict(zone.incoming_counts),
                    'source_colors': zone.source_colors
                })
            self.gui.update_zone_stats(zones_data)

            return True

        except Exception:
            return True


class Application:
    """Main application controller"""

    def __init__(self, video_path, models_dir):
        self.app = QApplication(sys.argv)
        self.app.setStyle('Fusion')

        # Create GUI
        self.gui = TrafficAnalysisGUI()

        # Create analyzer
        self.analyzer = TrafficAnalyzer(video_path, models_dir, self.gui)

        # Allow editing zones while analysis is running (e.g., delete a zone mid-run)
        self.gui.zones_changed.connect(self.analyzer.on_zones_changed)

        # Connect start button to analyzer
        self.gui.start_btn.clicked.connect(self.on_start_clicked)
        
        # Connect shutdown signal to application shutdown (not just analyzer shutdown)
        self.gui.shutdown_requested.connect(self.on_shutdown_requested)

        # Processing timer
        self.process_timer = QTimer()
        self.process_timer.timeout.connect(self.process_loop)

    def on_start_clicked(self):
        """Handle start button click"""
        if not self.analyzer.running:
            # First time starting
            self.analyzer.start_analysis()
            self.process_timer.start(1)  # ~1000 FPS max polling rate

        # Toggle analysis state in GUI
        self.gui.start_analysis()
    
    def on_shutdown_requested(self):
        """Handle shutdown button click"""
        self.process_timer.stop()
        QTimer.singleShot(0, self._shutdown_and_close)

    def process_loop(self):
        """Main processing loop"""
        if not self.analyzer.process_frame():
            self.process_timer.stop()
            self.gui.status_label.setText("● Video Complete")
            self.gui.status_label.setStyleSheet("""
                color: #4361ee;
                font-size: 14px;
                font-weight: bold;
                padding: 5px 15px;
            """)
            # Video ended - shutdown and close application
            print("Video processing complete. Closing application...")
            # Perform shutdown in a separate call to avoid blocking
            QTimer.singleShot(100, self._shutdown_and_close)
    
    def _shutdown_and_close(self):
        """Shutdown analyzer and close application"""
  

        self.analyzer.shutting_down = True
        self.analyzer.running = False
        if self.analyzer.feeder:
            self.analyzer.feeder.stopped = True
        
        self.process_timer.stop()
        
        try:
            self.gui.close()
            self.app.quit()
            sys.stdout.flush()
            sys.stderr.flush()
        except:
            pass
        
        os._exit(0)

    def run(self):
        """Run the application"""
        # Explicitly size window to available screen geometry so it starts maximized
        screen_geo = self.app.primaryScreen().availableGeometry()
        self.gui.setGeometry(screen_geo)
        self.gui.show()
        return self.app.exec_()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Traffic Analysis with GUI')
    parser.add_argument('--video_path', default='../../assets/traffic.mp4', type=str,
                        help='Path to input video')
    parser.add_argument('--models_dir', type=str, default='../../models',
                        help='Path to models directory')
    args = parser.parse_args()

    app = Application(args.video_path, args.models_dir)
    sys.exit(app.run())
