import os
import sys
import threading
import queue
import cv2
import argparse
import numpy as np
from typing import Callable, Optional

from mx_pose import Mxpose
from coco17 import coco17_dict

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QCheckBox, QPushButton, QFileDialog, QFrame, QSizePolicy
)
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtCore import QTimer, Qt


# ------------------------- Viewer / Pipeline -------------------------

class Viewer:
    """
    Worker that:
      - reads frames from camera/video/image
      - pushes raw frames into mx_pose
      - pulls annotated results from mx_pose
      - publishes latest annotated result to output_queue for the Qt UI

    If a *video file* ends, calls on_video_end() (if provided).
    """
    def __init__(
        self,
        mx_pose: Mxpose,
        source,
        fps=30,
        on_video_end: Optional[Callable[[], None]] = None
    ):
        self.mx_pose = mx_pose
        self.source = source
        self.fps = fps
        self.on_video_end = on_video_end

        self.keypoint_info = coco17_dict['keypoint_info']
        self.skeleton_info = coco17_dict['skeleton_info']

        self.stop_event = threading.Event()
        self.thread = None

        # UI reads from here (never from mx_pose directly)
        self.output_queue = queue.Queue(maxsize=2)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _is_video_file(self) -> bool:
        if not isinstance(self.source, str):
            return False
        s = self.source.lower()
        return s.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'))

    def _is_image_file(self) -> bool:
        if not isinstance(self.source, str):
            return False
        s = self.source.lower()
        return s.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))

    def _video_capture(self):
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        return cap

    def _publish_latest(self, annotated_frame):
        # Keep only the latest frame
        try:
            while True:
                self.output_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.output_queue.put_nowait(annotated_frame)
        except queue.Full:
            pass

    def _run(self):
        src = self.source

        # image mode
        if self._is_image_file():
            frame = cv2.imread(src)
            if frame is None:
                print(f"Error: Could not read image file: {src}")
                return

            try:
                if not self.mx_pose.full():
                    self.mx_pose.put(frame.copy())
            except Exception as e:
                print(f"Error pushing image to pipeline: {e}")
                return

            while not self.stop_event.is_set():
                try:
                    if not self.mx_pose.empty():
                        annotated = self.mx_pose.get()
                        self._publish_latest(annotated)
                        break
                except Exception:
                    break
            return

        # video/cam mode
        cap = self._video_capture()
        if not cap.isOpened():
            print(f"Error: Could not open source: {src}")
            return

        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        if cap_fps and cap_fps > 1:
            self.fps = int(cap_fps)

        ended_naturally = False

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                ended_naturally = True
                break

            try:
                if not self.mx_pose.full():
                    self.mx_pose.put(frame.copy())
            except Exception:
                break

            try:
                if not self.mx_pose.empty():
                    annotated = self.mx_pose.get()
                    self._publish_latest(annotated)
            except Exception:
                break

        cap.release()

        # If a video file ended, call the callback to go back to cam
        if ended_naturally and self._is_video_file() and (self.on_video_end is not None):
            try:
                self.on_video_end()
            except Exception as e:
                print(f"on_video_end callback error: {e}")

    # ---------- Drawing with independent toggles (COLORS UNCHANGED) ----------

    def draw_overlays(self, base_img, annotated_frame,
                      show_bbox=True, show_pose=True, show_label=True,
                      kpt_thr=0.5, radius=2, line_width=2):
        img = base_img
        for i in range(getattr(annotated_frame, "num_detections", 0)):
            pose = annotated_frame.poses[i]
            img = self.draw_mmpose(
                img=img,
                bbox=pose.bbox,
                keypoints=pose.kpots[0],
                scores=pose.score,
                label=pose.name,
                keypoint_info=self.keypoint_info,
                skeleton_info=self.skeleton_info,
                show_bbox=show_bbox,
                show_pose=show_pose,
                show_label=show_label,
                kpt_thr=kpt_thr,
                radius=radius,
                line_width=line_width
            )
        return img

    def draw_mmpose(self, img, bbox, keypoints, scores, label,
                    keypoint_info, skeleton_info,
                    show_bbox=True, show_pose=True, show_label=True,
                    kpt_thr=0.5, radius=2, line_width=2):

        # label color UNCHANGED
        if show_label:
            text_x = int(bbox[0])
            text_y = int(bbox[1]) - 10 if int(bbox[1]) - 10 > 10 else int(bbox[1]) + 10
            cv2.putText(img, str(label), (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2.LINE_AA)

        # bbox color UNCHANGED
        if show_bbox:
            cv2.rectangle(
                img,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (128, 0, 128),
                3
            )

        # keypoints + skeleton colors UNCHANGED (from coco17_dict)
        if show_pose:
            assert len(keypoints.shape) == 2
            vis_kpt = [s >= kpt_thr for s in scores][0]

            link_dict = {}
            for idx, kpt_info in keypoint_info.items():
                kpt_color = tuple(kpt_info['color'])
                link_dict[kpt_info['name']] = kpt_info['id']
                kpt = keypoints[idx]
                if vis_kpt[idx]:
                    cv2.circle(img, (int(kpt[0]), int(kpt[1])), int(radius), kpt_color, -1)

            for _, ske_info in skeleton_info.items():
                link = ske_info['link']
                pt0, pt1 = link_dict[link[0]], link_dict[link[1]]
                if vis_kpt[pt0] and vis_kpt[pt1]:
                    link_color = ske_info['color']
                    kpt0 = keypoints[pt0]
                    kpt1 = keypoints[pt1]
                    cv2.line(
                        img,
                        (int(kpt0[0]), int(kpt0[1])),
                        (int(kpt1[0]), int(kpt1[1])),
                        link_color,
                        thickness=line_width
                    )
        return img


# ------------------------- Manager -------------------------

class ViewerManager:
    """
    Rebuilds (mx_pose, viewer) on source switch to avoid deadlocks.
    Also supports auto-switch-back-to-camera when a loaded video ends.
    """
    def __init__(self, mx_modeldir: str, source, fps=30):
        self.mx_modeldir = mx_modeldir
        self.fps = fps
        self.viewer: Optional[Viewer] = None
        self.mx_pose: Optional[Mxpose] = None
        self.source = source

        self.build_new(source)

    def build_new(self, source):
        if self.viewer is not None:
            self.viewer.stop()

        if self.mx_pose is not None:
            try:
                self.mx_pose.stop()
            except Exception:
                pass

        def on_video_end():
            # move back to camera in a background thread
            threading.Thread(
                target=lambda: self.build_new(self.source),
                daemon=True
            ).start()

        self.mx_pose = Mxpose(mx_modeldir=self.mx_modeldir)
        self.viewer = Viewer(mx_pose=self.mx_pose, source=source, fps=self.fps, on_video_end=on_video_end)
        self.viewer.start()
        self.source = source

    def stop_all(self):
        if self.viewer is not None:
            self.viewer.stop()
        if self.mx_pose is not None:
            try:
                self.mx_pose.stop()
            except Exception:
                pass


# ------------------------- Light Blue UI Style (ONLY UI) -------------------------

APP_QSS = """
QMainWindow {
    background: #eef5ff;
    color: #0f172a;
}

QLabel#FrameLabel {
    background: #ffffff;
    border: 2px solid #b7d5ff;
    border-radius: 14px;
}

QFrame#SidePanel {
    background: #f7fbff;
    border: 2px solid #b7d5ff;
    border-radius: 16px;
}

QLabel#PanelTitle {
    font-size: 18px;
    font-weight: 800;
    color: #0b3c6d;
}

QCheckBox {
    font-size: 15px;
    font-weight: 600;
    spacing: 10px;
    color: #0f172a;
}

QCheckBox::indicator {
    width: 22px;
    height: 22px;
}

QCheckBox::indicator:unchecked {
    border: 2px solid #5b9bff;
    background: #ffffff;
    border-radius: 5px;
}

QCheckBox::indicator:checked {
    border: 2px solid #1e88e5;
    background: #1e88e5;
    border-radius: 5px;
}

QPushButton {
    background: #ffffff;
    border: 2px solid #5b9bff;
    color: #08335e;
    padding: 12px 14px;
    border-radius: 12px;
    font-size: 15px;
    font-weight: 800;
}

QPushButton:hover {
    background: #e7f1ff;
}

QPushButton#PrimaryButton {
    background: #1e88e5;
    color: white;
    border: 2px solid #1e88e5;
}

QPushButton#PrimaryButton:hover {
    background: #1874c7;
    border: 2px solid #1874c7;
}

QPushButton#SecondaryButton {
    background: #ffffff;
    color: #08335e;
    border: 2px solid #b7d5ff;
}

QPushButton#SecondaryButton:hover {
    background: #f0f7ff;
}

QPushButton#DangerButton {
    background: #ffffff;
    border: 2px solid #ff6b6b;
    color: #b00020;
}

QPushButton#DangerButton:hover {
    background: #ffecec;
}

QStatusBar {
    background: #f0f7ff;
    border-top: 2px solid #b7d5ff;
    font-size: 13px;
    color: #0b3c6d;
}
"""


# ------------------------- UI -------------------------

class ViewerUI(QMainWindow):
    def __init__(self, manager: ViewerManager):
        super().__init__()
        self.manager = manager

        self.setWindowTitle("MemryX Pose Viewer")
        self.setMinimumSize(1200, 760)

        # Recording state
        self.recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.record_path: Optional[str] = None
        self.record_fps: float = 30.0

        self._build_ui()
        self._apply_style()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self._update_status("Ready")

    def _apply_style(self):
        self.setStyleSheet(APP_QSS)
        font = QFont()
        font.setPointSize(11)
        self.setFont(font)

    def _build_ui(self):
        root = QWidget(self)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(16)

        # Frame view
        self.frame_label = QLabel("Waiting for frames…", self)
        self.frame_label.setObjectName("FrameLabel")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root_layout.addWidget(self.frame_label, stretch=4)

        # Side panel
        side = QFrame(self)
        side.setObjectName("SidePanel")
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(18, 18, 18, 18)
        side_layout.setSpacing(14)

        title = QLabel("Display Options", self)
        title.setObjectName("PanelTitle")
        side_layout.addWidget(title)

        self.bbox_checkbox = QCheckBox("Bounding Boxes", self)
        self.bbox_checkbox.setChecked(True)
        side_layout.addWidget(self.bbox_checkbox)

        self.pose_checkbox = QCheckBox("Keypoints + Skeleton", self)
        self.pose_checkbox.setChecked(True)
        side_layout.addWidget(self.pose_checkbox)

        self.label_checkbox = QCheckBox("Labels", self)
        self.label_checkbox.setChecked(True)
        side_layout.addWidget(self.label_checkbox)

        self.black_bg_checkbox = QCheckBox("Black Background (overlays only)", self)
        self.black_bg_checkbox.setChecked(False)
        side_layout.addWidget(self.black_bg_checkbox)

        # Buttons
        self.load_button = QPushButton("Load Source…", self)
        self.load_button.setObjectName("PrimaryButton")
        self.load_button.clicked.connect(self.load_source)
        side_layout.addWidget(self.load_button)

        self.record_button = QPushButton("Start Recording…", self)
        self.record_button.setObjectName("SecondaryButton")
        self.record_button.clicked.connect(self.toggle_recording)
        side_layout.addWidget(self.record_button)

        self.quit_button = QPushButton("Quit", self)
        self.quit_button.setObjectName("DangerButton")
        self.quit_button.clicked.connect(self.close)
        side_layout.addWidget(self.quit_button)

        side_layout.addStretch(1)

        root_layout.addWidget(side, stretch=1)

        self.setCentralWidget(root)
        self.statusBar().showMessage("Ready")

    def _update_status(self, msg=None):
        src = self.manager.source
        rec = "REC ●" if self.recording else "REC off"
        if msg:
            self.statusBar().showMessage(f"{msg}   |   Source: {src}   |   {rec}")
        else:
            self.statusBar().showMessage(f"Source: {src}   |   {rec}")

    # -------- Recording --------

    def _stop_recording(self, reason: str = ""):
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                pass
        self.video_writer = None
        self.recording = False
        self.record_path = None
        self.record_button.setText("Start Recording…")
        self._update_status(reason or "Recording stopped")

    def toggle_recording(self):
        if self.recording:
            self._stop_recording("Recording saved")
            return

        # Choose save path
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Recording",
            "output.mp4",
            "MP4 Video (*.mp4)"
        )
        if not path:
            return
        if not path.lower().endswith(".mp4"):
            path += ".mp4"

        # We will initialize VideoWriter on the next received frame (need size)
        self.record_path = path
        self.recording = True
        self.video_writer = None
        self.record_button.setText("Stop Recording")
        self._update_status("Recording armed (waiting for frame)")

    # -------- Source switching --------

    def load_source(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Source",
            "",
            "Media Files (*.mp4 *.avi *.mov *.mkv *.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)"
        )
        if not file_path:
            return

        # Stop recording on source switch (safer; avoids mixed streams)
        if self.recording:
            self._stop_recording("Stopped recording (source changed)")

        self._update_status("Loading…")

        def rebuild():
            self.manager.build_new(file_path)

        threading.Thread(target=rebuild, daemon=True).start()

    def closeEvent(self, event):
        if self.recording:
            self._stop_recording("Recording stopped (exit)")
        self.manager.stop_all()
        super().closeEvent(event)

    # -------- Frame update --------

    def update_frame(self):
        self._update_status()

        viewer = self.manager.viewer
        if viewer is None:
            return

        try:
            annotated = viewer.output_queue.get_nowait()
        except queue.Empty:
            return

        src_img = getattr(annotated, "image", None)
        if src_img is None:
            return

        show_bbox = self.bbox_checkbox.isChecked()
        show_pose = self.pose_checkbox.isChecked()
        show_label = self.label_checkbox.isChecked()
        black_bg = self.black_bg_checkbox.isChecked()

        base = np.zeros_like(src_img) if black_bg else src_img.copy()

        if (show_bbox or show_pose or show_label) and getattr(annotated, "num_detections", 0) > 0:
            out_bgr = viewer.draw_overlays(
                base_img=base,
                annotated_frame=annotated,
                show_bbox=show_bbox,
                show_pose=show_pose,
                show_label=show_label
            )
        else:
            out_bgr = base

        # Initialize VideoWriter lazily when recording starts and we have a frame size
        if self.recording:
            if self.video_writer is None and self.record_path is not None:
                h, w = out_bgr.shape[:2]
                # Use viewer fps if available, otherwise fallback
                fps = float(getattr(viewer, "fps", 30) or 30)
                self.record_fps = fps
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.record_path, fourcc, fps, (w, h))
                if not self.video_writer.isOpened():
                    self._stop_recording("Failed to start recording (writer open error)")
                else:
                    self._update_status(f"Recording to {os.path.basename(self.record_path)}")

            # Write frame
            if self.video_writer is not None:
                try:
                    self.video_writer.write(out_bgr)
                except Exception:
                    self._stop_recording("Recording stopped (write error)")

        # Display in Qt
        rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self.frame_label.setText("")
        self.frame_label.setPixmap(pixmap.scaled(
            self.frame_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))


# ------------------------- Main -------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Viewer with a specified source.")
    parser.add_argument('--source', type=str, default='/dev/video0',
                        help="Source: camera device (e.g. /dev/video0 or 0), image path, or video path.")
    args = parser.parse_args()

    pose_dir = os.path.dirname(os.path.dirname(os.getcwd()))

    mx_modeldir = os.path.join(pose_dir,'video_inference/rtmpose_estimate')
    
    source = args.source 

    app = QApplication(sys.argv)
    manager = ViewerManager(mx_modeldir=mx_modeldir, source=source, fps=30)

    ui = ViewerUI(manager)
    ui.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
