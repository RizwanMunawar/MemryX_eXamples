import os
import cv2 as cv
import numpy as np
from pathlib import Path
from queue import Queue
from memryx import AsyncAccl
import argparse
import threading
import pyautogui
import threading

browser_lock = threading.Lock()

def scroll_up():
    with browser_lock:
        try:
            pyautogui.scroll(3)
            print("↑ Scroll UP")
        except Exception as e:
            print(f"Scroll error: {e}")

def scroll_down():
    with browser_lock:
        try:
            pyautogui.scroll(-3)
            print("↓ Scroll DOWN")
        except Exception as e:
            print(f"Scroll error: {e}")

def scroll_fast_up():
    with browser_lock:
        try:
            pyautogui.scroll(10)
        except:
            pass

def scroll_fast_down():
    with browser_lock:
        try:
            pyautogui.scroll(-10)
        except:
            pass


# ========== Pose + Gesture Application n ==========

class App:
    def __init__(self, cam, model_input_shape, mirror=False, src_is_cam=True, debug=False):
        self.cam = cam
        self.input_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.input_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
        self.model_input_shape = model_input_shape
        self.capture_queue = Queue(maxsize=5)
        self.mirror = mirror
        self.box_score = 0.25 
        self.kpt_score = 0.5
        self.nms_thr = 0.2
        self.src_is_cam = src_is_cam
        self.debug = debug

        # Gesture cooldown and tracking
        self.last_action = None
        self.last_time = cv.getTickCount()
        self.gesture_cooldown = 0.3
        
        self.consecutive_frames = {"right_up": 0, "left_up": 0, "both_up": 0, "none": 0}
        self.current_gesture = "none"

        # Skeleton keypoint pairs for drawing
        self.KEYPOINT_PAIRS = [
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
            (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
            (5, 11), (6, 12), (11, 12), (11, 13),
            (13, 15), (12, 14), (14, 16)
        ]

    def generate_frame(self):
        while True:
            ok, frame = self.cam.read()
            if not ok:
                print("Camera frame not captured.")
                return None
            if self.mirror:
                frame = cv.flip(frame, 1)
            if self.capture_queue.full():
                self.capture_queue.get()
            self.capture_queue.put(frame)
            out, self.ratio = self.preprocess_image(frame)
            return out

    def preprocess_image(self, image):
        h, w = image.shape[:2]
        r = min(self.model_input_shape[0] / h, self.model_input_shape[1] / w)
        image_resized = cv.resize(image, (int(w * r), int(h * r)), interpolation=cv.INTER_LINEAR)
        frame_rgb = cv.cvtColor(image_resized, cv.COLOR_BGR2RGB)
        padded_img = np.ones((self.model_input_shape[0], self.model_input_shape[1], 3), dtype=np.uint8) * 114
        padded_img[:int(h * r), :int(w * r)] = frame_rgb
        padded_img = np.transpose(padded_img, (2, 0, 1))
        padded_img = np.expand_dims(padded_img, axis=0).astype(np.float32) / 255.0
        return padded_img, r

    def xywh2xyxy(self, box):
        box_xyxy = box.copy()
        box_xyxy[..., 0] = box[..., 0] - box[..., 2] / 2
        box_xyxy[..., 1] = box[..., 1] - box[..., 3] / 2
        box_xyxy[..., 2] = box[..., 0] + box[..., 2] / 2
        box_xyxy[..., 3] = box[..., 1] + box[..., 3] / 2
        return box_xyxy

    def compute_iou(self, box, boxes):
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])
        inter_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - inter_area
        return inter_area / union_area

    def nms_process(self, boxes, scores, iou_thr):
        sorted_idx = np.argsort(scores)[::-1]
        keep_idx = []
        while sorted_idx.size > 0:
            idx = sorted_idx[0]
            keep_idx.append(idx)
            ious = self.compute_iou(boxes[idx, :], boxes[sorted_idx[1:], :])
            rest_idx = np.where(ious < iou_thr)[0]
            sorted_idx = sorted_idx[rest_idx + 1]
        return keep_idx

    def draw_skeleton(self, frame, kpt):
        """Draw skeleton on frame for visualization"""
        for pair in self.KEYPOINT_PAIRS:
            idx1, idx2 = pair
            x1, y1 = int(kpt[idx1*3]), int(kpt[idx1*3+1])
            x2, y2 = int(kpt[idx2*3]), int(kpt[idx2*3+1])
            
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        for i in range(17):
            x, y, conf = int(kpt[i*3]), int(kpt[i*3+1]), kpt[i*3+2]
            if x > 0 and y > 0 and conf > self.kpt_score:
                cv.circle(frame, (x, y), 5, (0, 0, 255), -1)
        
        return frame

    def process_model_output(self, *ofmaps):
        predict = ofmaps[0].squeeze(0).T
        predict = predict[predict[:, 4] > self.box_score, :]
        
        if self.capture_queue.empty():
            return
            
        frame = self.capture_queue.get()
        self.capture_queue.task_done()
        
        if predict.shape[0] == 0:
            # No detection
            self.draw_instructions(frame, "NO PERSON DETECTED")
            cv.imshow("Gesture Control", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                self.cleanup()
            return
            
        scores = predict[:, 4]
        boxes = self.xywh2xyxy(predict[:, 0:4] / self.ratio)
        kpts = predict[:, 5:]

        for i in range(kpts.shape[0]):
            for j in range(kpts.shape[1] // 3):
                if kpts[i, 3*j+2] < self.kpt_score:
                    kpts[i, 3*j:3*(j+1)] = [-1, -1, -1]
                else:
                    kpts[i, 3*j] /= self.ratio
                    kpts[i, 3*j+1] /= self.ratio

        idxes = self.nms_process(boxes, scores, self.nms_thr)
        boxes = boxes[idxes, :]
        kpts = kpts[idxes, :]

        for kpt in kpts:
            frame = self.draw_skeleton(frame, kpt)
            frame = self.detect_gesture_and_control(frame, kpt)

        self.draw_instructions(frame, self.current_gesture)
        cv.imshow("Gesture Control", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            self.cleanup()

    def draw_instructions(self, frame, gesture_state):
        """Draw gesture instructions on frame"""
        instructions = [
            "RIGHT hand UP: Scroll UP",
            "LEFT hand UP: Scroll DOWN", 
            "BOTH hands UP: Fast scroll UP",
            f"Current: {gesture_state}",
            "Press 'q' to quit"
        ]
        
        y_offset = 30
        for i, text in enumerate(instructions):
            cv.putText(frame, text, (10, y_offset + i*25), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv.putText(frame, text, (10, y_offset + i*25), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    def detect_gesture_and_control(self, frame, kpt):
        # Extract keypoints
        left_wrist = kpt[9*3:9*3+3]
        right_wrist = kpt[10*3:10*3+3]
        left_shoulder = kpt[5*3:5*3+3]
        right_shoulder = kpt[6*3:6*3+3]
        nose = kpt[0*3:0*3+3]  # Reference point

        # Check if critical keypoints are detected
        if not (right_wrist[2] > self.kpt_score and left_wrist[2] > self.kpt_score):
            self.current_gesture = "hands not visible"
            return frame
            
        if not (left_shoulder[2] > self.kpt_score and right_shoulder[2] > self.kpt_score):
            self.current_gesture = "shoulders not visible"
            return frame
        
        # Draw wrists for debugging
        cv.circle(frame, (int(left_wrist[0]), int(left_wrist[1])), 12, (255, 0, 0), -1)
        cv.circle(frame, (int(right_wrist[0]), int(right_wrist[1])), 12, (0, 255, 0), -1)
        cv.circle(frame, (int(left_shoulder[0]), int(left_shoulder[1])), 8, (255, 255, 0), -1)
        cv.circle(frame, (int(right_shoulder[0]), int(right_shoulder[1])), 8, (0, 255, 255), -1)
        
        # Calculate vertical distances
        left_diff = left_shoulder[1] - left_wrist[1]
        right_diff = right_shoulder[1] - right_wrist[1]
        
        # Display debug info
        cv.putText(frame, f"L_diff: {int(left_diff)}", (int(left_wrist[0])+15, int(left_wrist[1])), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv.putText(frame, f"R_diff: {int(right_diff)}", (int(right_wrist[0])+15, int(right_wrist[1])), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Threshold: wrist must be at least 40 pixels above shoulder
        threshold = 40
        
        right_hand_up = (right_diff > threshold)
        left_hand_up = (left_diff > threshold)
        
        # Determine gesture
        if right_hand_up and left_hand_up:
            gesture = "both_up"
        elif right_hand_up:
            gesture = "right_up"
        elif left_hand_up:
            gesture = "left_up"
        else:
            gesture = "none"
        
        self.current_gesture = gesture
        
        # Show detected state
        state_color = {
            "both_up": (255, 0, 255),
            "right_up": (0, 255, 0),
            "left_up": (255, 0, 0),
            "none": (128, 128, 128)
        }
        
        cv.rectangle(frame, (10, 180), (300, 220), state_color.get(gesture, (0,0,0)), -1)
        cv.putText(frame, f"GESTURE: {gesture}", (15, 205), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Trigger actions
        if gesture == "both_up":
            self.trigger_action("fast_scroll_up", scroll_fast_up, frame, cooldown=0.4)
        elif gesture == "right_up":
            self.trigger_action("scroll_up", scroll_up, frame)
        elif gesture == "left_up":
            self.trigger_action("scroll_down", scroll_down, frame)
        
        return frame

    def trigger_action(self, action_name, func, frame, cooldown=None):
        """Trigger an action with cooldown protection"""
        if cooldown is None:
            cooldown = self.gesture_cooldown
            
        now = cv.getTickCount()
        time_diff = (now - self.last_time) / cv.getTickFrequency()
        
        if self.last_action != action_name or time_diff > cooldown:
            # Execute action
            threading.Thread(target=func, daemon=True).start()
            
            # Visual feedback
            cv.putText(frame, f"ACTION: {action_name.upper()}", (10, frame.shape[0] - 20), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
            
            self.last_action = action_name
            self.last_time = now

    def cleanup(self):
        """Clean up resources"""
        self.cam.release()
        cv.destroyAllWindows()
        exit(0)


# ========== Memryx Runtime Setup ==========

def run_mxa(dfp, post_model, app):
    pose_dfp = AsyncAccl(dfp=dfp)
    pose_dfp.set_postprocessing_model(post_model, model_idx=0)
    pose_dfp.connect_input(app.generate_frame)
    pose_dfp.connect_output(app.process_model_output)
    pose_dfp.wait()


# ========== Entry Point ==========

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gesture-based web control")
    parser.add_argument("--dfp", type=str, default="../../models/YOLO_v8_medium_pose_640_640_3_onnx.dfp")
    parser.add_argument("--post_model", type=str, default="../../models/YOLO_v8_medium_pose_640_640_3_onnx_post.onnx")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()

    cam = cv.VideoCapture(args.camera)
    if not cam.isOpened():
        print(f" Error: Could not open camera {args.camera}")
        exit(1)
    
    model_input_shape = (640, 640)
    app = App(cam, model_input_shape, mirror=not args.no_mirror, 
              src_is_cam=True, debug=args.debug)
    
    try:
        run_mxa(args.dfp, args.post_model, app)
    except KeyboardInterrupt:
        print("\n\n Interrupted by user")
        app.cleanup()
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        app.cleanup()