"""
============
Information:
============
Project: Fire and Smoke Detection example code on MXA
File Name: fire_smoke_model.py
Author :Abdulrahman Hamza (Hamza)
============
Description:
============
A helper class to run Fire/Smoke Detection pre- and post-processing
for YOLOv8 Small 640x640 model on MX3.
"""

###################################################################################################

# Imports

import numpy as np
import cv2
import onnxruntime as ort

###################################################################################################

FIRE_SMOKE_CLASSES = ('smoke', 'fire')
POST_OUTPUT_KEYS = [
    (80, 80, 64),
    (40, 40, 64),
    (20, 20, 64),
    (80, 80, 2),
    (40, 40, 2),
    (20, 20, 2),
]

###################################################################################################
###################################################################################################
###################################################################################################

class FireSmokeModel:
    """
    A helper class to run Fire/Smoke Detection pre- and post-processing.
    """

###################################################################################################
    def __init__(self, stream_img_size=None, confidence_thres=0.25, iou_thres=0.45, post_model_path=None):
        """
        The initialization function.
        
        Args:
            stream_img_size: Tuple of (height, width, channels) for the input stream
            confidence_thres: Confidence threshold for detections
            iou_thres: IoU threshold for NMS
            post_model_path: Optional ONNX post-processing model path for cropped YOLO head outputs
        """

        self.input_width = 640
        self.input_height = 640
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        self.stream_mode = False
        self.img_width = None
        self.img_height = None
        self.length = None
        
        self.prev_detections = []
        self.smoothing_alpha = 0.6  
        self.min_iou_match = 0.3   

        self.post_model_path = post_model_path
        self.post_session = None
        self.post_input_names = []
        if self.post_model_path:
            self.post_session = ort.InferenceSession(
                self.post_model_path,
                providers=["CPUExecutionProvider"],
            )
            self.post_input_names = [inp.name for inp in self.post_session.get_inputs()]
        
        if stream_img_size:
            # Pre-calculate values for preprocessing
            self.preprocess(np.zeros(stream_img_size, dtype=np.uint8))
            self.stream_mode = True

###################################################################################################
    def _run_post_model(self, fmap):
        """
        Run ONNX post-processing model on six cropped-head outputs.
        """
        if self.post_session is None or len(fmap) < 6:
            return None

        outputs_by_shape = {}
        for tensor in fmap:
            arr = np.asarray(tensor)
            if arr.ndim == 4 and arr.shape[2] == 1:
                h, w, c = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[3])
                outputs_by_shape[(h, w, c)] = arr

        if any(key not in outputs_by_shape for key in POST_OUTPUT_KEYS):
            return None

        ordered_inputs = [outputs_by_shape[key] for key in POST_OUTPUT_KEYS]
        feed = {
            # HW1C -> [1, C, H*W] for the post model inputs
            name: np.ascontiguousarray(
                np.transpose(tensor, (2, 3, 0, 1)).reshape(1, tensor.shape[3], -1),
                dtype=np.float32,
            )
            for name, tensor in zip(self.post_input_names, ordered_inputs)
        }

        return self.post_session.run(None, feed)[0]

###################################################################################################
    def preprocess(self, img):
        """
        Fire/Smoke Detection Pre-processing.
        
        Args:
            img: Input image in BGR format (OpenCV default)
            
        Returns:
            Preprocessed image ready for inference (HWNC format, normalized)
        """
        # Get original image dimensions
        self.img_height, self.img_width = img.shape[:2]
        
        # Prepare a square image for inference (letterbox)
        self.length = max(self.img_height, self.img_width)
        image = np.zeros((self.length, self.length, 3), np.uint8)
        image[0:self.img_height, 0:self.img_width] = img

        # Preprocess using OpenCV's blobFromImage
        # - Normalize to 0-1
        # - Resize to 640x640
        # - Swap BGR to RGB
        img = cv2.dnn.blobFromImage(
            image, 
            scalefactor=1/255.0, 
            size=(self.input_width, self.input_height), 
            swapRB=True
        )

        # MxAccl stream input expects HWNC (e.g., [640, 640, 1, 3]).
        img = np.transpose(img, (2, 3, 0, 1))
        return np.ascontiguousarray(img, dtype=np.float32)

###################################################################################################
    def postprocess(self, fmap):
        if fmap is None or len(fmap) == 0:
            return []

        # For cropped-head DFP output, run the ONNX post model locally first.
        post_output = self._run_post_model(fmap)
        if post_output is None:
            post_output = fmap[0]
        
        # Normalize to [6, N]
        if post_output.ndim == 3 and post_output.shape[0] == 1:
            post_output = post_output[0]
        elif post_output.ndim == 4 and post_output.shape[2] == 1:
            post_output = np.transpose(post_output, (2, 3, 0, 1)).reshape(post_output.shape[3], -1)
        
        if post_output.ndim != 2:
            return []
        
        if post_output.shape[0] != 6 and post_output.shape[1] == 6:
            post_output = post_output.T
        if post_output.shape[0] != 6:
            return []
        
        outputs = post_output.T  # [N, 6]
        
        # Extract bounding boxes and class scores
        boxes = outputs[:, :4]  # [N, 4] -> x_center, y_center, width, height
        class_scores = outputs[:, 4:6]  # [N, 2] -> smoke, fire
        if class_scores.size == 0:
            return []
        
        # Apply sigmoid only if scores are logits (outside 0-1 range)
        # Post-processing model may already apply sigmoid
        if np.max(class_scores) > 1.0 or np.min(class_scores) < 0.0:
            class_scores = 1.0 / (1.0 + np.exp(-np.clip(class_scores, -50, 50)))
        
        # Calculate scaling factors
        x_factor = self.length / self.input_width
        y_factor = self.length / self.input_height
        
        # Find the class with the highest score for each detection
        max_scores = np.max(class_scores, axis=1)  # (8400,)
        class_ids = np.argmax(class_scores, axis=1)  # (8400,)
        
        # Filter by confidence threshold
        valid_indices = np.where(max_scores >= self.confidence_thres)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # Select only valid detections
        valid_boxes = boxes[valid_indices]
        valid_class_ids = class_ids[valid_indices]
        valid_scores = max_scores[valid_indices]
        
        # Convert from (x_center, y_center, w, h) to (left, top, width, height) for NMS
        # Scale to original image dimensions
        valid_boxes[:, 0] = (valid_boxes[:, 0] - valid_boxes[:, 2] / 2) * x_factor  # left
        valid_boxes[:, 1] = (valid_boxes[:, 1] - valid_boxes[:, 3] / 2) * y_factor  # top
        valid_boxes[:, 2] = valid_boxes[:, 2] * x_factor  # width
        valid_boxes[:, 3] = valid_boxes[:, 3] * y_factor  # height
        
        # Create detection list
        detections = []
        for i in range(len(valid_indices)):
            class_id = int(valid_class_ids[i])
            score = float(valid_scores[i])
            
            # Get class name
            if class_id < len(FIRE_SMOKE_CLASSES):
                class_name = FIRE_SMOKE_CLASSES[class_id]
            else:
                class_name = f'class_{class_id}'
            
            det = {
                'bbox': valid_boxes[i].astype(int).tolist(),  # [left, top, width, height]
                'class': class_name,
                'class_idx': class_id,
                'score': score
            }
            detections.append(det)
        
        # Apply NMS
        if len(detections) > 0:
            boxes_for_nms = [d['bbox'] for d in detections]
            scores_for_nms = [d['score'] for d in detections]
            
            indices = cv2.dnn.NMSBoxes(
                boxes_for_nms, 
                scores_for_nms, 
                self.confidence_thres, 
                self.iou_thres
            )
            
            if len(indices) > 0:
                # Flatten indices if needed (OpenCV returns nested arrays in some versions)
                if isinstance(indices[0], (list, np.ndarray)):
                    indices = [i[0] if isinstance(i, np.ndarray) else i for i in indices]
                
                final_detections = [detections[i] for i in indices]
            else:
                final_detections = []
        else:
            final_detections = []
        
        # Convert bbox from [left, top, width, height] to [left, top, right, bottom]
        for det in final_detections:
            l, t, w, h = det['bbox']
            det['bbox'] = [l, t, l + w, t + h]  # Use list for mutability
        
        # Apply temporal smoothing to reduce jitter
        final_detections = self._smooth_detections(final_detections)
        
        # Convert bbox to tuple
        for det in final_detections:
            det['bbox'] = tuple(det['bbox'])
        
        return final_detections

###################################################################################################
    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes [l, t, r, b]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0
        
        return inter_area / union_area

###################################################################################################
    def _smooth_detections(self, detections):
        if len(self.prev_detections) == 0:
            self.prev_detections = detections
            return detections
        
        smoothed = []
        used_prev = set()
        
        for det in detections:
            best_match = None
            best_iou = self.min_iou_match
            
            # Find matching detection from previous frame
            for i, prev_det in enumerate(self.prev_detections):
                if i in used_prev:
                    continue
                if det['class_idx'] != prev_det['class_idx']:
                    continue
                    
                iou = self._compute_iou(det['bbox'], prev_det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = i
            
            if best_match is not None:
                # Smooth the bounding box with EMA
                prev_box = self.prev_detections[best_match]['bbox']
                curr_box = det['bbox']
                
                smoothed_box = [
                    int(self.smoothing_alpha * curr_box[j] + (1 - self.smoothing_alpha) * prev_box[j])
                    for j in range(4)
                ]
                
                det['bbox'] = smoothed_box
                used_prev.add(best_match)
            
            smoothed.append(det)
        
        self.prev_detections = smoothed
        return smoothed

###################################################################################################
if __name__ == "__main__":
    pass

# eof
