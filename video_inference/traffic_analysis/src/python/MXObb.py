import queue
from pathlib import Path
from dataclasses import dataclass, field
import cv2
import numpy as np
import memryx as mx

@dataclass
class NMSArgs:
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    max_det: int = 300
    nc: int = 15
    rotated: bool = True
    
@dataclass
class Detection:
    bbox: list[int]
    rot: float
    conf: float
    cls: str

@dataclass
class AnnotatedFrame:
    image: np.ndarray
    detections: list[Detection] = field(default_factory=list)

class MXObb:
    detector_imgsz = 1024
    classes = {
        0: 'plane', 1: 'ship', 2: 'storage tank', 3: 'baseball diamond', 
        4: 'tennis court', 5: 'basketball court', 6: 'ground track field', 
        7: 'harbor', 8: 'bridge', 9: 'large vehicle', 10: 'small vehicle', 
        11: 'helicopter', 12: 'roundabout', 13: 'soccer ball field', 
        14: 'swimming pool'
    }

    def __init__(self, models_dir, nms_args=None, target_classes=None):
        """
        Args:
            models_dir: Path to models directory
            nms_args: NMS arguments
            target_classes: List of class IDs to detect (e.g., [9, 10] for vehicles only)
        """
        self.nms_args = nms_args if nms_args else NMSArgs()
        self.target_classes = set(target_classes) if target_classes else None
        self._stopped = False
        self._outstanding_frames = 0

        self.input_q  = queue.Queue(maxsize=2)
        self.stage0_q = queue.Queue(maxsize=4)
        self.output_q = queue.Queue(maxsize=2)

        self.accl = mx.AsyncAccl(str(Path(models_dir) / "yolov8s-obb.dfp"))
        self.accl.set_postprocessing_model(str(Path(models_dir) / "yolov8s-obb_post.onnx"))
        self.accl.connect_input(self._detector_source)
        self.accl.connect_output(self._detector_sink)
        
        self._prepare_letterbox_params()

    def _prepare_letterbox_params(self):
        """Pre-compute constants for letterbox to avoid repeated calculations"""
        self.letterbox_shape = (self.detector_imgsz, self.detector_imgsz)
        self.letterbox_color = (114, 114, 114)

    def _letterbox(self, im):
        """Optimized letterbox with pre-computed parameters"""
        shape = im.shape[:2]
        
        # Compute scale
        r = min(self.letterbox_shape[0] / shape[0], self.letterbox_shape[1] / shape[1])
        
        # Compute new unpadded dimensions
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        
        # Compute padding
        dw = (self.letterbox_shape[1] - new_unpad[0]) / 2
        dh = (self.letterbox_shape[0] - new_unpad[1]) / 2
        
        # Resize only if needed
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # Add padding
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.letterbox_color)
        
        return im

    def _scale_boxes(self, img1_shape, boxes, img0_shape):
        """Vectorized box scaling"""
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)

        boxes[:, 0] -= pad[0]
        boxes[:, 1] -= pad[1]
        boxes[:, :4] /= gain
        return boxes

    # --- ASYNC INFRASTRUCTURE ---
    def put(self, image, block=True, timeout=None):
        annotated_frame = AnnotatedFrame(np.array(image))
        self.input_q.put(annotated_frame, block, timeout)
        self._outstanding_frames += 1

    def get(self, block=True, timeout=None):
        self._outstanding_frames -= 1
        annotated_frame = self.output_q.get(block, timeout)
        return annotated_frame


    def _detector_source(self):
        annotated_frame = self.input_q.get()
        if annotated_frame is None:
            self.output_q.put(None)  # Signal main loop to exit
            return None
        self.stage0_q.put(annotated_frame)

        ifmap = self._preprocess(annotated_frame.image)
        return ifmap

    def _detector_sink(self, *outputs):
        annotated_frame = self.stage0_q.get()
        image = annotated_frame.image
        
        detections = self._postprocess(outputs[0], image.shape)
        
        # Convert to Detection objects (only for target classes if specified)
        for det in detections:
            # det: [x, y, w, h, rot, conf, cls]
            annotated_frame.detections.append(
                Detection(
                    bbox = det[:4].tolist(),
                    rot = det[4],
                    conf = det[5],
                    cls = self.classes[int(det[6])]
                )
            )

        self.output_q.put(annotated_frame)

    def _preprocess(self, image):
        """Optimized preprocessing with reduced operations"""
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Letterbox resize
        resized = self._letterbox(image)
        
        # Normalize and transpose in one go
        # Shape: (H, W, C) -> (1, C, H, W)
        normalized = resized.astype(np.float32) / 255.0
        ifmap = np.transpose(normalized, (2, 0, 1))
        ifmap = np.expand_dims(ifmap, 0)
        
        return ifmap.astype(np.float32)
    
    def _postprocess(self, outputs, original_shape):
        """Highly optimized postprocessing with class filtering"""
        preds = outputs.transpose((0, 2, 1))[0]

        # ---------------------------------------------------------
        # 1. CLASS FILTERING (if target_classes specified)
        # ---------------------------------------------------------
        if self.target_classes is not None:
            # Extract class scores for target classes only
            # Assuming class scores are in columns 4:19
            class_scores = preds[:, 4:19]
            class_ids = np.argmax(class_scores, axis=1)
            
            # Filter to only target classes
            target_mask = np.isin(class_ids, list(self.target_classes))
            if not np.any(target_mask):
                return np.empty((0, 7))
            
            preds = preds[target_mask]
            class_ids = class_ids[target_mask]
            scores = np.max(class_scores[target_mask], axis=1)
        else:
            scores = np.max(preds[:, 4:19], axis=1)
            class_ids = np.argmax(preds[:, 4:19], axis=1)

        # ---------------------------------------------------------
        # 2. CONFIDENCE FILTERING
        # ---------------------------------------------------------
        conf_mask = scores > self.nms_args.conf_thres
        
        if not np.any(conf_mask):
            return np.empty((0, 7))

        # Apply confidence filter
        preds = preds[conf_mask]
        scores = scores[conf_mask]
        class_ids = class_ids[conf_mask]

        # ---------------------------------------------------------
        # 3. PREPARE FOR NMS
        # ---------------------------------------------------------
        # Pre-convert to degrees once
        angles_deg = np.degrees(preds[:, -1])
        
        # Create rotated boxes format for OpenCV
        # Using array operations to minimize Python loops
        boxes_xy = preds[:, :2]
        boxes_wh = preds[:, 2:4]
        
        # Build r_boxes_cv using numpy operations where possible
        r_boxes_cv = [
            ((float(x), float(y)), (float(w), float(h)), float(a))
            for (x, y), (w, h), a in zip(boxes_xy, boxes_wh, angles_deg)
        ]

        # ---------------------------------------------------------
        # 4. RUN NMS
        # ---------------------------------------------------------
        nms_indices = cv2.dnn.NMSBoxesRotated(
            r_boxes_cv, 
            scores.tolist(),  # Convert to list once
            self.nms_args.conf_thres, 
            self.nms_args.iou_thres
        )
        
        if len(nms_indices) == 0:
            return np.empty((0, 7))
            
        # ---------------------------------------------------------
        # 5. GATHER FINAL RESULTS
        # ---------------------------------------------------------
        nms_indices = nms_indices.flatten()
        final_preds = preds[nms_indices]
        final_scores = scores[nms_indices]
        final_cls = class_ids[nms_indices]
        
        # Scale boxes back to original image size
        network_shape = (self.detector_imgsz, self.detector_imgsz)
        final_preds[:, :4] = self._scale_boxes(network_shape, final_preds[:, :4], original_shape)
        
        # Return: [x, y, w, h, rot, conf, cls]
        return np.column_stack((
            final_preds[:, :4], 
            final_preds[:, -1], 
            final_scores, 
            final_cls
        ))

    def stop(self):
        while self._outstanding_frames > 0:
            try:
                self.get(timeout=0.1)
            except queue.Empty:
                continue
        self.input_q.put(None)
        self._stopped = True