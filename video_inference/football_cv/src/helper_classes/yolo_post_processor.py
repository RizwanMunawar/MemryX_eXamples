from .utilities import sigmoid, xywh2xyxy_vectorized, nms
import numpy as np


# ================================
# YOLOv8 Full Postprocessor
# ================================
class YOLOV8Postprocessor:
    def __init__(self, num_classes=4, conf_thres=0.65, iou_thres=0.45):
        self.num_classes = num_classes
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def __call__(self, output):
        """
        Process raw model output and return dict of detections.
        Optimized to use vectorized numpy operations.
        """

        # Transpose to shape (N, 4 + num_classes) -> e.g., (8400, 84)
        out = output[0].T

        # Validate tensor shape
        expected_cols = 4 + self.num_classes
        if out.shape[1] != expected_cols:
            raise ValueError(f"Invalid model output shape {out.shape}, expected (*, {expected_cols})")

        # Split coordinates and classes
        # box_deltas: (N, 4) | class_logits: (N, num_classes)
        box_deltas = out[:, :4]
        class_logits = out[:, 4:]

        # =========================================================
        # 1. Fast Filtering (Pre-Sigmoid)
        # =========================================================
        # find the max confidence score for each anchor.
        # filter using raw logits to avoid computing exp/sigmoid 
        # for thousands of background boxes.
        # conf > thres  <=>  logit > ln(thres / (1-thres))
        
        max_logits = np.max(class_logits, axis=1)
        class_ids = np.argmax(class_logits, axis=1)

        # Calculate logit threshold
        eps = 1e-9
        thr = np.clip(self.conf_thres, eps, 1 - eps)
        logit_thres = np.log(thr / (1 - thr))

        # Create mask for valid detections
        mask = max_logits > logit_thres

        # Apply mask immediately -> reduces data size from ~8400 to ~20
        box_deltas = box_deltas[mask]
        class_ids = class_ids[mask]
        max_logits = max_logits[mask]

        if len(box_deltas) == 0:
            return {"boxes": np.zeros((0, 4)), "conf": np.array([]), "cls": np.array([])}

        # =========================================================
        # 2. Process Survivors
        # =========================================================
        # Compute Sigmoid only for the few remaining boxes
        scores = sigmoid(max_logits)

        # Vectorized Box Decoding (xywh -> xyxy)
        boxes = xywh2xyxy_vectorized(box_deltas)

        # =========================================================
        # 3. Batched NMS
        # =========================================================
        return self.apply_nms(boxes, scores, class_ids)

    

    def apply_nms(self, boxes, scores, labels):
        """Run NMS on all classes at once using the offset trick."""
        if len(boxes) == 0:
            return {"boxes": np.zeros((0, 4)), "conf": np.array([]), "cls": np.array([])}

        # Strategy: Offset boxes by class_id * max_image_size.
        # This moves boxes of different classes far apart so they never overlap,
        # allowing us to run a single NMS pass for everything.
        max_wh = 4096 
        offsets = labels.astype(float) * max_wh
        boxes_offset = boxes + offsets[:, None]

        # Call the standalone nms function
        keep = nms(boxes_offset, scores, self.iou_thres)

        return {
            "boxes": boxes[keep],
            "conf": scores[keep],
            "cls": labels[keep],
        }