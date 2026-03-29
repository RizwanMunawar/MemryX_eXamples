"""
yolov26.py
YoloV26-640 pre/post processing
---------
Copyright (c) 2024 MemryX Inc.

GPL v3 License

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, see http://www.gnu.org/licenses/gpl-3.0

"""

###################################################################################################

# Imports

import numpy as np
import cv2

###################################################################################################

COCO_CLASSES = ( "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
        "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",)

###################################################################################################
###################################################################################################
###################################################################################################

class YoloV26:
    """
    A helper class to run YOLOv26 pre- and post-proccessing.
    """

###################################################################################################
    def __init__(self, stream_img_size=None):
        """
        The initialization function.
        """

        self.name = 'YoloV26-640'
        self.input_size = (640, 640, 3) 

        self.stream_mode = False
        if stream_img_size:
            # Pre-calculate ratio/pad values for preprocessing
            self.preprocess(np.zeros(stream_img_size))
            self.stream_mode = True

###################################################################################################
    def preprocess(self, img):
        """
        YOLOv26 Pre-proccessing.
        """
        h0, w0 = img.shape[:2] # orig hw

        r = self.input_size[0] / max(h0, w0)  # resize img to img_size
        if r != 1:  
            interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]

        img, ratio, dwdh = self._letterbox(img, new_shape=self.input_size, auto=False)
        img = img.astype(np.float32)
        img /= 255.0 # Scale

        shapes = (h0, w0), ((h / h0, w / w0), dwdh)

        if not self.stream_mode:
            self.ratio = r
            self.pad = dwdh

        # Update input shape to what the original ONNX model expects ie B,C,H,W
        img = np.transpose(img, (2,0,1))
        img = np.expand_dims(img, axis=0)

        return img
    
###################################################################################################
    def _letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=False, stride=32):
        """
        A letterbox function.
        """
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        return im, r, (dw, dh)

###################################################################################################
    def postprocess(self, fmap):
        """
        YOLOv26 Post-proccessing.
        Output format: (1, 300, 6) where each detection is [x1, y1, x2, y2, confidence, class_id]
        """
        
        post_output = fmap[0]  # Shape: (1, 300, 6)
        
        # Check if output is empty
        if len(post_output) == 0:
            return []
        
        # Get the detections array - shape (300, 6)
        detections = post_output[0]
        
        dets = []
        
        # Iterate through each detection
        for detection in detections:
            # Extract values: [x1, y1, x2, y2, confidence, class_id]
            x1, y1, x2, y2, confidence, class_id = detection
            
            # Filter by confidence threshold
            if confidence < 0.4:
                continue
            
            # Skip invalid detections (x1 < 0 often indicates padding/invalid detection)
            if x1 < 0:
                continue
            
            # Adjust for padding
            unpad = np.array([x1, y1, x2, y2]) - np.array([self.pad[0], self.pad[1], self.pad[0], self.pad[1]])
            x1_adj, y1_adj, x2_adj, y2_adj = (unpad / self.ratio).astype(int)
            
            # Create detection dictionary
            det = {
                'bbox': (int(x1_adj), int(y1_adj), int(x2_adj), int(y2_adj)),
                'class': COCO_CLASSES[int(class_id)],
                'class_idx': int(class_id),
                'score': float(confidence)
            }
            dets.append(det)
        
        return dets

###################################################################################################
if __name__=="__main__":
    pass

# eof
