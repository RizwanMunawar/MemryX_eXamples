import numpy as np
import cv2

###################################################################################################

COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

###################################################################################################

class YoloV8Seg:
    """
    A helper class to run YOLOv8 pre- and post-proccessing.
    """

    class NumpyPostProcess:
        """
        Super fast numpy implementation of YOLOv8 post-processing.
        """

        def __init__(self, skip_sigmoid: bool = False):
            self.skip_sigmoid = skip_sigmoid

            self.anchors = self._generate_anchors()
            self.scales = self._generate_scales()
            self._weights = np.arange(16, dtype=np.float32)
            self.confidence_thres = 0.4

        def _generate_anchors(self, sizes=[80, 40, 20]):
            yscales = []
            xscales = []
            for s in sizes:
                r = np.arange(s) + 0.5
                yscales.append(np.repeat(r, s))
                xscales.append(np.repeat(r[None, ...], s, axis=0).flatten())

            yscales = np.concatenate(yscales)
            xscales = np.concatenate(xscales)
            anchors = np.stack([xscales, yscales], axis=1)

            return anchors

        def _generate_scales(self, sizes=[80, 40, 20]):
            factors = [8, 16, 32]
            s = np.concatenate(
                [np.ones([int(s * s)]) * f for s, f in zip(sizes, factors)]
            )
            return s[:, None]

        def convert_to_xywh(self, boxes, valid_indices):
            
            # Distribution Focal Loss decoding
            boxes = self.dfl(boxes)

            # converts distances to actual [x_center, y_center, width, height]
            boxes = self.dist2bbox(
                boxes, self.anchors[valid_indices], self.scales[valid_indices]
            )
            
            return boxes
    
        @staticmethod
        def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
            x = x - np.max(x, axis=axis, keepdims=True)
            np.exp(x, out=x)
            x /= np.sum(x, axis=axis, keepdims=True)
            return x

        @staticmethod
        def _sigmoid(x: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-x))

        def dfl(self, x: np.ndarray) -> np.ndarray:
            x = x.reshape(-1, 4, 16)
            p = self._softmax(x, axis=2)
            p = p * self._weights[None, None, :]
            out = np.sum(p, axis=2, keepdims=False)
            return out

        def dist2bbox(
            self, x: np.ndarray, anchors: np.ndarray, scales: np.ndarray
        ) -> np.ndarray:
            lt = x[:, :2]
            rb = x[:, 2:]

            x1y1 = anchors - lt
            x2y2 = anchors + rb

            wh = x2y2 - x1y1
            c_xy = (x1y1 + x2y2) / 2

            out = np.concatenate([c_xy, wh], axis=1)
            out = out * scales

            return out

        def __call__(
            self,
            lbox,
            lcls,
            mask_proto,
            lmask_coef,
            mbox,
            mcls,
            mmask_coef,
            sbox,
            scls,
            smask_coef,
            onnx_format=False,
        ) -> np.ndarray:
            """
            lbox:    (80, 80, 1, 64)        // large box
            lcls:    (80, 80, 1, 80)        // large class
            mask_proto:  (160, 160, 1, 32)  // mask prototype
            lmask_coef: (80, 80, 1, 32)     // large mask coefficients
            mbox:    (40, 40, 1, 64)        // medium box
            mcls:    (40, 40, 1, 80)        // medium class
            mmask_coef: (40, 40, 1, 32)     // medium mask coefficients
            sbox:    (20, 20, 1, 64)        // small box
            scls:    (20, 20, 1, 80)        // small class
            smask_coef: (20, 20, 1, 32)     // small mask coefficients
            """


            if onnx_format:
                lbox = np.moveaxis(lbox, 1, -1)
                lcls = np.moveaxis(lcls, 1, -1)
                mbox = np.moveaxis(mbox, 1, -1)
                mcls = np.moveaxis(mcls, 1, -1)
                sbox = np.moveaxis(sbox, 1, -1)
                scls = np.moveaxis(scls, 1, -1) 

            boxes = np.concatenate([
                lbox.reshape(-1, 64), 
                mbox.reshape(-1, 64), 
                sbox.reshape(-1, 64)
            ], axis=0)
            
            classes = np.concatenate([
                lcls.reshape(-1, 80), 
                mcls.reshape(-1, 80), 
                scls.reshape(-1, 80)
            ], axis=0)

            mask_coef = np.concatenate(
                [
                    lmask_coef.reshape(-1, 32),
                    mmask_coef.reshape(-1, 32),
                    smask_coef.reshape(-1, 32),
                ], axis=0)

            # Process mask_proto (160, 160, 1, 32) -> (32, 160*160)
            mask_proto = mask_proto.squeeze(axis=2)  # (160, 160, 32)
            mask_proto = mask_proto.transpose(2, 0, 1)  # (32, 160, 160)
            mask_proto = mask_proto.reshape(32, -1)  # (32, 160*160)
            prob = self._sigmoid(mask_proto)
            mask_proto = prob * mask_proto  # element-wise multiplication
            
            return boxes, classes, mask_coef, mask_proto
        
    ###################################################################################################
    def __init__(self, cam, model_type="tflite"):
        """
        The initialization function.
        """

        self.cam = cam
        self.ori_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.ori_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.length = max((self.ori_height, self.ori_width))

        self.post = self.NumpyPostProcess(skip_sigmoid=True)
        self.input_size = (640, 640, 3)
        self.input_width = 640
        self.input_height = 640
        self.confidence_thres = 0.4
        self.iou_thres = 0.6
        self.model_type = model_type

        # Preallocate
        self.pre_img = np.zeros((self.length, self.length, 3), np.uint8)

    ###################################################################################################
    def preprocess(self, img):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """

        # Prepare a square image for inference
        if len(img.shape) > 3:
            img = np.squeeze(img, axis=2)
        self.pre_img[0 : self.ori_height, 0 : self.ori_width] = img

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(
            self.pre_img, scalefactor=1 / 255, size=(640, 640), swapRB=True
        )

        # Return the preprocessed image data
        return blob

    def scale_mask(self, masks, im0_shape, ratio_pad=None):

        ori_height, ori_width = im0_shape[0], im0_shape[1]
        masks = cv2.resize(
            masks, (self.length, self.length), interpolation=cv2.INTER_LINEAR
        )
        masks = masks[:ori_height, :ori_width]

        # Ensure mask has 3 dimensions
        if len(masks.shape) == 2:
            masks = np.expand_dims(masks, axis=-1)

        return masks

    def crop_mask(self, masks, boxes):
        """
        Crop the masks to the bounding boxes.

        Args:
            masks: (N, H, W) predicted masks
            boxes: (N, 4) bounding boxes in (left, top, width, height) format

        Returns:
            Cropped masks with the same shape as input masks.
        """
        N, H, W = masks.shape

        cropped = np.zeros((N, H, W), dtype=np.bool_)

        for i, (left, top, w, h) in enumerate(boxes):
            left, top, right, bottom = map(int, (left, top, left + w, top + h))
            left = max(left, 0)
            right = min(right, W-1)
            top = max(top, 0)
            bottom = min(bottom, H-1)
            cropped[i, top:bottom, left:right] = masks[i, top:bottom, left:right]

        return cropped

    @staticmethod
    def masks2segments(masks):
        # Convert masks to contour segments
        segments = []
        for x in masks.astype("uint8"):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # no segments found
                print("No segments found!!!!")
            segments.append(c.astype("float32"))
        return segments

    def process_mask(self, masks, boxes):
        # Process masks
        masks = masks.reshape(-1, 160, 160)  # NHW = (N, 160, 160)
        masks = masks.transpose(1, 2, 0)  # HWN = (160, 160, N)

        masks = self.scale_mask(
            masks, (self.ori_height, self.ori_width)
        )  # Scale the mask to original input image size

        masks = np.einsum("HWN -> NHW", masks)  # Reshape masks

        masks = masks > 0.5  # Binarize the masks
        segments = self.crop_mask(masks, boxes)

        return segments

    ###################################################################################################
    def postprocess(self, output):
        """
        Performs post-processing on the YOLOv8 model's output to extract bounding boxes, scores, and class IDs.

        Args:
            output (numpy.ndarray): The output of the model.

        Returns:
            list: A list of detections where each detection is a dictionary containing
                    'bbox', 'class_id', 'class', and 'score'.
        """

        # Transpose the output to shape (8400, 84), and mask_proto to (32, 160*160)
        if self.model_type == "numpy":
            outputs = self.post(*output)
            boxes, cls_scores, mask_coef, mask_proto = outputs
        else:    
            outputs = output[1]  # (1, 116, 8400)
            outputs = np.transpose(outputs[0])  # (8400, 116)
            boxes = outputs[:, :4]  # (8400, 4)
            cls_scores = outputs[:, 4:84]  # (8400, 80)
            mask_coef = outputs[:, 84:]  # (8400, 32)

            mask_proto = output[0]
            mask_proto = mask_proto[0] # remove batch dimension
            if self.model_type == "tflite":
                mask_proto = np.transpose(mask_proto, (2, 0, 1))  # (160, 160, 32) -> (32, 160, 160)

            mask_proto = mask_proto.reshape(32, -1)  # (32, 160*160)

        # Find the class with the highest score for each detection
        max_scores = np.max(cls_scores, axis=1) # (8400,) - max score for each prediction  
        class_ids = np.argmax(cls_scores, axis=1)  # (8400,) - index of the best class

        # Filter out detections with scores below the confidence threshold
        valid_indices = np.where(max_scores >= self.confidence_thres)[0]
        if len(valid_indices) == 0:
            return []  # Return an empty list if no valid detections

        
        # Select only valid detections
        valid_mask_coef = mask_coef[valid_indices]
        valid_class_ids = class_ids[valid_indices]
        valid_scores = max_scores[valid_indices]
        valid_boxes = boxes[valid_indices]

        ### NOTE: In order to speed up, do processing for valid detections only###        
        if self.model_type == "numpy":
            valid_boxes = self.post.convert_to_xywh(valid_boxes, valid_indices)
            if not self.post.skip_sigmoid:
                valid_scores = self.post._sigmoid(valid_scores)

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.length / self.input_width
        y_factor = self.length / self.input_height


        # Convert bounding box coordinates from (x_center, y_center, w, h) to (left, top, width, height)
        valid_boxes[:, 0] = (valid_boxes[:, 0] - valid_boxes[:, 2] / 2) * x_factor  # left
        valid_boxes[:, 1] = (valid_boxes[:, 1] - valid_boxes[:, 3] / 2) * y_factor  # top
        valid_boxes[:, 2] = valid_boxes[:, 2] * x_factor  # width
        valid_boxes[:, 3] = valid_boxes[:, 3] * y_factor  # height

        # Create detection dictionaries
        detections = [
            {
                "bbox": valid_boxes[i].astype(int).tolist(),
                "class_id": int(valid_class_ids[i]),
                "class": COCO_CLASSES[int(valid_class_ids[i])],
                "score": valid_scores[i],
            }
            for i in range(len(valid_boxes))
        ]

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        if len(detections) > 0:
            # NMS requires two lists: bounding boxes and confidence scores
            boxes_for_nms = [d["bbox"] for d in detections]
            scores_for_nms = [d["score"] for d in detections]

            indices = cv2.dnn.NMSBoxes(
                boxes_for_nms, scores_for_nms, self.confidence_thres, self.iou_thres
            )

            # Check if indices is not empty
            if len(indices) > 0:
                # Flatten indices if they are returned as a list of arrays
                if isinstance(indices[0], list) or isinstance(indices[0], np.ndarray):
                    indices = [i[0] for i in indices]

                mask_coef = valid_mask_coef[indices]
                boxes = valid_boxes[indices]

                # get segments
                masks = (
                    mask_coef @ mask_proto
                )  # (N, 32) @ (32, 160 * 160) -> (N, 160 * 160)
                segments = self.process_mask(masks, boxes)

                # Filter detections based on NMS
                final_detections = []
                for i, segment in zip(indices, segments):
                    detections[i]["segment"] = segment
                    final_detections.append(detections[i])

            else:
                final_detections = []
        else:
            final_detections = []

        # Return the list of final detections
        return final_detections
