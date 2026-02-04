import numpy as np
import cv2 
from typing import List, Tuple


class Detect:
    def __init__(self, nms_thr=0.45, score_thr=0.7):
        self.nms_thr            = nms_thr
        self.score_thr          = score_thr
        self.model_input_size         = np.array([416,416])

    @property
    def name(self):
        return self.__class__.__name__

    def nms(self,boxes, scores, nms_thr):

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

    def multiclass_nms(self,boxes, scores, nms_thr, score_thr):

        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > score_thr
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                keep = self.nms(valid_boxes, valid_scores, nms_thr)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate(
                        [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None, None
        return np.concatenate(final_dets, 0), keep

    def preprocess(self,img: np.ndarray):

        if img.shape[:2] == tuple(self.model_input_size[:2]):
            padded_img = img.copy()
            ratio = 1.
        else:
            if len(img.shape) == 3:
                padded_img = np.ones(
                    (self.model_input_size[0], self.model_input_size[1], 3),
                    dtype=np.uint8) * 114
            else:
                padded_img = np.ones(self.model_input_size, dtype=np.uint8) * 114

            ratio = min(self.model_input_size[0] / img.shape[0],
                self.model_input_size[1] / img.shape[1])
            resized_img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
            ).astype(np.uint8)
            padded_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
            padded_img[:padded_shape[0], :padded_shape[1]] = resized_img

        return padded_img, ratio

    def postprocess(self,outputs: List[np.ndarray],ratio: float = 1.,) -> Tuple[np.ndarray, np.ndarray]:
        if outputs.shape[-1] == 4:
            grids = []
            expanded_strides = []
            strides = [8, 16, 32]

            hsizes = [self.model_input_size[0] // stride for stride in strides]
            wsizes = [self.model_input_size[1] // stride for stride in strides]

            for hsize, wsize, stride in zip(hsizes, wsizes, strides):
                xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
                grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
                grids.append(grid)
                shape = grid.shape[:2]
                expanded_strides.append(np.full((*shape, 1), stride))

            grids = np.concatenate(grids, 1)
            expanded_strides = np.concatenate(expanded_strides, 1)
            outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
            outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

            predictions = outputs[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
            boxes_xyxy /= ratio
            dets, keep = self.multiclass_nms(boxes_xyxy,
                                    scores,
                                    nms_thr=self.nms_thr,
                                    score_thr=self.score_thr)
            if dets is not None:
                pack_dets = (dets[:, :4], dets[:, 4], dets[:, 5])
                final_boxes, final_scores, final_cls_inds = pack_dets
                isscore = final_scores > 0.3
                iscat = final_cls_inds == 0
                isbbox = [i and j for (i, j) in zip(isscore, iscat)]
                final_boxes = final_boxes[isbbox]

        elif outputs.shape[-1] == 5:
            pack_dets = (outputs[0, :, :4], outputs[0, :, 4])
            final_boxes, final_scores = pack_dets
            final_boxes /= ratio
            isscore = final_scores > 0.3
            isbbox = [i for i in isscore]
            final_boxes = final_boxes[isbbox]

        return final_boxes
