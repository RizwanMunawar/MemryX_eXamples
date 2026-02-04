import numpy as np
import cv2 
from typing import List, Tuple


class Rtmpose:
    def __init__(self):
        self.model_input_size = (192, 256)
        self.mean = (123.675, 116.28, 103.53)
        self.std = (58.395, 57.12, 57.375)

    def bbox_xyxy2cs(self, bbox: np.ndarray, padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
        dim = bbox.ndim
        if dim == 1:
            bbox = bbox[None, :]

        x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
        center = np.hstack([x1 + x2, y1 + y2]) * 0.5
        scale = np.hstack([x2 - x1, y2 - y1]) * padding

        if dim == 1:
            center = center[0]
            scale = scale[0]

        return center, scale

    def _rotate_point(self, pt: np.ndarray, angle_rad: float) -> np.ndarray:
        sn, cs = np.sin(angle_rad), np.cos(angle_rad)
        rot_mat = np.array([[cs, -sn], [sn, cs]])
        return rot_mat @ pt

    def _get_3rd_point(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        direction = a - b
        c = b + np.r_[-direction[1], direction[0]]
        return c

    def get_warp_matrix(self, center: np.ndarray, scale: np.ndarray, rot: float,
                        output_size: Tuple[int, int], shift: Tuple[float, float] = (0., 0.),
                        inv: bool = False) -> np.ndarray:
        shift = np.array(shift)
        src_w = scale[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.deg2rad(rot)
        src_dir = self._rotate_point(np.array([0., src_w * -0.5]), rot_rad)
        dst_dir = np.array([0., dst_w * -0.5])

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale * shift
        src[1, :] = center + src_dir + scale * shift
        src[2, :] = self._get_3rd_point(src[0, :], src[1, :])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        dst[2, :] = self._get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return warp_mat

    def top_down_affine(self, input_size: dict, bbox_scale: dict, bbox_center: dict,
                        img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        w, h = input_size
        warp_size = (int(w), int(h))

        aspect_ratio = w / h
        b_w, b_h = np.hsplit(bbox_scale, [1])
        bbox_scale = np.where(b_w > b_h * aspect_ratio,
                            np.hstack([b_w, b_w / aspect_ratio]),
                            np.hstack([b_h * aspect_ratio, b_h]))

        center = bbox_center
        scale = bbox_scale
        rot = 0
        warp_mat = self.get_warp_matrix(center, scale, rot, output_size=(w, h))

        img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

        return img, bbox_scale

    def get_simcc_maximum(self, simcc_x: np.ndarray,
                        simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N, K, Wx = simcc_x.shape
        simcc_x = simcc_x.reshape(N * K, -1)
        simcc_y = simcc_y.reshape(N * K, -1)

        x_locs = np.argmax(simcc_x, axis=1)
        y_locs = np.argmax(simcc_y, axis=1)
        locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
        max_val_x = np.amax(simcc_x, axis=1)
        max_val_y = np.amax(simcc_y, axis=1)

        vals = 0.5 * (max_val_x + max_val_y)
        locs[vals <= 0.] = -1

        locs = locs.reshape(N, K, 2)
        vals = vals.reshape(N, K)

        return locs, vals

    def preprocess(self, img: np.ndarray, bbox: list):
        bbox = np.array(bbox)
        center, scale = self.bbox_xyxy2cs(bbox, padding=1.25)
        resized_img, scale = self.top_down_affine(self.model_input_size, scale, center, img)

        mean = np.array(self.mean)
        std = np.array(self.std)
        resized_img = (resized_img - mean) / std
        return resized_img, center, scale

    def postprocess(self, outputs: List[np.ndarray], center: Tuple[int, int],
                    scale: Tuple[int, int], simcc_split_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        simcc_x, simcc_y = outputs
        locs, scores = self.get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs / simcc_split_ratio

        keypoints = keypoints / self.model_input_size * scale
        keypoints = keypoints + center - scale / 2

        return keypoints, scores
