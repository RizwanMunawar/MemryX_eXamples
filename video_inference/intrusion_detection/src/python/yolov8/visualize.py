#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2
import numpy as np

__all__ = ["vis"]


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None, is_learning_mode=False):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255


    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    
    info_text = 'Frame: %d | FPS: %.1f | Objects: %d' % (frame_id, fps, len(tlwhs))
    font = cv2.FONT_HERSHEY_DUPLEX
    text_scale_info = 0.6
    text_thickness_info = 1
    text_size = cv2.getTextSize(info_text, font, text_scale_info, text_thickness_info)[0]

    if is_learning_mode:
        learning_text = 'LEARNING MODE (Grace Period)'
        learning_text_size = cv2.getTextSize(learning_text, font, text_scale_info, text_thickness_info)[0]
        learning_y_pos = 30
        learning_x_pos = im_w - learning_text_size[0] - 25
        
        learning_bg_coords = [(learning_x_pos - 10, learning_y_pos - learning_text_size[1] - 10),
                              (learning_x_pos + learning_text_size[0] + 10, learning_y_pos + 10)]
        
        overlay_learning = im.copy()
        cv2.rectangle(overlay_learning, learning_bg_coords[0], learning_bg_coords[1], (0, 165, 255), -1)
        cv2.rectangle(overlay_learning, learning_bg_coords[0], learning_bg_coords[1], (255, 255, 255), 2)
        cv2.addWeighted(overlay_learning, 0.8, im, 0.2, 0, im)
        
        cv2.putText(im, learning_text, (learning_x_pos, learning_y_pos), font, text_scale_info, (255, 255, 255), text_thickness_info, cv2.LINE_AA)
    
    padding = 12
    box_coords = [(5, im_h - text_size[1] - 2*padding - 5), 
                  (text_size[0] + 2*padding + 5, im_h - 5)]
    
    overlay = im.copy()
    cv2.rectangle(overlay, box_coords[0], box_coords[1], (30, 30, 30), -1)
    cv2.rectangle(overlay, box_coords[0], box_coords[1], (60, 180, 255), 2)
    cv2.addWeighted(overlay, 0.7, im, 0.3, 0, im)
    
    text_pos = (padding + 5, im_h - padding - 5)
    cv2.putText(im, info_text, text_pos, font, text_scale_info, (255, 255, 255), text_thickness_info, cv2.LINE_AA)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = 'ID: {}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        
        corner_length = 15
        cv2.line(im, (intbox[0], intbox[1]), (intbox[0] + corner_length, intbox[1]), color, line_thickness + 1)
        cv2.line(im, (intbox[0], intbox[1]), (intbox[0], intbox[1] + corner_length), color, line_thickness + 1)
        
        id_font = cv2.FONT_HERSHEY_DUPLEX
        id_scale = 0.7
        id_thickness = 2
        id_text_size = cv2.getTextSize(id_text, id_font, id_scale, id_thickness)[0]
        
        label_padding = 6
        label_y_offset = -8
        label_bg_coords = [(intbox[0], intbox[1] + label_y_offset - id_text_size[1] - label_padding),
                          (intbox[0] + id_text_size[0] + 2*label_padding, intbox[1] + label_y_offset + label_padding)]
        
        cv2.rectangle(im, label_bg_coords[0], label_bg_coords[1], color, -1)
        cv2.rectangle(im, label_bg_coords[0], label_bg_coords[1], (255, 255, 255), 1)
        
        cv2.putText(im, id_text, (intbox[0] + label_padding, intbox[1] + label_y_offset - label_padding//2), 
                   id_font, id_scale, (255, 255, 255), id_thickness, cv2.LINE_AA)
    return im


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
