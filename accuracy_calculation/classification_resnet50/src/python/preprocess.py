import cv2
import numpy as np
########################################
# Helper functions for Pre-processing
#######################################
# from MlCommons GitHub MLCommons/inference/vision/classification_and_detection/python/dataset.py
def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img

# from MlCommons GitHub MLCommons/inference/vision/classification_and_detection/python/dataset.py
def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img

# from MlCommons GitHub MLCommons/inference/vision/classification_and_detection/python/dataset.py
def pre_process_vgg(img, dims=None, need_transpose=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output_height, output_width, _ = dims
    cv2_interpol = cv2.INTER_AREA
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2_interpol)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')

    # normalize image
    means = np.array([123.68, 116.78, 103.94], dtype=np.float32)
    img -= means

    # transpose if needed
    if need_transpose:
        img = img.transpose([2, 0, 1])
    return img
