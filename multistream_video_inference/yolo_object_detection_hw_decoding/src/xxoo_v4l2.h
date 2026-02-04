#pragma once

#include <vector>

typedef void *xxoo_cam_t;

typedef enum
{
    MXACE_IMG_FMT_MJPG = 1,
    MXACE_IMG_FMT_RGB24, // RGB888
    MXACE_IMG_FMT_GREY,
    MXACE_IMG_FMT_YUYV, // 16 bit per pixel
    MXACE_IMG_FMT_OTHERS,
} xxoo_cam_pixel_format_e;

typedef struct
{
    int width;
    int height;
    xxoo_cam_pixel_format_e pixfmt;
    int fps;
} xxoo_cam_setting_t;

// open a camera, for Linux, it use V4L2, it reads /dev/video# for cam_id
// not support Windows yet
xxoo_cam_t xxoo_cam_open(int cam_id, xxoo_cam_setting_t *cc_setting);

// get camera settings
int xxoo_cam_get_setting(xxoo_cam_t cc, xxoo_cam_setting_t *cc_setting);

// read frame buffer pointer
void *xxoo_cam_get_frame(xxoo_cam_t cc, int *bufSize);

// return frame buffer pointer
int xxoo_cam_put_frame(xxoo_cam_t cc, void *frame_buf);

// close the camera
int xxoo_cam_close(xxoo_cam_t cc);
