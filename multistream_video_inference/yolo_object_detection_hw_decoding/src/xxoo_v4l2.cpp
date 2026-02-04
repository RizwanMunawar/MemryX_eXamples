#include "xxoo_v4l2.h"

#include <map>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

#include <endian.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>

#include <fcntl.h> /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include <linux/videodev2.h>

#define NUM_MMAP_BUFFER 5

struct mmap_buf_t
{
    void *start[3];
    size_t length[3];
};

typedef struct
{
    int fd;
    char dev_name[30];
    enum v4l2_buf_type vdo_buf_type;
    bool is_mplane;
    struct v4l2_plane *vdo_planes;
    int vdo_num_planes;
    struct mmap_buf_t *buffers;
    unsigned int n_buffers;

    std::map<void *, struct v4l2_buffer> v4l2_buf_map;

    int width;
    int height;
    int pixelformat;
    int fps;
} MXACE_camera_capture_body_t;

#define CLEAR(x) memset(&(x), 0, sizeof(x))

static int xioctl(int fh, int request, void *arg)
{
    int r;

    do
    {
        r = ioctl(fh, request, arg);
    } while (-1 == r && EINTR == errno);

    return r;
}

static char *v4l2_fourcc2s(__u32 fourcc, char *buf)
{
    buf[0] = fourcc & 0x7f;
    buf[1] = (fourcc >> 8) & 0x7f;
    buf[2] = (fourcc >> 16) & 0x7f;
    buf[3] = (fourcc >> 24) & 0x7f;
    if (fourcc & (1 << 31))
    {
        buf[4] = '-';
        buf[5] = 'B';
        buf[6] = 'E';
        buf[7] = '\0';
    }
    else
    {
        buf[4] = '\0';
    }
    return buf;
}

xxoo_cam_t xxoo_cam_open(int cam_id, xxoo_cam_setting_t *cc_setting)
{


    MXACE_camera_capture_body_t *cc_bdy = (MXACE_camera_capture_body_t *)malloc(sizeof(MXACE_camera_capture_body_t));

    if (!cc_bdy) {
        printf("Memory allocation failed\n");
        return NULL;
    }

    // Manually call constructor for std::map
    new (&cc_bdy->v4l2_buf_map) std::map<void *, struct v4l2_buffer>();


    sprintf(cc_bdy->dev_name, "/dev/video%d", cam_id);

    struct stat st;

    if (-1 == stat(cc_bdy->dev_name, &st))
    {
        printf("Cannot identify '%s': %d, %s\n",
                cc_bdy->dev_name, errno, strerror(errno));
        fprintf(stderr, "Cannot identify '%s': %d, %s\n",
                cc_bdy->dev_name, errno, strerror(errno));
        cc_bdy->v4l2_buf_map.~map();
        free(cc_bdy);
        return NULL;
    }

    if (!S_ISCHR(st.st_mode))
    {
        fprintf(stderr, "%s is no devicen", cc_bdy->dev_name);
        cc_bdy->v4l2_buf_map.~map();
        free(cc_bdy);
        return NULL;
    }

    int fd = open(cc_bdy->dev_name, O_RDWR /* required */ | O_NONBLOCK, 0);

    if (-1 == fd)
    {
        fprintf(stderr, "Cannot open '%s': %d, %s\n",
                cc_bdy->dev_name, errno, strerror(errno));
        cc_bdy->v4l2_buf_map.~map();
        free(cc_bdy);
        return NULL;
    }

    cc_bdy->fd = fd;

    struct v4l2_capability cap;
    struct v4l2_cropcap cropcap;
    struct v4l2_crop crop;
    struct v4l2_format fmt;
    unsigned int min;

    if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &cap))
    {
        if (EINVAL == errno)
        {
            fprintf(stderr, "%s is no V4L2 device\n",
                    cc_bdy->dev_name);
            cc_bdy->v4l2_buf_map.~map();
        free(cc_bdy);
            return NULL;
        }
        else
        {
            printf("VIDIOC_QUERYCAP error\n");
            cc_bdy->v4l2_buf_map.~map();
        free(cc_bdy);
            return NULL;
        }
    }

    if (!(cap.capabilities & (V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_VIDEO_CAPTURE_MPLANE)))
    {
        fprintf(stderr, "%s is no video capture device\n",
                cc_bdy->dev_name);
        cc_bdy->v4l2_buf_map.~map();
        free(cc_bdy);
        return NULL;
    }

    if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)
    {
        cc_bdy->vdo_buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        cc_bdy->is_mplane = false;
    }
    else
    {
        cc_bdy->vdo_buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        cc_bdy->is_mplane = true;
    }

    if (!(cap.capabilities & V4L2_CAP_STREAMING))
    {
        fprintf(stderr, "%s does not support streaming i/o\n",
                cc_bdy->dev_name);
        cc_bdy->v4l2_buf_map.~map();
        free(cc_bdy);
        return NULL;
    }

    /* Select video input, video standard and tune here. */

    CLEAR(cropcap);

    cropcap.type = cc_bdy->vdo_buf_type;

    if (0 == xioctl(fd, VIDIOC_CROPCAP, &cropcap))
    {
        crop.type = cc_bdy->vdo_buf_type;
        crop.c = cropcap.defrect; /* reset to default */

        if (-1 == xioctl(fd, VIDIOC_S_CROP, &crop))
        {
            switch (errno)
            {
            case EINVAL:
                /* Cropping not supported. */
                break;
            default:
                /* Errors ignored. */
                break;
            }
        }
    }
    else
    {
        /* Errors ignored. */
    }

// set camera settings
#if 1
    if (cc_setting)
    {
        struct v4l2_format format;
        memset(&format, 0, sizeof(format));
        format.type = cc_bdy->vdo_buf_type;
        format.fmt.pix.width = cc_setting->width;
        format.fmt.pix.height = cc_setting->height;
        switch (cc_setting->pixfmt)
        {
        case MXACE_IMG_FMT_MJPG:
            format.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
            break;
        case MXACE_IMG_FMT_RGB24:
            format.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
            break;
        case MXACE_IMG_FMT_GREY:
            format.fmt.pix.pixelformat = V4L2_PIX_FMT_GREY;
            break;
        case MXACE_IMG_FMT_YUYV:
            format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
            break;
	default:
	    perror("Invalid cc_setting->pixfmt");
	    return NULL;
        }
        format.fmt.pix.field = V4L2_FIELD_INTERLACED;

        if (ioctl(fd, VIDIOC_S_FMT, &format) == -1)
        {
            perror("Setting pixel format");
            return NULL;
        }

        struct v4l2_streamparm streamparm;
        memset(&streamparm, 0, sizeof(streamparm));
        streamparm.type = cc_bdy->vdo_buf_type;
        streamparm.parm.capture.timeperframe.numerator = 1;
        streamparm.parm.capture.timeperframe.denominator = cc_setting->fps;

        if (ioctl(fd, VIDIOC_S_PARM, &streamparm) == -1)
        {
            perror("Setting frame rate");
            return NULL;
        }
    }
#endif

    CLEAR(fmt);

    fmt.type = cc_bdy->vdo_buf_type;
    if (-1 == xioctl(fd, VIDIOC_G_FMT, &fmt))
    {
        printf("VIDIOC_G_FMT error\n");
        cc_bdy->v4l2_buf_map.~map();
        free(cc_bdy);
        return NULL;
    }

    char fmt_str[5] = {0};

    if (cc_bdy->is_mplane)
    {
        cc_bdy->vdo_num_planes = fmt.fmt.pix_mp.num_planes;
        cc_bdy->vdo_planes = (struct v4l2_plane *)calloc(cc_bdy->vdo_num_planes, sizeof(struct v4l2_plane));
    }

    // printf("VIDIOC_G_FMT:\n");
    // printf("is multiple plane : %s\n",
    //    (cc_bdy->vdo_buf_type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) ? "Yes" : "No");
    // if (cc_bdy->is_mplane)
    //     printf("number of planes : %d\n", cc_bdy->vdo_num_planes);
    // printf("video format : %s\n", v4l2_fourcc2s(fmt.fmt.pix.pixelformat, fmt_str));
    // printf("video width : %d\n", fmt.fmt.pix.width);
    // printf("video height : %d\n", fmt.fmt.pix.height);

    cc_bdy->width = fmt.fmt.pix.width;
    cc_bdy->height = fmt.fmt.pix.height;
    cc_bdy->pixelformat = fmt.fmt.pix.pixelformat;

    struct v4l2_streamparm streamparm;
    memset(&streamparm, 0, sizeof(streamparm));
    streamparm.type = cc_bdy->vdo_buf_type;

    if (ioctl(fd, VIDIOC_G_PARM, &streamparm) == -1)
    {
        perror("Getting frame rate");
        return NULL;
    }

    cc_bdy->fps = streamparm.parm.capture.timeperframe.denominator;

    /* Buggy driver paranoia. */
    min = fmt.fmt.pix.width * 2;
    if (fmt.fmt.pix.bytesperline < min)
        fmt.fmt.pix.bytesperline = min;
    min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
    if (fmt.fmt.pix.sizeimage < min)
        fmt.fmt.pix.sizeimage = min;

    // init mmap

    struct v4l2_requestbuffers req;

    CLEAR(req);

    req.count = NUM_MMAP_BUFFER;
    req.type = cc_bdy->vdo_buf_type;
    req.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req))
    {
        if (EINVAL == errno)
        {
            fprintf(stderr, "%s does not support memory mapping", cc_bdy->dev_name);
            cc_bdy->v4l2_buf_map.~map();
        free(cc_bdy);
            return NULL;
        }
        else
        {
            printf("VIDIOC_REQBUFS error\n");
            cc_bdy->v4l2_buf_map.~map();
        free(cc_bdy);
            return NULL;
        }
    }

    if (req.count < 2)
    {
        fprintf(stderr, "Insufficient buffer memory on %s\n", cc_bdy->dev_name);
        cc_bdy->v4l2_buf_map.~map();
        free(cc_bdy);
        return NULL;
    }

    cc_bdy->buffers = (struct mmap_buf_t *)calloc(req.count, sizeof(struct mmap_buf_t));

    if (!cc_bdy->buffers)
    {
        fprintf(stderr, "Out of memory\n");
        cc_bdy->v4l2_buf_map.~map();
        free(cc_bdy);
        return NULL;
    }

    cc_bdy->v4l2_buf_map.clear();

    uint32_t nb = 0;
    for (nb = 0; nb < req.count; ++nb)
    {
        struct v4l2_buffer v4l2buf;

        CLEAR(v4l2buf);

        v4l2buf.type = cc_bdy->vdo_buf_type;
        v4l2buf.memory = V4L2_MEMORY_MMAP;
        v4l2buf.index = nb;

        if (cc_bdy->is_mplane)
        {
            v4l2buf.length = cc_bdy->vdo_num_planes;
            v4l2buf.m.planes = cc_bdy->vdo_planes;
            memset(cc_bdy->vdo_planes, 0, sizeof(struct v4l2_plane));
        }

        if (-1 == xioctl(fd, VIDIOC_QUERYBUF, &v4l2buf))
        {
            printf("VIDIOC_QUERYBUF error\n");
            cc_bdy->v4l2_buf_map.~map();
        free(cc_bdy);
            return NULL;
        }

        if (cc_bdy->is_mplane)
        {
            for (int i = 0; i < cc_bdy->vdo_num_planes; i++)
            {
                cc_bdy->buffers[nb].length[i] = v4l2buf.m.planes[i].length;
                cc_bdy->buffers[nb].start[i] = mmap(NULL, v4l2buf.m.planes[i].length,
                                                    PROT_READ | PROT_WRITE,
                                                    MAP_SHARED, fd,
                                                    v4l2buf.m.planes[i].m.mem_offset);
            }
        }
        else
        {
            cc_bdy->buffers[nb].length[0] = v4l2buf.length;
            cc_bdy->buffers[nb].start[0] = mmap(NULL /* start anywhere */,
                                                v4l2buf.length,
                                                PROT_READ | PROT_WRITE /* required */,
                                                MAP_SHARED /* recommended */,
                                                fd, v4l2buf.m.offset);
        }

        if (MAP_FAILED == cc_bdy->buffers[nb].start)
        {
            printf("mmap error\n");
            cc_bdy->v4l2_buf_map.~map();
        free(cc_bdy);
            return NULL;
        }

        if (-1 == xioctl(fd, VIDIOC_QBUF, &v4l2buf))
        {
            printf("VIDIOC_QBUF error\n");
            cc_bdy->v4l2_buf_map.~map();
        free(cc_bdy);
            return NULL;
        }

        void *frame_buf = cc_bdy->buffers[nb].start[0]; // FIXME for MP
        cc_bdy->v4l2_buf_map[frame_buf] = v4l2buf;
    }

    cc_bdy->n_buffers = nb;

    if (-1 == xioctl(fd, VIDIOC_STREAMON, &cc_bdy->vdo_buf_type))
    {
        printf("VIDIOC_STREAMON error\n");
        cc_bdy->v4l2_buf_map.~map();
        free(cc_bdy);
        return NULL;
    }

    return (xxoo_cam_t)cc_bdy;
}

// get camera settings
int xxoo_cam_get_setting(xxoo_cam_t cc, xxoo_cam_setting_t *cc_setting)
{
    MXACE_camera_capture_body_t *cc_bdy = (MXACE_camera_capture_body_t *)cc;

    cc_setting->width = cc_bdy->width;
    cc_setting->height = cc_bdy->height;
    cc_setting->fps = cc_bdy->fps;

    switch (cc_bdy->pixelformat)
    {
    case V4L2_PIX_FMT_MJPEG:
        cc_setting->pixfmt = MXACE_IMG_FMT_MJPG;
        break;
    case V4L2_PIX_FMT_RGB24:
        cc_setting->pixfmt = MXACE_IMG_FMT_RGB24;
        break;
    case V4L2_PIX_FMT_GREY:
        cc_setting->pixfmt = MXACE_IMG_FMT_GREY;
        break;
    case V4L2_PIX_FMT_YUYV:
        cc_setting->pixfmt = MXACE_IMG_FMT_YUYV;
        break;
    default:
        cc_setting->pixfmt = MXACE_IMG_FMT_OTHERS;
        break;
    }

    return 0;
}

// read frame buffer pointer
void *xxoo_cam_get_frame(xxoo_cam_t cc, int *bufSize)
{
    MXACE_camera_capture_body_t *cc_bdy = (MXACE_camera_capture_body_t *)cc;

    struct v4l2_buffer v4l2buf;

#if 0
    memset(&v4l2buf, 0, sizeof(v4l2buf));
    v4l2buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    v4l2buf.memory = V4L2_MEMORY_MMAP;

    if (ioctl(cc_bdy->fd, VIDIOC_DQBUF, &v4l2buf) == -1)
    {
        perror("Dequeue buffer");
        return NULL;
    }

    *bufSize = v4l2buf.bytesused;
    return cc_bdy->buffers[v4l2buf.index].start[0];
#endif

#if 1
    for (;;)
    {
        fd_set fds;
        struct timeval tv;
        int r;

        CLEAR(v4l2buf);
        v4l2buf.type = cc_bdy->vdo_buf_type;
        v4l2buf.memory = V4L2_MEMORY_MMAP;

        FD_ZERO(&fds);
        FD_SET(cc_bdy->fd, &fds);

        /* Timeout. */
        tv.tv_sec = 20;
        tv.tv_usec = 0;

        r = select(cc_bdy->fd + 1, &fds, NULL, NULL, &tv);

        if (-1 == r)
        {
            if (EINTR == errno)
                continue;
            printf("v4l2 select error\n");
            return NULL;
        }

        if (0 == r)
        {
            fprintf(stderr, "v4l2 select timeout - HW Decode\n");
            exit(EXIT_FAILURE);
        }

        if (cc_bdy->is_mplane)
        {
            v4l2buf.length = cc_bdy->vdo_num_planes;
            v4l2buf.m.planes = cc_bdy->vdo_planes;
            memset(cc_bdy->vdo_planes, 0, sizeof(struct v4l2_plane));
        }

        if (-1 == xioctl(cc_bdy->fd, VIDIOC_DQBUF, &v4l2buf))
        {
            switch (errno)
            {
            case EAGAIN:
                continue;

            case EIO:
                /* Could ignore EIO, see spec. */
                /* fall through */

            default:
                printf("VIDIOC_DQBUF error: %s (errno: %d)\n", strerror(errno), errno);
                return NULL;
            }
        }

        break;
    }

    *bufSize = v4l2buf.bytesused;
    return cc_bdy->buffers[v4l2buf.index].start[0];
#endif
}

// return frame buffer pointer
int xxoo_cam_put_frame(xxoo_cam_t cc, void *frame_buf)
{
    MXACE_camera_capture_body_t *cc_bdy = (MXACE_camera_capture_body_t *)cc;

    struct v4l2_buffer &v4l2buf = cc_bdy->v4l2_buf_map[frame_buf];

    if (-1 == xioctl(cc_bdy->fd, VIDIOC_QBUF, &v4l2buf))
    {
        printf("VIDIOC_QBUF error\n");
        return -1;
    }

    return 0;
}

// close the camera
int xxoo_cam_close(xxoo_cam_t cc)
{
    MXACE_camera_capture_body_t *cc_bdy = (MXACE_camera_capture_body_t *)cc;

    // FIXME: free memory

    int ret = close(cc_bdy->fd);
    if (ret != 0)
    {
        printf("close error: %s (errno: %d)\n", strerror(errno), errno);
    }

    return ret;
}

// 定義一個函數來轉換 pixel format 為字串
const char *get_pixel_format_string(xxoo_cam_pixel_format_e format)
{
    switch (format)
    {
    case MXACE_IMG_FMT_MJPG:
        return "MJPG";
    case MXACE_IMG_FMT_RGB24:
        return "RGB24";
    case MXACE_IMG_FMT_GREY:
        return "GREY";
    case MXACE_IMG_FMT_YUYV:
        return "YUYV";
    case MXACE_IMG_FMT_OTHERS:
        return "OTHERS";
    default:
        return "UNKNOWN";
    }
}


