
#ifndef MXVIDEODECODER
#define MXVIDEODECODER

#include <stdio.h>
#include <signal.h>
#include <thread>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <atomic>

#include "xxoo_v4l2.h"

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>  // Added for handling video files
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/frame.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavutil/avassert.h>
#include <libavutil/imgutils.h>
}

struct DecoderConfig {
    std::string device_type; // hw device type, e.g. vaapi/drm/rkmpp/ni_quadra
    enum SourceType {
        USB,
        IP,
        FILE
    } source_type;              // an instance variable indicating the kind of input

    std::string access_str;     // e.g., device path or RTSP URL
    std::string decoder_name;   // string identifier for the decoder: valid values are ["mjpeg", "h264", "h264_rkmpp", "hevc", "hevc_rkmpp"]
    uint32_t disp_width;        // final width of the decoded frame in pixels
    uint32_t disp_height;       // final height of the decoded frame in pixels
    uint32_t usb_width = 640;         // input width if source_type == USB
    uint32_t usb_height = 480;        // input height if source_type == USB
    int usb_fps = 30;                // framerate if source_type == USB
    bool force_sw = false;      // bool to force SW decode method (default: false if not specified)
    bool full_pipeline = false;
};


class MXVideoDecoder {

    public:

        MXVideoDecoder(const DecoderConfig &config);

        ~MXVideoDecoder();
        
        // Initialize decoder: general purpose
        void initiate_decoder();
        
        // Retrieve decoded frame into OpenCV Mat
        AVFrame* get_frame();
        
        // Clean up and release resources
        void clean_up_decoder();

        bool is_running = true;

    private:
        
        void initiate_usb_decoder();

        void initiate_ip_decoder();

        void initiate_file_decoder();

        AVFrame* get_usb_frame();

        AVFrame* get_usb_frame_software();

        AVFrame* get_ip_frame();

        AVFrame* get_ip_frame_software();

        AVFrame* get_file_frame();

        AVFrame* get_file_frame_software();

        void scale_and_convert();

        // Common components for both camera and file decoding
        AVCodecContext *codec_ctx = nullptr;
        //AVCodec *decoder = nullptr;
        SwsContext *sws_ctx_display = nullptr;
        SwsContext *sws_ctx_inf = nullptr;
        AVPacket *pkt;
        AVFrame *gpu_frame = nullptr;
        AVFrame *filt_frame = nullptr;
        AVFrame *sw_frame = nullptr;
        AVFrame *rgb_disp_frame = nullptr;

        uint32_t disp_width;
        uint32_t disp_height;
        uint32_t disp_frame_size;
        uint32_t usb_width = 640;
        uint32_t usb_height = 480;
        int usb_fps = 15;
        bool force_sw = false;
        bool full_pipeline = false;

        std::string access_str;
        std::string decoder_name;
        
        // Camera-specific members
        void* cam_handle = nullptr;
        int cam_id;
        // Video file-specific members
        AVFormatContext *format_ctx = nullptr;  // Format context for handling video files
        int video_stream_index = -1;        // Index of the video stream in the file
        DecoderConfig::SourceType decoder_source;
        AVBufferRef *hw_device_ctx = nullptr;
        AVPixelFormat hw_pix_fmt;

        enum AVHWDeviceType device_type;
        
};

#endif
