#include "MXVideoDecoder.h"

AVPixelFormat HW_PIX_FMT;

MXVideoDecoder::MXVideoDecoder(const DecoderConfig &config){
    this->disp_width = config.disp_width;
    this->disp_height = config.disp_height;
    this->decoder_source = config.source_type;
    this->decoder_name = config.decoder_name;
    this->access_str = config.access_str;
    this->disp_frame_size = this->disp_width * this->disp_height * 3;
    this->force_sw = config.force_sw;
    if(!config.force_sw) {
        this->device_type = av_hwdevice_find_type_by_name(config.device_type.c_str());
        std::cout << "Using device type: " << config.device_type << std::endl;
    }
    if (config.source_type == DecoderConfig::USB) {
        this->usb_width = config.usb_width;
        this->usb_height = config.usb_height;
        this->usb_fps = config.usb_fps;
    }
    if (config.device_type == "vaapi") {
        this->hw_pix_fmt = av_get_pix_fmt(config.device_type.c_str());
        HW_PIX_FMT = hw_pix_fmt;
    }
    else {
        this->hw_pix_fmt = av_get_pix_fmt("drm_prime");
        HW_PIX_FMT = hw_pix_fmt;
    }
    av_log_set_level(AV_LOG_QUIET);
}

MXVideoDecoder::~MXVideoDecoder(){
    clean_up_decoder();
}


// Custom get_format callback that selects our desired hardware pixel format.
enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
    const enum AVPixelFormat *p;
    for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
        if (*p == HW_PIX_FMT)
            return *p;
    }
    std::cerr << "Failed to get HW surface format.\n";
    return AV_PIX_FMT_NONE;
}

void MXVideoDecoder::scale_and_convert() {
    if (sws_ctx_display == NULL) {
        // Initialize software scaler
        sws_ctx_display = sws_getContext(
            sw_frame->width, sw_frame->height, (AVPixelFormat)sw_frame->format, // Source: width, height, format
            disp_width, disp_height, AV_PIX_FMT_RGB24,                          // Target: width, height, format
            SWS_FAST_BILINEAR, NULL, NULL, NULL);

        if (!sws_ctx_display) {
            fprintf(stderr, "Failed to initialize software scaler\n");
            return;
        }
    }

    // Scale and convert colorspace
    if (sws_scale(
            sws_ctx_display,
            sw_frame->data, sw_frame->linesize, 0, sw_frame->height,           // Source
            rgb_disp_frame->data, rgb_disp_frame->linesize                     // Target
        ) < 0) {
        fprintf(stderr, "Failed to perform software scaling\n");
    }
    return;
}

void MXVideoDecoder::initiate_decoder() {
    switch (decoder_source) {
        case DecoderConfig::USB:
            initiate_usb_decoder();
            return;
        case DecoderConfig::IP:
            initiate_ip_decoder();
            return;
        case DecoderConfig::FILE:
            initiate_file_decoder();
            return;
        default:
            return;
    }
}

void MXVideoDecoder::initiate_ip_decoder() {
    AVDictionary      *opts    = NULL;
    av_dict_set(&opts, "&codec_options", "2", 0);


    if (avformat_open_input(&format_ctx, access_str.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Could not open input stream\n";
        return;
    }

    if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
        std::cerr << "Failed to retrieve input stream information\n";
        return;
    }

    // Find the video stream.
    for (unsigned i = 0; i < format_ctx->nb_streams; i++) {
        if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            break;
        }
    }
    if (video_stream_index == -1) {
        std::cerr << "No video stream found\n";
        return;
    }

    auto decoder = avcodec_find_decoder_by_name(decoder_name.c_str());
    if (!decoder) {
        std::cerr << "Decoder not found\n";
        return;
    }

    // Allocate the codec context.
    codec_ctx = avcodec_alloc_context3(decoder);
    if (!codec_ctx) {
        std::cerr << "Failed to allocate the codec context\n";
        return;
    }

    if (avcodec_parameters_to_context(codec_ctx, format_ctx->streams[video_stream_index]->codecpar) < 0) {
        std::cerr << "Failed to copy codec parameters to decoder context\n";
        return;
    }

    codec_ctx->thread_count = 1;
    codec_ctx->thread_type = FF_THREAD_SLICE;

    if (!force_sw) {
        // Create the hardware device context.
        AVBufferRef* hw_device_ctx = nullptr;
        if (av_hwdevice_ctx_create(&hw_device_ctx, device_type, nullptr, nullptr, 0) < 0) {
            std::cerr << "Failed to create HW device context\n";
            return;
        }
        codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

        if (codec_ctx->hw_device_ctx){
            printf("HW decoder: create context OK\n");
        }
        
        // Set our custom get_format callback.
        codec_ctx->get_format = get_hw_format;
    }

    // Open the codec.
    if (avcodec_open2(codec_ctx, decoder, NULL) < 0) {
        std::cerr << "Failed to open codec\n";
        return;
    }
    
    pkt = av_packet_alloc();
    gpu_frame = av_frame_alloc();
    filt_frame = av_frame_alloc();
    sw_frame = av_frame_alloc();
    rgb_disp_frame = av_frame_alloc();

    rgb_disp_frame->format = AV_PIX_FMT_RGB24;
    rgb_disp_frame->width = disp_width;
    rgb_disp_frame->height = disp_height;

    if (av_frame_get_buffer(rgb_disp_frame, 0) < 0){
        fprintf(stderr, "Failed to allocate RGB frame buffer\n");
        return;
    }

}


void MXVideoDecoder::initiate_usb_decoder() {
    xxoo_cam_setting_t cam_settings = {usb_width, usb_height, MXACE_IMG_FMT_MJPG, usb_fps};
    cam_id = atoi(access_str.c_str());

    cam_handle = xxoo_cam_open(cam_id , &cam_settings);

    pkt = av_packet_alloc();
    gpu_frame = av_frame_alloc();
    sw_frame = av_frame_alloc();
    rgb_disp_frame = av_frame_alloc();

    if (!pkt || !gpu_frame || !sw_frame || !rgb_disp_frame){
        fprintf(stderr, "Failed to allocate frames or packet\n");
    }

    auto decoder = avcodec_find_decoder_by_name(decoder_name.c_str());
    if (!decoder){
        fprintf(stderr, "Failed to find codec\n");
        return;
    }

    codec_ctx = avcodec_alloc_context3(decoder);
    if (!codec_ctx){
        fprintf(stderr, "Failed to allocate the codec context\n");
        return;
    }

    if (!force_sw) {
        // Initialize hardware decoder
        codec_ctx->hw_device_ctx = NULL;
        if (av_hwdevice_ctx_create(&codec_ctx->hw_device_ctx, device_type, NULL, NULL, 0) < 0){
            fprintf(stderr, "Failed to create specified HW device.\n");
            return;
        }

        if (codec_ctx->hw_device_ctx){
            printf("HW decoder: create context OK\n");
        }
    }
    if ((avcodec_open2(codec_ctx, decoder, NULL)) < 0){
        fprintf(stderr, "Failed to open codec\n");
        return;
    }

    rgb_disp_frame->format = AV_PIX_FMT_RGB24;
    rgb_disp_frame->width = disp_width;
    rgb_disp_frame->height = disp_height;

    if (av_frame_get_buffer(rgb_disp_frame, 0) < 0){
        fprintf(stderr, "Failed to allocate RGB frame buffer\n");
        return;
    }
    
    if (!codec_ctx){
        fprintf(stderr, "Hardware decoder initialization failed\n");
        return;
    }
}

void MXVideoDecoder::initiate_file_decoder() {
    if (avformat_open_input(&format_ctx, access_str.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Could not open input stream\n";
        return;
    }

    if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
        std::cerr << "Failed to retrieve input stream information\n";
        return;
    }

    // Find the video stream.
    for (unsigned i = 0; i < format_ctx->nb_streams; i++) {
        if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            break;
        }
    }
    if (video_stream_index == -1) {
        std::cerr << "No video stream found\n";
        return;
    }

    auto decoder = avcodec_find_decoder_by_name(decoder_name.c_str());
    if (!decoder) {
        std::cerr << "Decoder not found\n";
        return;
    }

    // Allocate the codec context.
    codec_ctx = avcodec_alloc_context3(decoder);
    if (!codec_ctx) {
        std::cerr << "Failed to allocate the codec context\n";
        return;
    }

    if (avcodec_parameters_to_context(codec_ctx, format_ctx->streams[video_stream_index]->codecpar) < 0) {
        std::cerr << "Failed to copy codec parameters to decoder context\n";
        return;
    }

    if(!force_sw) {
        // Create the hardware device context.
        AVBufferRef* hw_device_ctx = nullptr;
        if (av_hwdevice_ctx_create(&hw_device_ctx, device_type, nullptr, nullptr, 0) < 0) {
            std::cerr << "Failed to create HW device context\n";
            return;
        }
        codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

        if (codec_ctx->hw_device_ctx){
            printf("HW decoder: create context OK\n");
        }
        
        // Set our custom get_format callback.
        codec_ctx->get_format = get_hw_format;
    }
    
    // Open the codec.
    if (avcodec_open2(codec_ctx, decoder, nullptr) < 0) {
        std::cerr << "Failed to open codec\n";
        return;
    }
    
    pkt = av_packet_alloc();
    gpu_frame = av_frame_alloc();
    sw_frame = av_frame_alloc();
    rgb_disp_frame = av_frame_alloc();

    rgb_disp_frame->format = AV_PIX_FMT_RGB24;
    rgb_disp_frame->width = disp_width;
    rgb_disp_frame->height = disp_height;

    if (av_frame_get_buffer(rgb_disp_frame, 0) < 0){
        fprintf(stderr, "Failed to allocate RGB frame buffer\n");
        return;
    }
}

AVFrame* MXVideoDecoder::get_frame() {

    switch (decoder_source) {
        case DecoderConfig::USB:
            if (force_sw) {
                return get_usb_frame_software();
            }
            else {
                return get_usb_frame();
            }
        case DecoderConfig::IP:
            if (force_sw) {
                return get_ip_frame_software();
            }
            else {
                return get_ip_frame();
            }
        case DecoderConfig::FILE:
            if (force_sw) {
                return get_file_frame_software();
            }
            else {
                return get_file_frame();
            }
        default:
            return nullptr;
    }
}

AVFrame* MXVideoDecoder::get_ip_frame() {    
    // Main decoding loop.
    if(av_read_frame(format_ctx, pkt) != 0) {
	    is_running = false;
	    return nullptr;
    }
 
    // Send packet for decoding.
    if (pkt->stream_index == video_stream_index) {
        if (avcodec_send_packet(codec_ctx, pkt) < 0) {
            std::cerr << "Error sending packet for decoding\n";
        }
    }
 
    // If a gpu frame is not received, try again with new packet
    while (avcodec_receive_frame(codec_ctx, gpu_frame) != 0) {
        av_read_frame(format_ctx, pkt);
        if (pkt->stream_index == video_stream_index) {
            if (avcodec_send_packet(codec_ctx, pkt) < 0) {
                std::cerr << "Error sending packet for decoding\n";
            }
        }
    }

    if (av_hwframe_transfer_data(sw_frame, gpu_frame, 0) < 0) {
        std::cerr << "Error transferring frame to system memory\n";
    }
    else {
        scale_and_convert();
    }

    av_frame_unref(gpu_frame);
    av_packet_unref(pkt);
    return rgb_disp_frame;
}

AVFrame* MXVideoDecoder::get_ip_frame_software() {
    // Main decoding loop.
    if (av_read_frame(format_ctx, pkt) != 0) {
	    is_running = false;
	    return nullptr;
    }

    // Send packet for decoding.
    if (pkt->stream_index == video_stream_index) {
        if (avcodec_send_packet(codec_ctx, pkt) < 0) {
            std::cerr << "Error sending packet for decoding\n";
        }
    }

    // Try receiving a decoded frame
    while (avcodec_receive_frame(codec_ctx, sw_frame) != 0) {
        av_read_frame(format_ctx, pkt);
        if (pkt->stream_index == video_stream_index) {
            if (avcodec_send_packet(codec_ctx, pkt) < 0) {
                std::cerr << "Error sending packet for decoding\n";
	            continue;
            }
        }
    }

    // At this point, sw_frame contains the decoded frame in system memory

    // Perform colorspace conversion and scaling
    scale_and_convert(); 
    av_frame_unref(sw_frame);
    av_packet_unref(pkt);
    return rgb_disp_frame;
}


AVFrame* MXVideoDecoder::get_usb_frame() {
    int jpgSize;
    if(!cam_handle){
            fprintf(stderr, "Camera handle destroyed - Check camera handle opening\n");        
    }
    uint8_t *jpgBuf = (uint8_t *)xxoo_cam_get_frame(cam_handle, &jpgSize);
    while(jpgBuf ==  NULL){
        jpgBuf = (uint8_t *)xxoo_cam_get_frame(cam_handle, &jpgSize);
    }

    pkt->data = jpgBuf;
    pkt->size = jpgSize;
    
    int ret;

    if(!codec_ctx ||  !pkt)
    {
        std::cout<<"Error with codex and pkt \n";
        return nullptr;
    }
    if ((ret = avcodec_send_packet(codec_ctx, pkt)) < 0){
        fprintf(stderr, "Error sending packet for decoding\n");
        return nullptr;
    }
    
    while ((ret = avcodec_receive_frame(codec_ctx, gpu_frame)) >= 0){

        xxoo_cam_put_frame(cam_handle, jpgBuf);

        if ((ret = av_hwframe_transfer_data(sw_frame, gpu_frame, 0)) < 0){
            fprintf(stderr, "Error transferring frame\n");
            break;
        }

        // Perform colorspace conversion and scaling
        scale_and_convert();       
    }
    return rgb_disp_frame;
}

AVFrame* MXVideoDecoder::get_usb_frame_software() {
    int jpgSize;
    if(!cam_handle){
            fprintf(stderr, "Camera handle destroyed - Check camera handle opening\n");
            return nullptr;
    }
    uint8_t *jpgBuf = (uint8_t *)xxoo_cam_get_frame(cam_handle, &jpgSize);
    while(jpgBuf ==  NULL){
        jpgBuf = (uint8_t *)xxoo_cam_get_frame(cam_handle, &jpgSize);
    }

    pkt->data = jpgBuf;
    pkt->size = jpgSize;
    
    int ret;

    if(!codec_ctx || !pkt)
    {
        std::cout<<"Error with codex and pkt \n";
        return nullptr;
    }
    if ((ret = avcodec_send_packet(codec_ctx, pkt)) < 0){
        fprintf(stderr, "Error sending packet for decoding\n");
        return nullptr;
    }
    
    while ((ret = avcodec_receive_frame(codec_ctx, sw_frame)) >= 0){

        xxoo_cam_put_frame(cam_handle, jpgBuf);

        // Perform colorspace conversion and scaling
        scale_and_convert();        
    }
    return rgb_disp_frame;
}

AVFrame* MXVideoDecoder::get_file_frame() {
    // Main decoding loop
    if (av_read_frame(format_ctx, pkt) != 0) {
        is_running = false;
        return nullptr;
    }
 
    // Send packet for decoding.
    if (pkt->stream_index == video_stream_index) {
        if (avcodec_send_packet(codec_ctx, pkt) < 0) {
            std::cerr << "Error sending packet for decoding\n";
        }
    }

    // If a gpu frame is not received, try again with new packet
    while (avcodec_receive_frame(codec_ctx, gpu_frame) != 0) {
        av_read_frame(format_ctx, pkt);
        if (pkt->stream_index == video_stream_index) {
            if (avcodec_send_packet(codec_ctx, pkt) < 0) {
                std::cerr << "Error sending packet for decoding\n";
            }
        }
    }
    
    if (av_hwframe_transfer_data(sw_frame, gpu_frame, 0) < 0) {
        std::cerr << "Error transferring frame to system memory\n";
    }
    else {
        scale_and_convert();
    }

    av_frame_unref(gpu_frame);
    av_packet_unref(pkt);
    return rgb_disp_frame;
}

AVFrame* MXVideoDecoder::get_file_frame_software() {
    // Main decoding loop.
    if (av_read_frame(format_ctx, pkt) != 0) {
        is_running = false;
        return nullptr;
    }

    // Send packet for decoding.
    if (pkt->stream_index == video_stream_index) {
        if (avcodec_send_packet(codec_ctx, pkt) < 0) {
            std::cerr << "Error sending packet for decoding\n";
        }
    }

    // Try receiving a decoded frame
    while (avcodec_receive_frame(codec_ctx, sw_frame) != 0) {
        av_read_frame(format_ctx, pkt);
        if (pkt->stream_index == video_stream_index) {
            if (avcodec_send_packet(codec_ctx, pkt) < 0) {
                std::cerr << "Error sending packet for decoding\n";
            }
        }
    }

    // Perform colorspace conversion and scaling
    scale_and_convert();
    av_frame_unref(sw_frame);
    av_packet_unref(pkt);
    return rgb_disp_frame;
}



void MXVideoDecoder::clean_up_decoder() {

    if(decoder_source == DecoderConfig::USB) {
        if (codec_ctx) {
            avcodec_free_context(&codec_ctx);
        }
        if (format_ctx) {
            avformat_close_input(&format_ctx);
        }
        if (sws_ctx_display) {
            sws_freeContext(sws_ctx_display);
        }
        if (rgb_disp_frame) {
            av_frame_free(&rgb_disp_frame);
        }
        if (pkt) {
            av_packet_free(&pkt);
        }
    }
    if (decoder_source == DecoderConfig::IP || decoder_source == DecoderConfig::FILE) {
        
        if (filt_frame) {
            av_frame_free(&filt_frame);
        }
        if (sw_frame) {
            av_frame_free(&sw_frame);
        }
        if (gpu_frame) {
            av_frame_free(&gpu_frame);
        }
        if (codec_ctx) {
            avcodec_free_context(&codec_ctx);
        }
        if (format_ctx) {
            avformat_close_input(&format_ctx);
        }
        if (sws_ctx_display) {
            sws_freeContext(sws_ctx_display);
        }
        if (rgb_disp_frame) {
            av_frame_free(&rgb_disp_frame);
        }
        if (pkt) {
            av_packet_free(&pkt);
        }
    }
}
