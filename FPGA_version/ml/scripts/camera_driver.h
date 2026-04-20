#pragma once

#include <cstdint>
#include <cstddef>
#include <linux/videodev2.h>

// Capture resolution from Logitech C270
static constexpr int CAM_CAPTURE_WIDTH  = 640;
static constexpr int CAM_CAPTURE_HEIGHT = 480;

// Output resolution expected by the star tracker CNN
static constexpr int CAM_OUTPUT_WIDTH   = 160;
static constexpr int CAM_OUTPUT_HEIGHT  = 120;
static constexpr int CAM_OUTPUT_PIXELS  = CAM_OUTPUT_WIDTH * CAM_OUTPUT_HEIGHT;

// Number of V4L2 mmap buffers to allocate
static constexpr int CAM_NUM_BUFFERS = 4;

// Number of frames to discard on startup to let the C270 auto-exposure settle
static constexpr int CAM_WARMUP_FRAMES = 15;

struct MappedBuffer {
    void*  start;
    size_t length;
};

struct CameraDriver {
    int          fd;
    MappedBuffer buffers[CAM_NUM_BUFFERS];
    int          n_buffers;
    bool         streaming;
};

// Open and initialise the camera at `device` (default /dev/video0).
// Sets 640x480 YUYV, allocates mmap buffers, queues them, and starts streaming.
// Discards CAM_WARMUP_FRAMES so auto-exposure settles before first capture.
// Returns true on success; prints error details to stderr on failure.
bool cam_open(CameraDriver& cam, const char* device = "/dev/video0");

// Stop streaming, unmap buffers, and close the file descriptor.
void cam_close(CameraDriver& cam);

// Capture one raw 640x480 YUYV frame into `raw_yuyv`.
// `raw_yuyv` must point to at least CAM_CAPTURE_WIDTH * CAM_CAPTURE_HEIGHT * 2 bytes.
// Returns true on success.
bool cam_capture_raw(CameraDriver& cam, uint8_t* raw_yuyv);

// Capture one frame and produce a 160x120 grayscale image in `out`.
// Internally extracts luma from YUYV and performs 4x4 block-average downsampling.
// `out` must point to at least CAM_OUTPUT_PIXELS bytes.
// Returns true on success; returns false and zeroes `out` on failure.
bool cam_capture_gray(CameraDriver& cam, uint8_t* out);
