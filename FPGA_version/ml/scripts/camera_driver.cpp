// V4L2 USB camera driver for the Logitech C270 on PYNQ Z2.
// Uses memory-mapped buffers (no copies through kernel) and YUYV pixel format,
// which the C270 supports natively without requiring JPEG decoding.

#include "camera_driver.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <sys/time.h>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static int xioctl(int fd, unsigned long request, void* arg)
{
    int r;
    do { r = ioctl(fd, request, arg); } while (r == -1 && errno == EINTR);
    return r;
}

// Discard one frame from the queue (used during warmup).
static bool discard_frame(CameraDriver& cam)
{
    v4l2_buffer buf{};
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (xioctl(cam.fd, VIDIOC_DQBUF, &buf) == -1) {
        perror("cam: VIDIOC_DQBUF (warmup)");
        return false;
    }
    if (xioctl(cam.fd, VIDIOC_QBUF, &buf) == -1) {
        perror("cam: VIDIOC_QBUF (warmup re-queue)");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool cam_open(CameraDriver& cam, const char* device)
{
    cam.fd        = -1;
    cam.n_buffers = 0;
    cam.streaming = false;
    memset(cam.buffers, 0, sizeof(cam.buffers));

    // --- Open device ---
    cam.fd = open(device, O_RDWR | O_NONBLOCK, 0);
    if (cam.fd == -1) {
        fprintf(stderr, "cam: cannot open '%s': %s\n", device, strerror(errno));
        return false;
    }

    // --- Verify it is a capture device ---
    v4l2_capability cap{};
    if (xioctl(cam.fd, VIDIOC_QUERYCAP, &cap) == -1) {
        perror("cam: VIDIOC_QUERYCAP");
        goto fail;
    }
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        fprintf(stderr, "cam: '%s' is not a video capture device\n", device);
        goto fail;
    }
    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        fprintf(stderr, "cam: '%s' does not support streaming\n", device);
        goto fail;
    }

    // --- Set format: 640x480 YUYV ---
    {
        v4l2_format fmt{};
        fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width       = CAM_CAPTURE_WIDTH;
        fmt.fmt.pix.height      = CAM_CAPTURE_HEIGHT;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        fmt.fmt.pix.field       = V4L2_FIELD_NONE;

        if (xioctl(cam.fd, VIDIOC_S_FMT, &fmt) == -1) {
            perror("cam: VIDIOC_S_FMT");
            goto fail;
        }
        if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) {
            fprintf(stderr, "cam: driver did not accept YUYV format\n");
            goto fail;
        }
        if (fmt.fmt.pix.width != (unsigned)CAM_CAPTURE_WIDTH ||
            fmt.fmt.pix.height != (unsigned)CAM_CAPTURE_HEIGHT) {
            fprintf(stderr, "cam: driver set %ux%u instead of %dx%d\n",
                    fmt.fmt.pix.width, fmt.fmt.pix.height,
                    CAM_CAPTURE_WIDTH, CAM_CAPTURE_HEIGHT);
            goto fail;
        }
    }

    // --- Request mmap buffers ---
    {
        v4l2_requestbuffers req{};
        req.count  = CAM_NUM_BUFFERS;
        req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;

        if (xioctl(cam.fd, VIDIOC_REQBUFS, &req) == -1) {
            perror("cam: VIDIOC_REQBUFS");
            goto fail;
        }
        if (req.count < 2) {
            fprintf(stderr, "cam: insufficient buffer memory\n");
            goto fail;
        }
        cam.n_buffers = (int)req.count;
    }

    // --- mmap each buffer ---
    for (int i = 0; i < cam.n_buffers; ++i) {
        v4l2_buffer buf{};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;

        if (xioctl(cam.fd, VIDIOC_QUERYBUF, &buf) == -1) {
            perror("cam: VIDIOC_QUERYBUF");
            goto fail_unmap;
        }

        cam.buffers[i].length = buf.length;
        cam.buffers[i].start  = mmap(nullptr, buf.length,
                                     PROT_READ | PROT_WRITE,
                                     MAP_SHARED, cam.fd, buf.m.offset);
        if (cam.buffers[i].start == MAP_FAILED) {
            perror("cam: mmap");
            cam.buffers[i].start = nullptr;
            goto fail_unmap;
        }
    }

    // --- Queue all buffers ---
    for (int i = 0; i < cam.n_buffers; ++i) {
        v4l2_buffer buf{};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;

        if (xioctl(cam.fd, VIDIOC_QBUF, &buf) == -1) {
            perror("cam: VIDIOC_QBUF (init)");
            goto fail_unmap;
        }
    }

    // --- Start streaming ---
    {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(cam.fd, VIDIOC_STREAMON, &type) == -1) {
            perror("cam: VIDIOC_STREAMON");
            goto fail_unmap;
        }
        cam.streaming = true;
    }

    // --- Warmup: discard frames so auto-exposure settles ---
    for (int i = 0; i < CAM_WARMUP_FRAMES; ++i) {
        // Block until a frame is ready (timeout 2 s)
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(cam.fd, &fds);
        timeval tv{ .tv_sec = 2, .tv_usec = 0 };
        int r = select(cam.fd + 1, &fds, nullptr, nullptr, &tv);
        if (r <= 0) {
            fprintf(stderr, "cam: timeout waiting for warmup frame %d\n", i);
            // Non-fatal: proceed with what we have
            break;
        }
        discard_frame(cam);
    }

    return true;

fail_unmap:
    for (int i = 0; i < cam.n_buffers; ++i) {
        if (cam.buffers[i].start && cam.buffers[i].start != MAP_FAILED)
            munmap(cam.buffers[i].start, cam.buffers[i].length);
        cam.buffers[i].start = nullptr;
    }
fail:
    close(cam.fd);
    cam.fd = -1;
    return false;
}

void cam_close(CameraDriver& cam)
{
    if (cam.fd == -1) return;

    if (cam.streaming) {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(cam.fd, VIDIOC_STREAMOFF, &type);
        cam.streaming = false;
    }

    for (int i = 0; i < cam.n_buffers; ++i) {
        if (cam.buffers[i].start && cam.buffers[i].start != MAP_FAILED)
            munmap(cam.buffers[i].start, cam.buffers[i].length);
        cam.buffers[i].start = nullptr;
    }

    close(cam.fd);
    cam.fd = -1;
}

bool cam_capture_raw(CameraDriver& cam, uint8_t* raw_yuyv)
{
    // Block until a frame is available (timeout 2 s)
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(cam.fd, &fds);
    timeval tv{ .tv_sec = 2, .tv_usec = 0 };

    int r = select(cam.fd + 1, &fds, nullptr, nullptr, &tv);
    if (r == -1) { perror("cam: select"); return false; }
    if (r ==  0) { fprintf(stderr, "cam: frame timeout\n"); return false; }

    v4l2_buffer buf{};
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (xioctl(cam.fd, VIDIOC_DQBUF, &buf) == -1) {
        perror("cam: VIDIOC_DQBUF");
        return false;
    }

    // Copy raw YUYV bytes out of the mmap region
    const size_t expected = (size_t)CAM_CAPTURE_WIDTH * CAM_CAPTURE_HEIGHT * 2;
    memcpy(raw_yuyv, cam.buffers[buf.index].start,
           buf.bytesused < expected ? buf.bytesused : expected);

    if (xioctl(cam.fd, VIDIOC_QBUF, &buf) == -1) {
        perror("cam: VIDIOC_QBUF (re-queue)");
        return false;
    }
    return true;
}

bool cam_capture_gray(CameraDriver& cam, uint8_t* out)
{
    // Allocate a temporary YUYV buffer on the heap to avoid a large stack frame.
    const int yuyv_bytes = CAM_CAPTURE_WIDTH * CAM_CAPTURE_HEIGHT * 2;
    uint8_t* yuyv = static_cast<uint8_t*>(malloc(yuyv_bytes));
    if (!yuyv) {
        fprintf(stderr, "cam: out of memory for YUYV buffer\n");
        memset(out, 0, CAM_OUTPUT_PIXELS);
        return false;
    }

    if (!cam_capture_raw(cam, yuyv)) {
        free(yuyv);
        memset(out, 0, CAM_OUTPUT_PIXELS);
        return false;
    }

    // Downsample 640x480 YUYV → 160x120 grayscale using 4x4 block averaging.
    //
    // YUYV layout (per two pixels): [Y0][U][Y1][V]
    // Luma byte Y is at every even byte offset: yuyv[2*col], yuyv[2*col+2], ...
    //
    // For each 4x4 source block → one 8-bit output pixel.
    // Scale factors: horizontal 640/160 = 4, vertical 480/120 = 4.

    constexpr int SCALE_X = CAM_CAPTURE_WIDTH  / CAM_OUTPUT_WIDTH;   // 4
    constexpr int SCALE_Y = CAM_CAPTURE_HEIGHT / CAM_OUTPUT_HEIGHT;  // 4
    constexpr int BLOCK   = SCALE_X * SCALE_Y;                       // 16

    for (int oy = 0; oy < CAM_OUTPUT_HEIGHT; ++oy) {
        for (int ox = 0; ox < CAM_OUTPUT_WIDTH; ++ox) {
            uint32_t sum = 0;
            for (int dy = 0; dy < SCALE_Y; ++dy) {
                int src_row = oy * SCALE_Y + dy;
                // Row base in YUYV: each row has CAM_CAPTURE_WIDTH*2 bytes.
                // Y byte for pixel at column c is at offset src_row*stride + c*2.
                const uint8_t* row = yuyv + src_row * CAM_CAPTURE_WIDTH * 2;
                for (int dx = 0; dx < SCALE_X; ++dx) {
                    int src_col = ox * SCALE_X + dx;
                    sum += row[src_col * 2];   // Y byte
                }
            }
            out[oy * CAM_OUTPUT_WIDTH + ox] = (uint8_t)(sum / BLOCK);
        }
    }

    free(yuyv);
    return true;
}
