// Smoke test for the C++ camera driver on PYNQ Z2.
// Usage:  ./camera_driver_test [/dev/video0]
// Captures one frame and prints basic statistics to stdout.
// Saves the raw 160x120 luma bytes to gray_out.bin for inspection.

#include "camera_driver.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

int main(int argc, char* argv[])
{
    const char* device = (argc > 1) ? argv[1] : "/dev/video0";

    CameraDriver cam;
    if (!cam_open(cam, device)) {
        fprintf(stderr, "test: failed to open camera at %s\n", device);
        return 1;
    }

    uint8_t gray[CAM_OUTPUT_PIXELS];

    if (!cam_capture_gray(cam, gray)) {
        fprintf(stderr, "test: frame capture failed\n");
        cam_close(cam);
        return 1;
    }

    // Basic statistics
    uint32_t sum = 0;
    uint8_t  mn  = 255, mx = 0;
    for (int i = 0; i < CAM_OUTPUT_PIXELS; ++i) {
        sum += gray[i];
        mn = std::min(mn, gray[i]);
        mx = std::max(mx, gray[i]);
    }
    double mean = (double)sum / CAM_OUTPUT_PIXELS;

    printf("Captured %dx%d grayscale frame\n", CAM_OUTPUT_WIDTH, CAM_OUTPUT_HEIGHT);
    printf("  min=%-3u  max=%-3u  mean=%.1f\n", mn, mx, mean);

    // Sanity check: an all-black frame from a live camera almost certainly
    // means something is wrong (cap is on, wrong device, etc.).
    if (mx < 3)
        fprintf(stderr, "test: WARNING – frame is nearly all black; "
                        "check that the lens cap is removed\n");

    // Dump raw bytes so you can inspect with numpy / hexdump
    FILE* f = fopen("gray_out.bin", "wb");
    if (f) {
        fwrite(gray, 1, CAM_OUTPUT_PIXELS, f);
        fclose(f);
        printf("  Raw luma bytes written to gray_out.bin "
               "(load with numpy.fromfile('gray_out.bin', dtype=uint8)"
               ".reshape(%d,%d))\n",
               CAM_OUTPUT_HEIGHT, CAM_OUTPUT_WIDTH);
    }

    cam_close(cam);
    return 0;
}
