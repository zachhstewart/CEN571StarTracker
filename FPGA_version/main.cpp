// Star tracker FPGA pipeline — top-level entry point.
// Runs continuously on the PYNQ Z2 PS (ARM) with the bitstream loaded.
//
// Usage:  sudo ./startracker [/dev/video0]
//
// On each iteration:
//   1. Captures a 160x120 grayscale frame from the Logitech C270
//   2. Pushes it to the HLS CNN IP via AXI
//   3. Reads back the predicted attitude class
//   4. Prints the result (GPIO pins are driven automatically by the HLS IP)

#include <cstdio>
#include <csignal>
#include "camera_driver.h"
#include "run_inference.h"

static volatile bool g_running = true;
static void on_sigint(int) { g_running = false; }

int main(int argc, char* argv[])
{
    const char* cam_device = (argc > 1) ? argv[1] : "/dev/video0";

    signal(SIGINT, on_sigint);

    // --- Open camera ---
    CameraDriver cam;
    if (!cam_open(cam, cam_device)) {
        fprintf(stderr, "main: failed to open camera\n");
        return 1;
    }

    // --- Open HLS IP interface ---
    Inferencer inf;
    if (!infer_open(inf)) {
        fprintf(stderr, "main: failed to open HLS inferencer\n");
        cam_close(cam);
        return 1;
    }

    printf("Star tracker running. Press Ctrl+C to stop.\n");
    printf("%-12s  %s\n", "Class", "Direction");
    printf("%-12s  %s\n", "-----", "---------");

    int last_cls = -1;
    while (g_running) {
        int cls = infer_run(inf, cam);
        if (cls < 0) {
            fprintf(stderr, "main: inference error — retrying\n");
            continue;
        }

        // Only print when the classification changes (reduces noise)
        if (cls != last_cls) {
            printf("%-12d  %s\n", cls, infer_class_name(cls));
            last_cls = cls;
        }
    }

    printf("\nShutting down.\n");
    infer_close(inf);
    cam_close(cam);
    return 0;
}
