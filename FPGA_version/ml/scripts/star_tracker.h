#ifndef STAR_TRACKER_H
#define STAR_TRACKER_H

// Set to 1 for software float simulation, 0 for HLS fixed-point mode.
// In FLOAT mode you can test quickly on a host compiler.
// In FIXED-POINT mode you exercise the quantized path intended for synthesis.
#define STAR_TRACKER_USE_FLOAT 1

// ---------------------------------------------------------------------------
// Classification labels — index must match the training label order.
// The six bits of direction_gpio are one-hot encoded with these positions:
//   bit 0 = UP, bit 1 = DOWN, bit 2 = LEFT,
//   bit 3 = RIGHT, bit 4 = FORWARD, bit 5 = BACKWARD
// ---------------------------------------------------------------------------
typedef enum {
    ST_DIR_UP       = 0,
    ST_DIR_DOWN     = 1,
    ST_DIR_LEFT     = 2,
    ST_DIR_RIGHT    = 3,
    ST_DIR_FORWARD  = 4,
    ST_DIR_BACKWARD = 5
} ST_Direction;

// One-hot mask for a given direction (usable in both C++ and HLS contexts).
#define ST_DIR_MASK(d) ((unsigned char)(1u << (d)))

#include "../models/star_tracker_weights.h"

#define ST_INPUT_PIXELS (ST_INPUT_WIDTH * ST_INPUT_HEIGHT)

// Feature map dimensions for stride-2, pad-1, kernel-3 convolutions (pad-2, kernel-5 for conv1).
// These compile-time formulas stay synchronized with the exported header.
#define ST_CONV1_OUT_W ((ST_INPUT_WIDTH  + 2 * ST_CONV1_PAD - ST_CONV1_K) / ST_CONV1_STRIDE + 1)
#define ST_CONV1_OUT_H ((ST_INPUT_HEIGHT + 2 * ST_CONV1_PAD - ST_CONV1_K) / ST_CONV1_STRIDE + 1)
#define ST_CONV2_OUT_W ((ST_CONV1_OUT_W  + 2 * ST_CONV2_PAD - ST_CONV2_K) / ST_CONV2_STRIDE + 1)
#define ST_CONV2_OUT_H ((ST_CONV1_OUT_H  + 2 * ST_CONV2_PAD - ST_CONV2_K) / ST_CONV2_STRIDE + 1)
#define ST_CONV3_OUT_W ((ST_CONV2_OUT_W  + 2 * ST_CONV3_PAD - ST_CONV3_K) / ST_CONV3_STRIDE + 1)
#define ST_CONV3_OUT_H ((ST_CONV2_OUT_H  + 2 * ST_CONV3_PAD - ST_CONV3_K) / ST_CONV3_STRIDE + 1)

// Spatial-pool bin sizes (must be integer: 20/5=4, 15/3=5).
#define ST_POOL_BIN_W (ST_CONV3_OUT_W / ST_POOL_W)
#define ST_POOL_BIN_H (ST_CONV3_OUT_H / ST_POOL_H)

// Number of bits used to represent each input pixel value (0-255).
#define PIXEL_WIDTH 8

#if STAR_TRACKER_USE_FLOAT
// FLOAT mode: simple numeric types for software debugging.
typedef float pixel_t;
typedef float accum_t;
#else
// FIXED mode: use compact integer types expected by HLS.
typedef ap_uint<PIXEL_WIDTH> pixel_t;
typedef ap_int<32> accum_t;
#endif

// Top-level CNN hardware function for HLS.
//
// Outputs:
//   predicted_class  – integer index 0-5 (ST_Direction enum)
//   direction_gpio   – one-hot 6-bit register driven as a plain output port
//                      (ap_none interface → holds value between invocations)
//                      bit 0 = UP, 1 = DOWN, 2 = LEFT,
//                      3 = RIGHT, 4 = FORWARD, 5 = BACKWARD
//
// In the Vivado block design, connect direction_gpio[5:0] directly to six
// FPGA output pins (or to an output port on the block design boundary).
// No AXI GPIO IP is needed — the HLS IP drives the pins itself.
void star_tracker_cnn(
    const pixel_t    input_image[ST_INPUT_PIXELS],
    int             *predicted_class,
    unsigned char   *direction_gpio
);

// Backward-compatible alias for older test harnesses (gpio ignored).
inline void star_tracker_mlp(
    const pixel_t input_image[ST_INPUT_PIXELS],
    int *predicted_class
) {
    unsigned char dummy_gpio = 0;
    star_tracker_cnn(input_image, predicted_class, &dummy_gpio);
}

#endif
