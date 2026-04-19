#ifndef STAR_TRACKER_H
#define STAR_TRACKER_H

// Set to 1 for software float simulation, 0 for HLS fixed-point mode.
// In FLOAT mode you can test quickly on a host compiler.
// In FIXED-POINT mode you exercise the quantized path intended for synthesis.
#define STAR_TRACKER_USE_FLOAT 1

#include "../models/star_tracker_weights.h"

#define ST_INPUT_PIXELS (ST_INPUT_WIDTH * ST_INPUT_HEIGHT)

// Feature map dimensions for stride-2, pad-1, kernel-3 convolutions.
// These compile-time formulas keep tensor sizes synchronized with the exported header.
#define ST_CONV1_OUT_W ((ST_INPUT_WIDTH + 2 * ST_CONV1_PAD - ST_CONV1_K) / ST_CONV1_STRIDE + 1)
#define ST_CONV1_OUT_H ((ST_INPUT_HEIGHT + 2 * ST_CONV1_PAD - ST_CONV1_K) / ST_CONV1_STRIDE + 1)
#define ST_CONV2_OUT_W ((ST_CONV1_OUT_W + 2 * ST_CONV2_PAD - ST_CONV2_K) / ST_CONV2_STRIDE + 1)
#define ST_CONV2_OUT_H ((ST_CONV1_OUT_H + 2 * ST_CONV2_PAD - ST_CONV2_K) / ST_CONV2_STRIDE + 1)

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

// Top-level Tiny CNN hardware function for HLS.
void star_tracker_cnn(
	const pixel_t input_image[ST_INPUT_PIXELS],
	int *predicted_class
);

// Backward-compatible alias for older test harnesses.
// Older code may still call star_tracker_mlp; route it to the CNN top function.
inline void star_tracker_mlp(
	const pixel_t input_image[ST_INPUT_PIXELS],
	int *predicted_class
) {
	star_tracker_cnn(input_image, predicted_class);
}

#endif

