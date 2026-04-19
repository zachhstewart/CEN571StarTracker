#include "star_tracker.h"

#if STAR_TRACKER_USE_FLOAT
// Header stores quantized ints, so float mode dequantizes back to approximate real weights.
static inline float dequantize_weight(weight_storage_t x) {
    return static_cast<float>(x) / static_cast<float>(1 << ST_FRAC_BITS);
}

static inline float get_input_value(pixel_t x) {
    return x;
}
#else
// Fixed-point path: convert integer pixels into Q(frac_bits) representation.
static inline accum_t to_fixed_input(pixel_t x) {
    return static_cast<accum_t>(x) << ST_FRAC_BITS;
}
#endif

static inline int conv1_w_idx(int oc, int ic, int ky, int kx) {
    return (((oc * ST_CONV1_IN_CH + ic) * ST_CONV1_K + ky) * ST_CONV1_K + kx);
}

static inline int conv2_w_idx(int oc, int ic, int ky, int kx) {
    return (((oc * ST_CONV2_IN_CH + ic) * ST_CONV2_K + ky) * ST_CONV2_K + kx);
}

static inline int input_idx(int y, int x) {
    return y * ST_INPUT_WIDTH + x;
}

void star_tracker_cnn(
    const pixel_t input_image[ST_INPUT_PIXELS],
    int *predicted_class
) {
    // HLS interfaces:
    // - Input image comes from BRAM.
    // - predicted_class and return are AXI-Lite control registers.
    #pragma HLS INTERFACE bram port=input_image
    #pragma HLS INTERFACE s_axilite port=predicted_class bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    accum_t conv1_out[ST_CONV1_OUT_CH][ST_CONV1_OUT_H][ST_CONV1_OUT_W];
    accum_t conv2_out[ST_CONV2_OUT_CH][ST_CONV2_OUT_H][ST_CONV2_OUT_W];
    accum_t gap[ST_CONV2_OUT_CH];
    accum_t logits[ST_NUM_CLASSES];

    // Stage 1: Conv1 + ReLU
    for (int oc = 0; oc < ST_CONV1_OUT_CH; ++oc) {
        for (int oy = 0; oy < ST_CONV1_OUT_H; ++oy) {
            for (int ox = 0; ox < ST_CONV1_OUT_W; ++ox) {
#if STAR_TRACKER_USE_FLOAT
                accum_t sum = dequantize_weight(conv1_b[oc]);
#else
                accum_t sum = conv1_b[oc];
#endif
                for (int ky = 0; ky < ST_CONV1_K; ++ky) {
                    for (int kx = 0; kx < ST_CONV1_K; ++kx) {
                        int in_y = oy * ST_CONV1_STRIDE + ky - ST_CONV1_PAD;
                        int in_x = ox * ST_CONV1_STRIDE + kx - ST_CONV1_PAD;
                        if (in_y >= 0 && in_y < ST_INPUT_HEIGHT && in_x >= 0 && in_x < ST_INPUT_WIDTH) {
#if STAR_TRACKER_USE_FLOAT
                            accum_t w = dequantize_weight(conv1_w[conv1_w_idx(oc, 0, ky, kx)]);
                            accum_t v = get_input_value(input_image[input_idx(in_y, in_x)]);
                            sum += w * v;
#else
                            accum_t w = conv1_w[conv1_w_idx(oc, 0, ky, kx)];
                            accum_t v = to_fixed_input(input_image[input_idx(in_y, in_x)]);
                            sum += (w * v) >> ST_FRAC_BITS;
#endif
                        }
                    }
                }
                conv1_out[oc][oy][ox] = (sum > 0) ? sum : (accum_t)0;
            }
        }
    }

    // Stage 2: Conv2 + ReLU
    for (int oc = 0; oc < ST_CONV2_OUT_CH; ++oc) {
        for (int oy = 0; oy < ST_CONV2_OUT_H; ++oy) {
            for (int ox = 0; ox < ST_CONV2_OUT_W; ++ox) {
#if STAR_TRACKER_USE_FLOAT
                accum_t sum = dequantize_weight(conv2_b[oc]);
#else
                accum_t sum = conv2_b[oc];
#endif
                for (int ic = 0; ic < ST_CONV2_IN_CH; ++ic) {
                    for (int ky = 0; ky < ST_CONV2_K; ++ky) {
                        for (int kx = 0; kx < ST_CONV2_K; ++kx) {
                            int in_y = oy * ST_CONV2_STRIDE + ky - ST_CONV2_PAD;
                            int in_x = ox * ST_CONV2_STRIDE + kx - ST_CONV2_PAD;
                            if (in_y >= 0 && in_y < ST_CONV1_OUT_H && in_x >= 0 && in_x < ST_CONV1_OUT_W) {
#if STAR_TRACKER_USE_FLOAT
                                accum_t w = dequantize_weight(conv2_w[conv2_w_idx(oc, ic, ky, kx)]);
                                sum += w * conv1_out[ic][in_y][in_x];
#else
                                accum_t w = conv2_w[conv2_w_idx(oc, ic, ky, kx)];
                                sum += (w * conv1_out[ic][in_y][in_x]) >> ST_FRAC_BITS;
#endif
                            }
                        }
                    }
                }
                conv2_out[oc][oy][ox] = (sum > 0) ? sum : (accum_t)0;
            }
        }
    }

    // Stage 3: Global Average Pooling (GAP)
    // This collapses each channel to one value and greatly reduces FC parameters.
    const int gap_count = ST_CONV2_OUT_H * ST_CONV2_OUT_W;
    for (int c = 0; c < ST_CONV2_OUT_CH; ++c) {
#if STAR_TRACKER_USE_FLOAT
        accum_t sum = 0.0f;
#else
        accum_t sum = 0;
#endif
        for (int y = 0; y < ST_CONV2_OUT_H; ++y) {
            for (int x = 0; x < ST_CONV2_OUT_W; ++x) {
                sum += conv2_out[c][y][x];
            }
        }
        gap[c] = sum / gap_count;
    }

    // Stage 4: Final fully-connected classifier.
    for (int cls = 0; cls < ST_NUM_CLASSES; ++cls) {
#if STAR_TRACKER_USE_FLOAT
        accum_t sum = dequantize_weight(fc_b[cls]);
#else
        accum_t sum = fc_b[cls];
#endif
        for (int c = 0; c < ST_CONV2_OUT_CH; ++c) {
            int idx = cls * ST_CONV2_OUT_CH + c;
#if STAR_TRACKER_USE_FLOAT
            accum_t w = dequantize_weight(fc_w[idx]);
            sum += w * gap[c];
#else
            accum_t w = fc_w[idx];
            sum += (w * gap[c]) >> ST_FRAC_BITS;
#endif
        }
        logits[cls] = sum;
    }

    // Stage 5: Argmax for predicted attitude class.
    int best_class = 0;
    accum_t best_logit = logits[0];
    for (int cls = 1; cls < ST_NUM_CLASSES; ++cls) {
        if (logits[cls] > best_logit) {
            best_logit = logits[cls];
            best_class = cls;
        }
    }

    *predicted_class = best_class;
}
