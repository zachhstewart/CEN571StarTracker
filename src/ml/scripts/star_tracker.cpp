#include "star_tracker.h"

#if STAR_TRACKER_USE_FLOAT
static inline float dequantize_weight(weight_storage_t x) {
    return static_cast<float>(x) / static_cast<float>(1 << ST_FRAC_BITS);
}
static inline float get_input_value(pixel_t x) { return x; }
#else
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
static inline int conv3_w_idx(int oc, int ic, int ky, int kx) {
    return (((oc * ST_CONV3_IN_CH + ic) * ST_CONV3_K + ky) * ST_CONV3_K + kx);
}
static inline int input_idx(int y, int x) { return y * ST_INPUT_WIDTH + x; }

// Index for flattened feature maps using ping-pong buffers
static inline int fm_idx(int c, int y, int x, int H, int W) {
    return (c * H + y) * W + x;
}

void star_tracker_cnn(
    const pixel_word_t input_image[ST_INPUT_WORDS],
    int *predicted_class
) {
    #pragma HLS INTERFACE bram port=input_image
    #pragma HLS INTERFACE s_axilite port=predicted_class bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    // Hard cap on DSP usage — forces multipliers to be time-multiplexed rather
    // than instantiated in parallel. Keeps DSP count well under the 220 limit.
    #pragma HLS ALLOCATION operation instances=mul limit=4

    accum_t buffer_A[ST_MAX_BUFFER_SIZE];
    accum_t buffer_B[ST_BUFFER_B_SIZE];
    accum_t pool_out[ST_CONV3_OUT_CH][ST_POOL_H][ST_POOL_W];
    accum_t fc1_out[ST_FC1_OUT];
    accum_t logits[ST_NUM_CLASSES];

    // Force all large intermediate arrays to block RAM.
    // By using just two buffers in ping-pong, BRAM footprint is minimized.
    #pragma HLS bind_storage variable=buffer_A type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=buffer_B type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=pool_out  type=RAM_2P impl=bram

    // ---- Stage 1: Conv1 + ReLU (BN folded into weights) ----
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
                            sum += dequantize_weight(conv1_w[conv1_w_idx(oc, 0, ky, kx)])
                                   * get_input_value(unpack_pixel(input_image, input_idx(in_y, in_x)));
#else
                            accum_t w = conv1_w[conv1_w_idx(oc, 0, ky, kx)];
                            sum += (w * to_fixed_input(unpack_pixel(input_image, input_idx(in_y, in_x)))) >> ST_FRAC_BITS;
#endif
                        }
                    }
                }
                buffer_A[fm_idx(oc, oy, ox, ST_CONV1_OUT_H, ST_CONV1_OUT_W)] = (sum > 0) ? sum : (accum_t)0;
            }
        }
    }

    // ---- Stage 2: Conv2 + ReLU ----
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
                                sum += dequantize_weight(conv2_w[conv2_w_idx(oc, ic, ky, kx)])
                                       * buffer_A[fm_idx(ic, in_y, in_x, ST_CONV1_OUT_H, ST_CONV1_OUT_W)];
#else
                                sum += (conv2_w[conv2_w_idx(oc, ic, ky, kx)] * buffer_A[fm_idx(ic, in_y, in_x, ST_CONV1_OUT_H, ST_CONV1_OUT_W)]) >> ST_FRAC_BITS;
#endif
                            }
                        }
                    }
                }
                buffer_B[fm_idx(oc, oy, ox, ST_CONV2_OUT_H, ST_CONV2_OUT_W)] = (sum > 0) ? sum : (accum_t)0;
            }
        }
    }

    // ---- Stage 3: Conv3 + ReLU ----
    for (int oc = 0; oc < ST_CONV3_OUT_CH; ++oc) {
        for (int oy = 0; oy < ST_CONV3_OUT_H; ++oy) {
            for (int ox = 0; ox < ST_CONV3_OUT_W; ++ox) {
#if STAR_TRACKER_USE_FLOAT
                accum_t sum = dequantize_weight(conv3_b[oc]);
#else
                accum_t sum = conv3_b[oc];
#endif
                for (int ic = 0; ic < ST_CONV3_IN_CH; ++ic) {
                    for (int ky = 0; ky < ST_CONV3_K; ++ky) {
                        for (int kx = 0; kx < ST_CONV3_K; ++kx) {
                            int in_y = oy * ST_CONV3_STRIDE + ky - ST_CONV3_PAD;
                            int in_x = ox * ST_CONV3_STRIDE + kx - ST_CONV3_PAD;
                            if (in_y >= 0 && in_y < ST_CONV2_OUT_H && in_x >= 0 && in_x < ST_CONV2_OUT_W) {
#if STAR_TRACKER_USE_FLOAT
                                sum += dequantize_weight(conv3_w[conv3_w_idx(oc, ic, ky, kx)])
                                       * buffer_B[fm_idx(ic, in_y, in_x, ST_CONV2_OUT_H, ST_CONV2_OUT_W)];
#else
                                sum += (conv3_w[conv3_w_idx(oc, ic, ky, kx)] * buffer_B[fm_idx(ic, in_y, in_x, ST_CONV2_OUT_H, ST_CONV2_OUT_W)]) >> ST_FRAC_BITS;
#endif
                            }
                        }
                    }
                }
                // Write Conv3 output back to buffer_A
                buffer_A[fm_idx(oc, oy, ox, ST_CONV3_OUT_H, ST_CONV3_OUT_W)] = (sum > 0) ? sum : (accum_t)0;
            }
        }
    }

    // ---- Stage 4: Spatial Average Pool (3×5) ----
    // Each bin covers ST_POOL_BIN_H × ST_POOL_BIN_W pixels (integer sizes).
    for (int c = 0; c < ST_CONV3_OUT_CH; ++c) {
        for (int ph = 0; ph < ST_POOL_H; ++ph) {
            for (int pw = 0; pw < ST_POOL_W; ++pw) {
#if STAR_TRACKER_USE_FLOAT
                accum_t sum = 0.0f;
#else
                accum_t sum = 0;
#endif
                for (int dy = 0; dy < ST_POOL_BIN_H; ++dy) {
                    for (int dx = 0; dx < ST_POOL_BIN_W; ++dx) {
                        int fy = ph * ST_POOL_BIN_H + dy;
                        int fx = pw * ST_POOL_BIN_W + dx;
                        sum += buffer_A[fm_idx(c, fy, fx, ST_CONV3_OUT_H, ST_CONV3_OUT_W)];
                    }
                }
                pool_out[c][ph][pw] = sum / (ST_POOL_BIN_H * ST_POOL_BIN_W);
            }
        }
    }

    // ---- Calculate Star Density (Mean Brightness) ----
#if STAR_TRACKER_USE_FLOAT
    accum_t star_density = 0.0f;
#else
    accum_t star_density = 0;
#endif
    for (int i = 0; i < ST_INPUT_PIXELS; ++i) {
#if STAR_TRACKER_USE_FLOAT
        star_density += get_input_value(unpack_pixel(input_image, i));
#else
        star_density += to_fixed_input(unpack_pixel(input_image, i));
#endif
    }
    star_density /= ST_INPUT_PIXELS;

    // ---- Stage 5: FC1 + ReLU ----
    for (int j = 0; j < ST_FC1_OUT; ++j) {
#if STAR_TRACKER_USE_FLOAT
        accum_t sum = dequantize_weight(fc1_b[j]);
#else
        accum_t sum = fc1_b[j];
#endif
        int idx = 0;
        for (int c = 0; c < ST_CONV3_OUT_CH; ++c) {
            for (int ph = 0; ph < ST_POOL_H; ++ph) {
                for (int pw = 0; pw < ST_POOL_W; ++pw) {
#if STAR_TRACKER_USE_FLOAT
                    sum += dequantize_weight(fc1_w[j * ST_FC1_IN + idx]) * pool_out[c][ph][pw];
#else
                    sum += (fc1_w[j * ST_FC1_IN + idx] * pool_out[c][ph][pw]) >> ST_FRAC_BITS;
#endif
                    ++idx;
                }
            }
        }
        
#if STAR_TRACKER_USE_FLOAT
        sum += dequantize_weight(fc1_w[j * ST_FC1_IN + idx]) * star_density;
#else
        sum += (fc1_w[j * ST_FC1_IN + idx] * star_density) >> ST_FRAC_BITS;
#endif

        fc1_out[j] = (sum > 0) ? sum : (accum_t)0;
    }

    // ---- Stage 6: FC2 (classifier) ----
    for (int cls = 0; cls < ST_NUM_CLASSES; ++cls) {
#if STAR_TRACKER_USE_FLOAT
        accum_t sum = dequantize_weight(fc2_b[cls]);
#else
        accum_t sum = fc2_b[cls];
#endif
        for (int j = 0; j < ST_FC1_OUT; ++j) {
#if STAR_TRACKER_USE_FLOAT
            sum += dequantize_weight(fc2_w[cls * ST_FC1_OUT + j]) * fc1_out[j];
#else
            sum += (fc2_w[cls * ST_FC1_OUT + j] * fc1_out[j]) >> ST_FRAC_BITS;
#endif
        }
        logits[cls] = sum;
    }

    // ---- Stage 7: Argmax ----
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
