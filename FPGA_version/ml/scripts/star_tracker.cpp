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

void star_tracker_cnn(
    const pixel_t input_image[ST_INPUT_PIXELS],
    int *predicted_class
) {
    #pragma HLS INTERFACE bram port=input_image
    #pragma HLS INTERFACE s_axilite port=predicted_class bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    accum_t conv1_out[ST_CONV1_OUT_CH][ST_CONV1_OUT_H][ST_CONV1_OUT_W];
    accum_t conv2_out[ST_CONV2_OUT_CH][ST_CONV2_OUT_H][ST_CONV2_OUT_W];
    accum_t conv3_out[ST_CONV3_OUT_CH][ST_CONV3_OUT_H][ST_CONV3_OUT_W];
    accum_t pool_out[ST_CONV3_OUT_CH][ST_POOL_H][ST_POOL_W];
    accum_t fc1_out[ST_FC1_OUT];
    accum_t logits[ST_NUM_CLASSES];

    // ---- Memory mapping -------------------------------------------------------
    // Default HLS behaviour maps every local array to distributed RAM (LUT-RAM).
    // On the XC7Z020 (PYNQ Z2) only 17,400 LUT-RAM sites exist, but the three
    // conv feature maps alone need ~44,000 — causing a DRC UTLZ-1 over-use error.
    //
    // Fix: pin large buffers to Block RAM (140 x 36 Kb tiles on XC7Z020).
    // Estimated BRAM36 usage after the pragmas below:
    //   conv1_out  76800 x 21 b  →  ~75 BRAM36
    //   conv2_out  38400 x 31 b  →  ~38 BRAM36
    //   conv3_out   1920 x 31 b  →  ~ 2 BRAM36
    //   pool_out     960 x 28 b  →  ~ 1 BRAM36
    //   ─────────────────────────────────────────
    //   Total                       ~116 / 140 BRAM36 (83 %)
    //
    // Small buffers (fc1_out, logits) are explicitly kept in LUT-RAM / FFs so
    // they do NOT accidentally claim a full BRAM tile in future HLS versions.
    // -------------------------------------------------------------------------
    #pragma HLS BIND_STORAGE variable=conv1_out type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=conv2_out type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=conv3_out type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=pool_out  type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=fc1_out   type=ram_1p impl=lutram
    #pragma HLS BIND_STORAGE variable=logits    type=ram_1p impl=lutram

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
                                   * get_input_value(input_image[input_idx(in_y, in_x)]);
#else
                            accum_t w = conv1_w[conv1_w_idx(oc, 0, ky, kx)];
                            sum += (w * to_fixed_input(input_image[input_idx(in_y, in_x)])) >> ST_FRAC_BITS;
#endif
                        }
                    }
                }
                conv1_out[oc][oy][ox] = (sum > 0) ? sum : (accum_t)0;
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
                                       * conv1_out[ic][in_y][in_x];
#else
                                sum += (conv2_w[conv2_w_idx(oc, ic, ky, kx)] * conv1_out[ic][in_y][in_x]) >> ST_FRAC_BITS;
#endif
                            }
                        }
                    }
                }
                conv2_out[oc][oy][ox] = (sum > 0) ? sum : (accum_t)0;
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
                                       * conv2_out[ic][in_y][in_x];
#else
                                sum += (conv3_w[conv3_w_idx(oc, ic, ky, kx)] * conv2_out[ic][in_y][in_x]) >> ST_FRAC_BITS;
#endif
                            }
                        }
                    }
                }
                conv3_out[oc][oy][ox] = (sum > 0) ? sum : (accum_t)0;
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
                        sum += conv3_out[c][fy][fx];
                    }
                }
                pool_out[c][ph][pw] = sum / (ST_POOL_BIN_H * ST_POOL_BIN_W);
            }
        }
    }

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
