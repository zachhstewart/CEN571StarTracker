#include <iostream>
#include "star_tracker.h"

// Lightweight sanity testbench for C-simulation.
// This does not verify model accuracy on real data; it checks that the end-to-end
// inference pipeline executes and returns a valid class index.

static void clear_image(pixel_word_t input_image[ST_INPUT_WORDS]) {
    for (int i = 0; i < ST_INPUT_WORDS; ++i)
        input_image[i] = 0;
}

static void set_pixel(pixel_word_t input_image[ST_INPUT_WORDS], int y, int x, pixel_t value) {
#if STAR_TRACKER_USE_FLOAT
    input_image[y * ST_INPUT_WIDTH + x] = value;
#else
    int flat = y * ST_INPUT_WIDTH + x;
    int byte_pos = (flat & 3) * 8;
    pixel_word_t mask = ~(pixel_word_t(0xFF) << byte_pos);
    input_image[flat >> 2] = (input_image[flat >> 2] & mask) | (pixel_word_t(value) << byte_pos);
#endif
}

static void build_cross_pattern(pixel_word_t input_image[ST_INPUT_WORDS]) {
    // Synthetic plus-sign shape near image center.
    clear_image(input_image);
    const int cx = ST_INPUT_WIDTH / 2;
    const int cy = ST_INPUT_HEIGHT / 2;

    for (int dx = -6; dx <= 6; ++dx) {
        if (cx + dx >= 0 && cx + dx < ST_INPUT_WIDTH) {
#if STAR_TRACKER_USE_FLOAT
            set_pixel(input_image, cy, cx + dx, 1.0f);
#else
            set_pixel(input_image, cy, cx + dx, 1);
#endif
        }
    }

    for (int dy = -6; dy <= 6; ++dy) {
        if (cy + dy >= 0 && cy + dy < ST_INPUT_HEIGHT) {
#if STAR_TRACKER_USE_FLOAT
            set_pixel(input_image, cy + dy, cx, 1.0f);
#else
            set_pixel(input_image, cy + dy, cx, 1);
#endif
        }
    }
}

static void build_diagonal_pattern(pixel_word_t input_image[ST_INPUT_WORDS]) {
    // Sparse diagonal points to produce a very different activation pattern.
    clear_image(input_image);
    int length = (ST_INPUT_WIDTH < ST_INPUT_HEIGHT) ? ST_INPUT_WIDTH : ST_INPUT_HEIGHT;
    for (int i = 0; i < length; i += 8) {
#if STAR_TRACKER_USE_FLOAT
        set_pixel(input_image, i, i, 1.0f);
#else
        set_pixel(input_image, i, i, 1);
#endif
    }
}

static bool run_case(const char* name, const pixel_word_t input_image[ST_INPUT_WORDS]) {
    int predicted_class = -1;
    star_tracker_cnn(input_image, &predicted_class);

    std::cout << name << ": predicted=" << predicted_class;

    if (predicted_class < 0 || predicted_class >= ST_NUM_CLASSES) {
        std::cout << "  [FAIL]" << std::endl;
        return false;
    }

    std::cout << "  [PASS]" << std::endl;
    return true;
}

int main() {
    // This print makes it explicit which arithmetic path is being tested.
#if STAR_TRACKER_USE_FLOAT
    std::cout << "Running in FLOAT simulation mode." << std::endl;
#else
    std::cout << "Running in FIXED-POINT HLS mode." << std::endl;
#endif

    bool all_passed = true;
    pixel_word_t input_image[ST_INPUT_WORDS];

    // A few synthetic patterns are enough for smoke testing the kernel interface.
    build_cross_pattern(input_image);
    all_passed &= run_case("Cross pattern", input_image);

    build_diagonal_pattern(input_image);
    all_passed &= run_case("Diagonal pattern", input_image);

    clear_image(input_image);
    all_passed &= run_case("All-zero image", input_image);

    if (all_passed) {
        std::cout << "All testbench cases passed." << std::endl;
        return 0;
    }

    std::cout << "One or more testbench cases failed." << std::endl;
    return 1;
}
