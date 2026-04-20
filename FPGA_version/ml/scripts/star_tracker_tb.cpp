#include <iostream>
#include "star_tracker.h"

// Lightweight sanity testbench for C-simulation.
// This does not verify model accuracy on real data; it checks that the end-to-end
// inference pipeline executes and returns a valid class index.

static void clear_image(pixel_t input_image[ST_INPUT_PIXELS]) {
    for (int i = 0; i < ST_INPUT_PIXELS; ++i) {
        input_image[i] = 0;
    }
}

static void set_pixel(pixel_t input_image[ST_INPUT_PIXELS], int y, int x, pixel_t value) {
    input_image[y * ST_INPUT_WIDTH + x] = value;
}

static void build_cross_pattern(pixel_t input_image[ST_INPUT_PIXELS]) {
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

static void build_diagonal_pattern(pixel_t input_image[ST_INPUT_PIXELS]) {
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

static const char* direction_name(int cls) {
    switch (cls) {
        case ST_DIR_UP:       return "UP";
        case ST_DIR_DOWN:     return "DOWN";
        case ST_DIR_LEFT:     return "LEFT";
        case ST_DIR_RIGHT:    return "RIGHT";
        case ST_DIR_FORWARD:  return "FORWARD";
        case ST_DIR_BACKWARD: return "BACKWARD";
        default:              return "UNKNOWN";
    }
}

static bool run_case(const char* name, const pixel_t input_image[ST_INPUT_PIXELS]) {
    int           predicted_class = -1;
    unsigned char gpio_out        = 0;
    star_tracker_cnn(input_image, &predicted_class, &gpio_out);

    std::cout << name
              << ": predicted=" << predicted_class
              << " (" << direction_name(predicted_class) << ")"
              << "  gpio=0b";
    for (int b = 5; b >= 0; --b)
        std::cout << ((gpio_out >> b) & 1);

    // Verify one-hot: exactly one bit set, and it matches predicted_class.
    bool one_hot_ok = (gpio_out == ST_DIR_MASK(predicted_class));
    bool class_ok   = (predicted_class >= 0 && predicted_class < ST_NUM_CLASSES);

    if (class_ok && one_hot_ok) {
        std::cout << "  [PASS]" << std::endl;
        return true;
    }
    std::cout << "  [FAIL]" << std::endl;
    return false;
}

int main() {
    // This print makes it explicit which arithmetic path is being tested.
#if STAR_TRACKER_USE_FLOAT
    std::cout << "Running in FLOAT simulation mode." << std::endl;
#else
    std::cout << "Running in FIXED-POINT HLS mode." << std::endl;
#endif

    bool all_passed = true;
    pixel_t input_image[ST_INPUT_PIXELS];

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
