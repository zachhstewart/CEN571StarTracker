// PS-side orchestration layer for the star tracker FPGA pipeline.
//
// Runtime flow (all on the ARM Cortex-A9):
//   1. cam_capture_gray()   → writes 160×120 grayscale into a DDR buffer
//   2. Write buffer address → HLS CNN AXI-Lite registers (0x40010000)
//   3. Assert ap_start      → HLS IP reads image from DDR, runs inference
//   4. Poll ap_done         → wait for the IP to finish (~13 M cycles @ 100 MHz)
//   5. Read predicted_class → return to caller
//
// The direction_gpio pins (UP/DOWN/LEFT/RIGHT/FORWARD/BACKWARD) are driven
// directly by the HLS IP's hardware output port — no software action needed.

#include "run_inference.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

// Number of polling iterations before giving up on ap_done.
// The CNN takes ~13 106 868 cycles. At 100 MHz that is ~131 ms.
// Each poll iteration costs a few nanoseconds, so 10 000 000 is ~5 s — ample.
static constexpr int POLL_TIMEOUT = 10'000'000;

// ---------------------------------------------------------------------------
// Register helpers (byte-offset addressed, 32-bit words)
// ---------------------------------------------------------------------------

static inline void reg_write(volatile uint32_t* base, uint32_t byte_off, uint32_t val)
{
    base[byte_off >> 2] = val;
}

static inline uint32_t reg_read(const volatile uint32_t* base, uint32_t byte_off)
{
    return base[byte_off >> 2];
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool infer_open(Inferencer& inf)
{
    inf.devmem_fd = -1;
    inf.ctrl_regs = nullptr;
    inf.img_virt  = nullptr;
    inf.img_phys  = IMAGE_PHYS_BASE;

    // Open /dev/mem (requires root or CAP_SYS_RAWIO on PYNQ)
    inf.devmem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (inf.devmem_fd < 0) {
        perror("infer: open /dev/mem");
        return false;
    }

    // Map HLS AXI-Lite control registers
    inf.ctrl_regs = static_cast<volatile uint32_t*>(
        mmap(nullptr, CNN_CTRL_MAP_SIZE,
             PROT_READ | PROT_WRITE, MAP_SHARED,
             inf.devmem_fd, CNN_CTRL_PHYS_BASE));
    if (inf.ctrl_regs == MAP_FAILED) {
        fprintf(stderr, "infer: mmap ctrl_regs @ 0x%08lX failed: %s\n",
                CNN_CTRL_PHYS_BASE, strerror(errno));
        goto fail;
    }

    // Map DDR image buffer
    // NOTE: On kernels with CONFIG_STRICT_DEVMEM=y this will fail unless the
    // physical address is listed in /sys/kernel/debug/memblock/reserved or
    // marked as a CMA region in the device tree. PYNQ images typically have
    // CONFIG_STRICT_DEVMEM=n, so the plain /dev/mem approach works.
    inf.img_virt = static_cast<uint8_t*>(
        mmap(nullptr, IMAGE_PHYS_SIZE,
             PROT_READ | PROT_WRITE, MAP_SHARED,
             inf.devmem_fd, IMAGE_PHYS_BASE));
    if (inf.img_virt == MAP_FAILED) {
        fprintf(stderr, "infer: mmap DDR buffer @ 0x%08lX failed: %s\n",
                IMAGE_PHYS_BASE, strerror(errno));
        goto fail;
    }

    // Verify the HLS IP is idle before handing back to the caller
    {
        uint32_t ctrl = reg_read(inf.ctrl_regs, REG_AP_CTRL);
        if (!(ctrl & AP_IDLE)) {
            fprintf(stderr, "infer: WARNING – HLS IP is not idle at startup "
                            "(AP_CTRL=0x%08X). Is the bitstream loaded?\n", ctrl);
        }
    }

    return true;

fail:
    if (inf.ctrl_regs && inf.ctrl_regs != MAP_FAILED)
        munmap(const_cast<uint32_t*>(inf.ctrl_regs), CNN_CTRL_MAP_SIZE);
    close(inf.devmem_fd);
    inf.devmem_fd = -1;
    return false;
}

void infer_close(Inferencer& inf)
{
    if (inf.img_virt && inf.img_virt != MAP_FAILED)
        munmap(inf.img_virt, IMAGE_PHYS_SIZE);
    if (inf.ctrl_regs && inf.ctrl_regs != MAP_FAILED)
        munmap(const_cast<uint32_t*>(inf.ctrl_regs), CNN_CTRL_MAP_SIZE);
    if (inf.devmem_fd >= 0)
        close(inf.devmem_fd);
    inf.devmem_fd = -1;
    inf.ctrl_regs = nullptr;
    inf.img_virt  = nullptr;
}

int infer_run(Inferencer& inf, CameraDriver& cam)
{
    // Step 1 — Capture a fresh 160×120 grayscale frame directly into the
    //          DDR buffer that the HLS IP will read from.
    if (!cam_capture_gray(cam, inf.img_virt)) {
        fprintf(stderr, "infer: camera capture failed\n");
        return -1;
    }

    // Step 2 — Write the buffer's 64-bit physical address into the HLS
    //          input_image AXI-Lite registers (little-endian, two 32-bit halves).
    reg_write(inf.ctrl_regs, REG_INPUT_IMAGE_LO,
              static_cast<uint32_t>(inf.img_phys & 0xFFFFFFFFUL));
    reg_write(inf.ctrl_regs, REG_INPUT_IMAGE_HI,
              static_cast<uint32_t>(inf.img_phys >> 32));

    // Step 3 — Assert ap_start to begin inference.
    reg_write(inf.ctrl_regs, REG_AP_CTRL, AP_START);

    // Step 4 — Poll ap_done (bit 1 of AP_CTRL).
    //          ap_done is Clear-on-Read, so one read after it goes high clears it.
    int timeout = POLL_TIMEOUT;
    while (!(reg_read(inf.ctrl_regs, REG_AP_CTRL) & AP_DONE)) {
        if (--timeout == 0) {
            fprintf(stderr, "infer: timeout waiting for ap_done "
                            "(AP_CTRL=0x%08X)\n",
                    reg_read(inf.ctrl_regs, REG_AP_CTRL));
            return -1;
        }
    }

    // Step 5 — Read the predicted class index (0–5).
    int cls = static_cast<int>(
        reg_read(inf.ctrl_regs, REG_PREDICTED_CLASS));

    if (cls < 0 || cls >= ST_NUM_CLASSES) {
        fprintf(stderr, "infer: predicted_class %d out of range\n", cls);
        return -1;
    }

    // The direction_gpio pins are already driven by the HLS IP hardware —
    // no software GPIO write needed here.
    return cls;
}

const char* infer_class_name(int cls)
{
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
