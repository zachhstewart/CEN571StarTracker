#pragma once

#include <cstdint>
#include "camera_driver.h"
#include "star_tracker.h"   // ST_Direction enum + ST_DIR_* constants

// ---------------------------------------------------------------------------
// Physical address map (from Vivado Address Editor)
// ---------------------------------------------------------------------------

// HLS CNN AXI-Lite control registers (GP0 AXI slave, 64 KB window)
#define CNN_CTRL_PHYS_BASE  0x40010000UL
#define CNN_CTRL_MAP_SIZE   0x00010000UL    // 64 KB

// DDR image buffer — must sit outside the Linux kernel's managed memory.
// The PYNQ Z2 has 512 MB DDR at 0x00000000–0x1FFFFFFF.
// We carve out 64 KB at the very top (0x1F000000); only 19 200 bytes are used.
// If your Linux device tree reserves a CMA region at a different address,
// change IMAGE_PHYS_BASE to match.
#define IMAGE_PHYS_BASE     0x1F000000UL
#define IMAGE_PHYS_SIZE     0x00010000UL    // 64 KB (19 200 bytes needed)

// ---------------------------------------------------------------------------
// AXI-Lite register offsets (from xstar_tracker_cnn_hw.h)
// ---------------------------------------------------------------------------
#define REG_AP_CTRL             0x00u   // bit0=ap_start, bit1=ap_done, bit2=ap_idle
#define REG_INPUT_IMAGE_LO      0x10u   // input image DDR address [31:0]
#define REG_INPUT_IMAGE_HI      0x14u   // input image DDR address [63:32]
#define REG_PREDICTED_CLASS     0x1Cu   // predicted class index (read)
#define REG_PREDICTED_CLASS_VLD 0x20u   // ap_vld flag for predicted_class (read/COR)

// AP_CTRL bit masks
#define AP_START    (1u << 0)
#define AP_DONE     (1u << 1)
#define AP_IDLE     (1u << 2)

// ---------------------------------------------------------------------------
// Inferencer handle
// ---------------------------------------------------------------------------
struct Inferencer {
    int               devmem_fd;    // fd for /dev/mem
    volatile uint32_t* ctrl_regs;  // mmap'd HLS AXI-Lite registers (virtual)
    uint8_t*          img_virt;    // mmap'd DDR image buffer (virtual)
    uint64_t          img_phys;    // physical address of image buffer
};

// Open /dev/mem, map the HLS control registers and the DDR image buffer.
// Returns true on success; prints error to stderr on failure.
bool infer_open(Inferencer& inf);

// Unmap everything and close /dev/mem.
void infer_close(Inferencer& inf);

// Capture one camera frame, run CNN inference, and return the predicted
// class index (0–5, matching ST_Direction). Returns -1 on error.
// The direction_gpio pins are driven by hardware — no extra call needed.
int  infer_run(Inferencer& inf, CameraDriver& cam);

// Human-readable label for a class index.
const char* infer_class_name(int cls);
