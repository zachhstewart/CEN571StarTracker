# arduino_led.xdc — PYNQ Z2 (xc7z020-clg400-1) Arduino header GPIO constraints
#
# Add this file to your Vivado project alongside your existing XDC.
#
# ── Required block design ─────────────────────────────────────────────────────
# In your Vivado block design, add:
#   1. AXI GPIO IP (name it e.g. "axi_gpio_arduino")
#      - GPIO Width: 14   (covers Arduino_IO[0:13])
#      - All Outputs: checked
#      - Enable Dual Channel: unchecked
#   2. Connect its AXI slave port to the PS AXI_GP0 interconnect
#   3. Note the base address assigned in the Address Editor (e.g. 0x41200000)
#      — this must match LED_GPIO_BASE in gpio_led_driver.h (after conversion
#        to the Linux sysfs GPIO number, found on boot via sysfs).
#   4. Export the GPIO output port and name it "Arduino_IO"
#
# ── LED wiring (female Arduino header, same layout as Arduino Uno) ─────────────
#   AR3  (D3)  → Backward LED  → AXI GPIO bit 3
#   AR5  (D5)  → Forward LED   → AXI GPIO bit 5
#   AR7  (D7)  → Down LED      → AXI GPIO bit 7
#   AR9  (D9)  → Up LED        → AXI GPIO bit 9
#   AR11 (D11) → Right LED     → AXI GPIO bit 11
#   AR13 (D13) → Left LED      → AXI GPIO bit 13
#
# Package pins from the PYNQ-Z2 master XDC (TUL Embedded / Xilinx board files).
# Verify these against your board revision before programming.
# ─────────────────────────────────────────────────────────────────────────────

# AR0  / D0
set_property -dict {PACKAGE_PIN T14  IOSTANDARD LVCMOS33} [get_ports {Arduino_IO[0]}]
# AR1  / D1
set_property -dict {PACKAGE_PIN U12  IOSTANDARD LVCMOS33} [get_ports {Arduino_IO[1]}]
# AR2  / D2
set_property -dict {PACKAGE_PIN U13  IOSTANDARD LVCMOS33} [get_ports {Arduino_IO[2]}]
# AR3  / D3  ── Backward LED (class 5)
set_property -dict {PACKAGE_PIN V13  IOSTANDARD LVCMOS33} [get_ports {Arduino_IO[3]}]
# AR4  / D4
set_property -dict {PACKAGE_PIN V15  IOSTANDARD LVCMOS33} [get_ports {Arduino_IO[4]}]
# AR5  / D5  ── Forward LED (class 4)
set_property -dict {PACKAGE_PIN T15  IOSTANDARD LVCMOS33} [get_ports {Arduino_IO[5]}]
# AR6  / D6
set_property -dict {PACKAGE_PIN R16  IOSTANDARD LVCMOS33} [get_ports {Arduino_IO[6]}]
# AR7  / D7  ── Down LED (class 1)
set_property -dict {PACKAGE_PIN U17  IOSTANDARD LVCMOS33} [get_ports {Arduino_IO[7]}]
# AR8  / D8
set_property -dict {PACKAGE_PIN V17  IOSTANDARD LVCMOS33} [get_ports {Arduino_IO[8]}]
# AR9  / D9  ── Up LED (class 0)
set_property -dict {PACKAGE_PIN V18  IOSTANDARD LVCMOS33} [get_ports {Arduino_IO[9]}]
# AR10 / D10
set_property -dict {PACKAGE_PIN T16  IOSTANDARD LVCMOS33} [get_ports {Arduino_IO[10]}]
# AR11 / D11 ── Right LED (class 3)
set_property -dict {PACKAGE_PIN R17  IOSTANDARD LVCMOS33} [get_ports {Arduino_IO[11]}]
# AR12 / D12
set_property -dict {PACKAGE_PIN P18  IOSTANDARD LVCMOS33} [get_ports {Arduino_IO[12]}]
# AR13 / D13 ── Left LED (class 2)
set_property -dict {PACKAGE_PIN N17  IOSTANDARD LVCMOS33} [get_ports {Arduino_IO[13]}]
