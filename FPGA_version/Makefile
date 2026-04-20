CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra
# ml/scripts is on the include path for star_tracker.h / ST_Direction enum
INCLUDES := -Iml/scripts

.PHONY: all test clean

# ── Full pipeline: camera → HLS inference (run on PYNQ Z2 with bitstream loaded)
all: startracker run_test

startracker: camera_driver.cpp run_inference.cpp main.cpp \
             camera_driver.h   run_inference.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	    camera_driver.cpp run_inference.cpp main.cpp \
	    -o $@

# ── Camera-only smoke test (no bitstream required)
run_test: camera_driver.cpp camera_driver_test.cpp camera_driver.h
	$(CXX) $(CXXFLAGS) camera_driver.cpp camera_driver_test.cpp -o $@

clean:
	rm -f startracker run_test gray_out.bin
