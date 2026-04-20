CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra

# Primary targets
.PHONY: all test clean

all: camera_driver_test

camera_driver_test: camera_driver.cpp camera_driver_test.cpp camera_driver.h
	$(CXX) $(CXXFLAGS) camera_driver.cpp camera_driver_test.cpp -o $@

clean:
	rm -f camera_driver_test gray_out.bin
