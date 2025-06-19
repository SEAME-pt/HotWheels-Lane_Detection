# Makefile for autonomous_jetson
# Compila todos os arquivos .cpp em src/ e gera o executável main

CXX = c++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g -fopenmp

# CUDA paths for Jetson
CUDA_PATH = /usr/local/cuda
CUDA_INCLUDE = $(CUDA_PATH)/include
CUDA_LIB = $(CUDA_PATH)/lib64

# Check if CUDA is available
CUDA_AVAILABLE = $(shell test -d $(CUDA_INCLUDE) && echo "yes" || echo "no")

# Qt5 paths
QT5_CFLAGS = $(shell pkg-config --cflags Qt5Core Qt5Gui 2>/dev/null || echo "")
QT5_LIBS = $(shell pkg-config --libs Qt5Core Qt5Gui 2>/dev/null || echo "")

# Include paths
INCLUDE_PATHS = -I. \
                -Icar_controls/includes \
                -Icar_controls/includes/inference \
                -Icar_controls/includes/objectDetection \
                -I/usr/include/eigen3 \
                -I/usr/include/opencv4

# Add CUDA include if available
ifeq ($(CUDA_AVAILABLE), yes)
    INCLUDE_PATHS += -I$(CUDA_INCLUDE)
    CUDA_LIBS = -L$(CUDA_LIB) -lcudart
    CXXFLAGS += -DCUDA_AVAILABLE
    $(info CUDA found - enabling CUDA support)
else
    CUDA_LIBS = 
    $(info CUDA not found - compiling without CUDA support)
endif

# Source files
CAR_CONTROLS_SRC = car_controls/sources/MPCOptimizer.cpp \
                   car_controls/sources/MPCPlanner.cpp \
                   car_controls/sources/Polyfitter.cpp

# Additional sources for full car control (with Qt)
CAR_CONTROLS_FULL_SRC = $(CAR_CONTROLS_SRC) \
                        car_controls/sources/EngineController.cpp \
                        car_controls/sources/JoysticksController.cpp \
                        car_controls/sources/PeripheralController.cpp

# Source files for integrated main
INTEGRATED_SRC = car_controls/sources/MPCOptimizer.cpp \
                 car_controls/sources/MPCPlanner.cpp \
                 car_controls/sources/Polyfitter.cpp \
                 car_controls/sources/ControlsManager.cpp \
                 car_controls/sources/EngineController.cpp \
                 car_controls/sources/JoysticksController.cpp \
                 car_controls/sources/PeripheralController.cpp

# Add ZeroMQ and other dependencies that ControlsManager needs
INTEGRATED_SRC += car_controls/../ZeroMQ/Publisher.cpp \
                  car_controls/../ZeroMQ/Subscriber.cpp \
                  car_controls/sources/inference/CameraStreamer.cpp

# Object files
CAR_CONTROLS_OBJ = $(CAR_CONTROLS_SRC:.cpp=.o)
CAR_CONTROLS_FULL_OBJ = $(CAR_CONTROLS_FULL_SRC:.cpp=.o)
INTEGRATED_OBJ = $(INTEGRATED_SRC:.cpp=.o)

# Libraries - order matters for linking!
OPENCV_LIBS = $(shell pkg-config --libs opencv4 || pkg-config --libs opencv)
MLPACK_LIBS = -lmlpack -larmadillo -llapack -lblas
FILESYSTEM_LIBS = -lstdc++fs
BASIC_LIBS = -lnlopt -pthread $(CUDA_LIBS) $(MLPACK_LIBS) $(FILESYSTEM_LIBS)
SDL_LIBS = -lSDL2

# Targets
TARGET_BASIC = main
TARGET_VISUAL = mpc_test
TARGET_SIMPLE = mpc_simple_test
TARGET_REAL = mpc_real_test

all: $(TARGET_BASIC)

# MOC processing for Qt (needed for signals/slots)
main.moc: main.cpp
	$(shell pkg-config --variable=host_bins Qt5Core)/moc main.cpp -o main.moc

# Basic main target
$(TARGET_BASIC): main.o $(INTEGRATED_OBJ) main.moc
	$(CXX) $(CXXFLAGS) $(QT5_CFLAGS) main.o $(INTEGRATED_OBJ) -o $@ $(BASIC_LIBS) $(OPENCV_LIBS) $(QT5_LIBS) $(SDL_LIBS)

# Visual test target (with full car control - requires Qt)
$(TARGET_VISUAL): mpc_test_main.o $(CAR_CONTROLS_FULL_OBJ)
	$(CXX) $(CXXFLAGS) $(QT5_CFLAGS) $^ -o $@ $(BASIC_LIBS) $(OPENCV_LIBS) $(QT5_LIBS) $(SDL_LIBS)

# Simple visual test (without car control - no Qt needed)
$(TARGET_SIMPLE): mpc_simple_test.o $(CAR_CONTROLS_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(BASIC_LIBS) $(OPENCV_LIBS)

# Real car control test (with SDL joystick - no Qt needed)
$(TARGET_REAL): mpc_real_test.o $(CAR_CONTROLS_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(BASIC_LIBS) $(OPENCV_LIBS) $(SDL_LIBS)

# Object file rules
main.o: main.cpp main.moc
	$(CXX) $(CXXFLAGS) $(QT5_CFLAGS) $(INCLUDE_PATHS) -c main.cpp -o $@

mpc_test_main.o: mpc_test_main.cpp
	$(CXX) $(CXXFLAGS) $(QT5_CFLAGS) $(INCLUDE_PATHS) -c $< -o $@

mpc_simple_test.o: mpc_simple_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATHS) -c $< -o $@

mpc_real_test.o: mpc_real_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATHS) -c $< -o $@

car_controls/sources/%.o: car_controls/sources/%.cpp
	$(CXX) $(CXXFLAGS) $(QT5_CFLAGS) $(INCLUDE_PATHS) -c $< -o $@

# Special targets
visual: $(TARGET_VISUAL)
simple: $(TARGET_SIMPLE)
real: $(TARGET_REAL)

clean:
	rm -f $(INTEGRATED_OBJ) main.o main.moc $(TARGET_BASIC) $(TARGET_VISUAL) $(TARGET_SIMPLE) $(TARGET_REAL)

install-deps:
	sudo apt update
	sudo apt install -y libopencv-dev libeigen3-dev libnlopt-dev libmlpack-dev libarmadillo-dev libsdl2-dev qtbase5-dev

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@if [ -d "$(CUDA_INCLUDE)" ]; then \
		echo "CUDA found at $(CUDA_PATH)"; \
		echo "CUDA include: $(CUDA_INCLUDE)"; \
		echo "CUDA lib: $(CUDA_LIB)"; \
	else \
		echo "CUDA not found. Expected at $(CUDA_PATH)"; \
		echo "You may need to install CUDA or adjust CUDA_PATH"; \
	fi

# Check library availability
check-libs:
	@echo "=== Checking Library Availability ==="
	@echo "OpenCV:"
	@pkg-config --exists opencv4 && echo "  opencv4: OK" || (pkg-config --exists opencv && echo "  opencv: OK" || echo "  opencv: NOT FOUND")
	@echo "MLPack:"
	@pkg-config --exists mlpack && echo "  mlpack: OK" || echo "  mlpack: NOT FOUND (using -lmlpack)"
	@echo "Armadillo:"
	@pkg-config --exists armadillo && echo "  armadillo: OK" || echo "  armadillo: NOT FOUND (using -larmadillo)"
	@echo "NLOpt:"
	@ldconfig -p | grep nlopt > /dev/null && echo "  nlopt: OK" || echo "  nlopt: NOT FOUND"
	@echo "Filesystem:"
	@ldconfig -p | grep stdc++fs > /dev/null && echo "  stdc++fs: OK" || echo "  stdc++fs: NOT FOUND"

# Debug compilation - show detailed info
debug-compile:
	@echo "=== Debug Compilation Info ==="
	@echo "CXX: $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "INCLUDE_PATHS: $(INCLUDE_PATHS)"
	@echo "CUDA_AVAILABLE: $(CUDA_AVAILABLE)"
	@echo "CUDA_LIBS: $(CUDA_LIBS)"
	@echo "OPENCV_LIBS: $(OPENCV_LIBS)"
	@echo "MLPACK_LIBS: $(MLPACK_LIBS)"
	@echo "FILESYSTEM_LIBS: $(FILESYSTEM_LIBS)"
	@echo "BASIC_LIBS: $(BASIC_LIBS)"
	@echo "=== Testing individual file compilation ==="
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATHS) -c car_controls/sources/Polyfitter.cpp -o /tmp/test_polyfitter.o -v
	@echo "Polyfitter compilation successful"

.PHONY: all visual simple real clean install-deps check-cuda check-libs debug-compile
