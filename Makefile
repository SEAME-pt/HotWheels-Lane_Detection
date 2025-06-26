# Makefile for autonomous_jetson

CXX = c++
CXXFLAGS = -std=c++17 -Wall -Wextra -g -fopenmp -fPIC
CXXFLAGS += -fprofile-arcs -ftest-coverage -O2 -g
LDFLAGS  += -fprofile-arcs -ftest-coverage


# CUDA paths for Jetson
CUDA_PATH = /usr/local/cuda
CUDA_INCLUDE = $(CUDA_PATH)/include
CUDA_LIB = $(CUDA_PATH)/lib64

# Check if CUDA is available (different logic for cross-compilation)
ifneq ($(CXX),aarch64-linux-gnu-g++)
    # Native compilation - check local CUDA
    CUDA_AVAILABLE = $(shell test -d $(CUDA_INCLUDE) && echo "yes" || echo "no")
else
    # Cross-compilation - assume CUDA is available on Jetson target
    CUDA_AVAILABLE = yes
endif

# Qt5 paths - different for cross-compilation
ifneq ($(CXX),aarch64-linux-gnu-g++)
    # Native compilation
    QT5_CFLAGS = $(shell pkg-config --cflags Qt5Core Qt5Gui Qt5Widgets 2>/dev/null || echo "")
    QT5_LIBS = $(shell pkg-config --libs Qt5Core Qt5Gui Qt5Widgets 2>/dev/null || echo "")
else
    # Cross-compilation - use sysroot Qt5
    QT5_CFLAGS = -DQT_WIDGETS_LIB -DQT_GUI_LIB -DQT_CORE_LIB \
                 -I/home/michel/new_qtjetson/sysroot/usr/include/aarch64-linux-gnu/qt5/QtWidgets \
                 -I/home/michel/new_qtjetson/sysroot/usr/include/aarch64-linux-gnu/qt5 \
                 -I/home/michel/new_qtjetson/sysroot/usr/include/aarch64-linux-gnu/qt5/QtGui \
                 -I/home/michel/new_qtjetson/sysroot/usr/include/aarch64-linux-gnu/qt5/QtCore
    QT5_LIBS = -L/home/michel/new_qtjetson/sysroot/usr/lib/aarch64-linux-gnu \
               -lQt5Widgets -lQt5Gui -lQt5Core
endif

# Include paths
INCLUDE_PATHS = -I. \
				-IZeroMQ \
                -Icar_controls/includes \
                -Icar_controls/includes/inference \
                -Icar_controls/includes/objectDetection \
                -I/usr/include/eigen3 \
                -I/usr/include/opencv4

# Check if OpenCV has CUDA support
OPENCV_CUDA_AVAILABLE = $(shell pkg-config --exists opencv4 && pkg-config --cflags opencv4 | grep -q cuda && echo "yes" || echo "no")

ifeq ($(CUDA_AVAILABLE), yes)
    INCLUDE_PATHS += -I$(CUDA_INCLUDE)
    CUDA_LIBS = -L$(CUDA_LIB) -lcudart
    CXXFLAGS += -DCUDA_AVAILABLE
    $(info CUDA found - enabling CUDA support)
    
    ifeq ($(OPENCV_CUDA_AVAILABLE), yes)
        CXXFLAGS += -DOPENCV_CUDA_AVAILABLE
        $(info OpenCV with CUDA support found)
    else
        $(info OpenCV without CUDA support - CUDA OpenCV features disabled)
    endif
else
    CUDA_LIBS =
    $(info CUDA not found - compiling without CUDA support)
endif

# All source files
SOURCES = main.cpp \
			ZeroMQ/Publisher.cpp \
			ZeroMQ/Subscriber.cpp \
			car_controls/sources/EngineController.cpp \
			car_controls/sources/ControlsManager.cpp \
			car_controls/sources/PeripheralController.cpp \
			car_controls/sources/JoysticksController.cpp \
			car_controls/sources/MPCPlanner.cpp \
			car_controls/sources/Polyfitter.cpp \
			car_controls/sources/inference/LaneCurveFitter.cpp \
			car_controls/sources/inference/KerasInferencer.cpp \
			car_controls/sources/inference/InferenceManager.cpp \
			car_controls/sources/inference/TensorRTInferencer.cpp \
			car_controls/sources/inference/CameraStreamer.cpp \
			car_controls/sources/inference/LanePostProcessor.cpp \
			car_controls/sources/inference/ONNXInferencer.cpp \
			car_controls/sources/MPCOptimizer.cpp \
			car_controls/sources/objectDetection/LabelManager.cpp \
			car_controls/sources/objectDetection/YOLOv5TRT.cpp

# Remove duplicates from SOURCES
SOURCES := $(sort $(SOURCES))
OBJECTS = $(SOURCES:.cpp=.o)

# MOC files for Qt classes
MOC_FILES = main.moc \
            car_controls/sources/ControlsManager.moc \
            car_controls/sources/EngineController.moc \
            car_controls/sources/JoysticksController.moc

# Libraries - different for cross-compilation
ifneq ($(CXX),aarch64-linux-gnu-g++)
    # Native compilation
    OPENCV_LIBS = $(shell pkg-config --libs opencv4 || pkg-config --libs opencv)
else
    # Cross-compilation - use sysroot OpenCV
    OPENCV_LIBS = -L/home/michel/new_qtjetson/sysroot/usr/lib/aarch64-linux-gnu \
                  -lopencv_stitching -lopencv_alphamat -lopencv_aruco -lopencv_barcode \
                  -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect \
                  -lopencv_dnn_superres -lopencv_dpm -lopencv_face -lopencv_freetype \
                  -lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash \
                  -lopencv_intensity_transform -lopencv_line_descriptor -lopencv_mcc \
                  -lopencv_quality -lopencv_rapid -lopencv_reg -lopencv_rgbd \
                  -lopencv_saliency -lopencv_shape -lopencv_stereo -lopencv_structured_light \
                  -lopencv_phase_unwrapping -lopencv_superres -lopencv_optflow \
                  -lopencv_surface_matching -lopencv_tracking -lopencv_highgui \
                  -lopencv_datasets -lopencv_text -lopencv_plot -lopencv_ml \
                  -lopencv_videostab -lopencv_videoio -lopencv_viz -lopencv_wechat_qrcode \
                  -lopencv_ximgproc -lopencv_video -lopencv_xobjdetect -lopencv_objdetect \
                  -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_dnn \
                  -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core \
                  -lopencv_cudaimgproc -lopencv_cudaarithm -lopencv_cudawarping
endif

MLPACK_LIBS = -lmlpack -larmadillo -llapack -lblas
ZMQ_LIBS = -lzmq

# Filesystem library - different for cross-compilation
ifneq ($(CXX),aarch64-linux-gnu-g++)
    FILESYSTEM_LIBS = -lstdc++fs
else
    # Use sysroot's filesystem library to avoid glibc version conflicts
    FILESYSTEM_LIBS = -L/home/michel/new_qtjetson/sysroot/usr/lib/gcc/aarch64-linux-gnu/9 -lstdc++fs
endif

ifeq ($(CUDA_AVAILABLE), yes)
	ifneq ($(wildcard /usr/local/lib/libnvinfer.so /usr/lib/aarch64-linux-gnu/libnvinfer.so /home/michel/new_qtjetson/sysroot/usr/lib/aarch64-linux-gnu/libnvinfer.so),)
        TENSORRT_LIBS = -L/home/michel/new_qtjetson/sysroot/usr/lib/aarch64-linux-gnu -lnvinfer -lnvinfer_plugin -lnvonnxparser
        CXXFLAGS += -DTENSORRT_AVAILABLE
        $(info TensorRT found - enabling TensorRT support)
    else
        TENSORRT_LIBS =
        $(info TensorRT not found - some features may be limited)
    endif
else
    TENSORRT_LIBS =
endif

ALL_LIBS = -lnlopt -pthread $(CUDA_LIBS) $(MLPACK_LIBS) $(FILESYSTEM_LIBS) $(ZMQ_LIBS) $(TENSORRT_LIBS) $(OPENCV_LIBS) $(QT5_LIBS) -lSDL2

# Target
TARGET = main

all: $(TARGET)

# MOC processing - different for cross-compilation
ifneq ($(CXX),aarch64-linux-gnu-g++)
    # Native compilation
    MOC = $(shell pkg-config --variable=host_bins Qt5Core)/moc
else
    # Cross-compilation - use sysroot MOC via qemu to match Qt version
    MOC = qemu-aarch64-static -L /home/michel/new_qtjetson/sysroot /home/michel/new_qtjetson/sysroot/usr/lib/qt5/bin/moc
endif

%.moc: %.cpp
	$(MOC) $< -o $@

car_controls/sources/%.moc: car_controls/includes/%.hpp
	$(MOC) $< -o $@

# Main target
$(TARGET): $(OBJECTS) $(MOC_FILES)
	$(CXX) $(CXXFLAGS) $(QT5_CFLAGS) $(OBJECTS) -o $@ $(LDFLAGS) $(ALL_LIBS)

# Compilation rules with MOC dependencies
main.o: main.cpp main.moc
	$(CXX) $(CXXFLAGS) $(QT5_CFLAGS) $(INCLUDE_PATHS) -c $< -o $@

car_controls/sources/ControlsManager.o: car_controls/sources/ControlsManager.cpp car_controls/sources/ControlsManager.moc
	$(CXX) $(CXXFLAGS) $(QT5_CFLAGS) $(INCLUDE_PATHS) -c $< -o $@

car_controls/sources/EngineController.o: car_controls/sources/EngineController.cpp car_controls/sources/EngineController.moc
	$(CXX) $(CXXFLAGS) $(QT5_CFLAGS) $(INCLUDE_PATHS) -c $< -o $@

car_controls/sources/JoysticksController.o: car_controls/sources/JoysticksController.cpp car_controls/sources/JoysticksController.moc
	$(CXX) $(CXXFLAGS) $(QT5_CFLAGS) $(INCLUDE_PATHS) -c $< -o $@

# Generic rule for other cpp files that don't need MOC
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(QT5_CFLAGS) $(INCLUDE_PATHS) -c $< -o $@

# Lane detection video test (mínimo, sem Qt)
lane_detection_video_test: lane_detection_video_test.cpp \
	car_controls/sources/inference/TensorRTInferencer.cpp \
	car_controls/sources/inference/LanePostProcessor.cpp \
	car_controls/sources/inference/LaneCurveFitter.cpp \
	car_controls/sources/inference/ONNXInferencer.cpp \
	ZeroMQ/Publisher.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATHS) $^ -o $@ $(LDFLAGS) $(ALL_LIBS)

# Clean
clean: # remove gcno recursivally
	rm -f $(OBJECTS) $(MOC_FILES) $(TARGET) */**.gcno */*/*.gcno */*/*/*.gcno

install-deps:
	sudo apt update
	sudo apt install -y libopencv-dev libeigen3-dev libnlopt-dev libmlpack-dev libarmadillo-dev libsdl2-dev qtbase5-dev qtbase5-dev-tools libzmq3-dev

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

# Check Qt5 installation
check-qt:
	@echo "=== Checking Qt5 Installation ==="
	@pkg-config --exists Qt5Core && echo "Qt5Core: OK" || echo "Qt5Core: NOT FOUND"
	@pkg-config --exists Qt5Gui && echo "Qt5Gui: OK" || echo "Qt5Gui: NOT FOUND"
	@pkg-config --exists Qt5Widgets && echo "Qt5Widgets: OK" || echo "Qt5Widgets: NOT FOUND"
	@echo "Qt5 MOC path: $(shell pkg-config --variable=host_bins Qt5Core 2>/dev/null || echo 'NOT FOUND')"
	@echo "Qt5 CFLAGS: $(QT5_CFLAGS)"
	@echo "Qt5 LIBS: $(QT5_LIBS)"

# Check library availability
check-libs:
	@echo "=== Checking Library Availability ==="
	@echo "OpenCV:"
	@pkg-config --exists opencv4 && echo "  opencv4: OK" || (pkg-config --exists opencv && echo "  opencv: OK" || echo "  opencv: NOT FOUND")
	@echo "OpenCV CUDA support:"
	@pkg-config --exists opencv4 && pkg-config --cflags opencv4 | grep -q cuda && echo "  OpenCV CUDA: OK" || echo "  OpenCV CUDA: NOT FOUND"
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
	@echo "QT5_CFLAGS: $(QT5_CFLAGS)"
	@echo "CUDA_AVAILABLE: $(CUDA_AVAILABLE)"
	@echo "CUDA_LIBS: $(CUDA_LIBS)"
	@echo "OPENCV_LIBS: $(OPENCV_LIBS)"
	@echo "MLPACK_LIBS: $(MLPACK_LIBS)"
	@echo "FILESYSTEM_LIBS: $(FILESYSTEM_LIBS)"
	@echo "BASIC_LIBS: $(BASIC_LIBS)"
	@echo "=== Combined flags for compilation ==="
	@echo "All include flags: $(QT5_CFLAGS) $(INCLUDE_PATHS)"
	@echo "=== Testing individual file compilation ==="
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATHS) -c car_controls/sources/Polyfitter.cpp -o /tmp/test_polyfitter.o -v
	@echo "Polyfitter compilation successful"
	@echo "No tests defined in this Makefile. Please add your test commands here."
# Show what flags are being used
.PHONY: all clean install-deps check-cuda check-libs debug-compile check-qt show-flags test lane_detection_video_test jetson

# Simple jetson cross-compilation target
jetson: clean
	@echo "=== Building for Jetson (ARM64) ==="
	@which aarch64-linux-gnu-g++ > /dev/null || (echo "❌ Cross-compiler not found! Install with: sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu" && exit 1)
	@test -d /home/michel/new_qtjetson/sysroot || (echo "❌ Sysroot not found at /home/michel/new_qtjetson/sysroot" && exit 1)
	@test -d /home/michel/new_qtjetson/sysroot/usr/local/cuda || (echo "❌ CUDA not found in sysroot. Run sync-jetson-sys.sh first" && exit 1)
	@echo "✅ Cross-compiler, sysroot, and CUDA found"
	$(MAKE) CXX=aarch64-linux-gnu-g++ \
		CXXFLAGS="-std=c++17 -Wall -Wextra -O2 -fopenmp -fPIC -DCUDA_AVAILABLE --sysroot=/home/michel/new_qtjetson/sysroot -D_GLIBCXX_USE_CXX11_ABI=1 -fno-stack-protector" \
		LDFLAGS="--sysroot=/home/michel/new_qtjetson/sysroot -Wl,--allow-shlib-undefined -static-libgcc -static-libstdc++" \
		INCLUDE_PATHS="-I. -IZeroMQ -Icar_controls/includes -Icar_controls/includes/inference -Icar_controls/includes/objectDetection -I/home/michel/new_qtjetson/sysroot/usr/include/eigen3 -I/home/michel/new_qtjetson/sysroot/usr/include/opencv4 -I/home/michel/new_qtjetson/sysroot/usr/local/cuda/include -I/home/michel/new_qtjetson/sysroot/usr/local/include" \
		CUDA_PATH="/home/michel/new_qtjetson/sysroot/usr/local/cuda" \
		CUDA_INCLUDE="/home/michel/new_qtjetson/sysroot/usr/local/cuda/include" \
		CUDA_LIB="/home/michel/new_qtjetson/sysroot/usr/local/cuda/lib64" \
		CUDA_AVAILABLE=yes \
		MLPACK_LIBS="-lmlpack -larmadillo -L/home/michel/new_qtjetson/sysroot/usr/lib/aarch64-linux-gnu/atlas -llapack -lblas" \
		all
	@echo "✅ Jetson build complete! Binary: main (ARM64)"
	@file main | grep -q ARM && echo "✅ Confirmed ARM64 binary" || echo "⚠️  Warning: Binary architecture verification failed"
