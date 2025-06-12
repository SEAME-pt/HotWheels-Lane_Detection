QT       += core testlib
CONFIG   += c++17
TARGET   = car-controls-tests

JETSON_SYSROOT = /home/seame/qtjetson/sysroot

# Include Paths
INCLUDEPATH += \
    $$PWD/includes \
    $$PWD/tests/mocks \
    $$PWD/sources \
    $${JETSON_SYSROOT}/usr/local/include/opencv4 \
    $${JETSON_SYSROOT}/usr/include/opencv4 \
    $${JETSON_SYSROOT}/usr/local/cuda/include \
    $${JETSON_SYSROOT}/usr/local/cuda-10.2/targets/aarch64-linux/include
    $${JETSON_SYSROOT}/usr/include/aarch64-linux-gnu

# Test Sources
TESTS_PATH = tests

SOURCES += \
    $$TESTS_PATH/unit/test_PeripheralController.cpp \
    $$TESTS_PATH/unit/test_TensorRTInferencer.cpp \
    $$TESTS_PATH/unit/test_CameraStreamer.cpp \
    $$TESTS_PATH/unit/test_LabelManager.cpp \
    $$TESTS_PATH/unit/test_YOLOv5TRT.cpp \
    ../../ZeroMQ/Publisher.cpp \
    ../../ZeroMQ/Subscriber.cpp \
    sources/PeripheralController.cpp \
    sources/inference/CameraStreamer.cpp \
    sources/inference/TensorRTInferencer.cpp \
    sources/inference/LanePostProcessor.cpp \
    sources/inference/LaneCurveFitter.cpp \
    sources/objectDetection/LabelManager.cpp \
    sources/objectDetection/YOLOv5TRT.cpp

HEADERS += \
    $$TESTS_PATH/mocks/MockPeripheralController.hpp \
    $$TESTS_PATH/mocks/MockInferencer.hpp \
    ../../ZeroMQ/Publisher.hpp \
    ../../ZeroMQ/Subscriber.hpp \
    includes/inference/CameraStreamer.hpp \
    includes/inference/TensorRTInferencer.hpp \
    includes/inference/LanePostProcessor.hpp \
	includes/inference/LaneCurveFitter.hpp \
    includes/inference/IInferencer.hpp \
    includes/objectDetection/LabelManager.hpp \
    includes/objectDetection/YOLOv5TRT.hpp

# CUDA includes
INCLUDEPATH += $${JETSON_SYSROOT}/usr/local/cuda-10.2/targets/aarch64-linux/include

# TensorRT includes
INCLUDEPATH += $${JETSON_SYSROOT}/usr/include/aarch64-linux-gnu

# OpenCV includes
INCLUDEPATH += $${JETSON_SYSROOT}/usr/local/include/opencv4
INCLUDEPATH += $${JETSON_SYSROOT}/usr/include/opencv4

# GStreamer includes
INCLUDEPATH += $${JETSON_SYSROOT}/usr/include/gstreamer-1.0
INCLUDEPATH += $${JETSON_SYSROOT}/usr/include/glib-2.0
INCLUDEPATH += $${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/glib-2.0/include

# Link GTest and GMock
LIBS += -lgtest_main -lpthread -lgmock -lgtest -lzmq

# Library paths
LIBS += -L$${JETSON_SYSROOT}/usr/local/lib
LIBS += -L$${JETSON_SYSROOT}/usr/local/cuda-10.2/targets/aarch64-linux/lib
LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/
LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra
LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/openblas
LIBS += -L/usr/local/lib  # <- Add this for GLEW/GLFW libs

# Eigen libraries
INCLUDEPATH += $${JETSON_SYSROOT}/usr/include/eigen3

# TensorRT, CUDA, OpenCV
LIBS += -lcudart -lnvinfer
LIBS += -l:libopencv_core.so.405 -l:libopencv_imgproc.so.405 -l:libopencv_imgcodecs.so.405 -l:libopencv_videoio.so.405 -l:libopencv_highgui.so.405 -l:libopencv_calib3d.so.405
LIBS += -l:libopencv_cudaarithm.so.405 -l:libopencv_cudawarping.so.405 -l:libopencv_cudaimgproc.so.405 -l:libopencv_cudacodec.so.405
LIBS += -lcublasLt -llapack -lblas
LIBS += -lnvmedia -lnvdla_compiler

# GStreamer libraries
LIBS += -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0

# OpenGL, GLEW, GLFW libraries (ORDER MATTERS!)
LIBS += -lGLEW -lglfw -lGL

# RPath for custom OpenCV runtime
QMAKE_LFLAGS += -Wl,-rpath-link,$${JETSON_SYSROOT}/usr/local/lib
QMAKE_LFLAGS += -Wl,-rpath-link,$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra
