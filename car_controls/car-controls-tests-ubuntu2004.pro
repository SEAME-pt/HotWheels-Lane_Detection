QT       += core testlib
CONFIG   += c++17
TARGET   = car-controls-tests

# Include Paths
INCLUDEPATH += \
    $$PWD/includes \
    $$PWD/tests/mocks \
    $$PWD/sources \
    /usr/local/include/opencv4 \
    /usr/include/opencv4 \
    /usr/local/cuda/include \
    /usr/local/cuda-10.2/targets/aarch64-linux/include
    /usr/include/aarch64-linux-gnu

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
INCLUDEPATH += /usr/local/cuda-10.2/targets/aarch64-linux/include

# TensorRT includes
INCLUDEPATH += /usr/include/aarch64-linux-gnu

# OpenCV includes
INCLUDEPATH += /usr/local/include/opencv4
INCLUDEPATH += /usr/include/opencv4

# GStreamer includes
INCLUDEPATH += /usr/include/gstreamer-1.0
INCLUDEPATH += /usr/include/glib-2.0
INCLUDEPATH += /usr/lib/aarch64-linux-gnu/glib-2.0/include

# Link GTest and GMock
LIBS += -lgtest_main -lpthread -lgmock -lgtest -lzmq

# Library paths
LIBS += -L/usr/local/lib
LIBS += -L/usr/local/cuda-10.2/targets/aarch64-linux/lib
LIBS += -L/usr/lib/aarch64-linux-gnu/
LIBS += -L/usr/lib/aarch64-linux-gnu/tegra
LIBS += -L/usr/lib/aarch64-linux-gnu/openblas
LIBS += -L/usr/local/lib  # <- Add this for GLEW/GLFW libs

# Eigen libraries
INCLUDEPATH += /usr/include/eigen3

# TensorRT, CUDA, OpenCV
LIBS += -lcudart -lnvinfer
LIBS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_highgui -lopencv_calib3d
LIBS += -lopencv_cudaarithm -lopencv_cudawarping -lopencv_cudaimgproc -lopencv_cudacodec
LIBS += -lcublasLt -llapack -lblas
LIBS += -lnvmedia -lnvdla_compiler

# GStreamer libraries
LIBS += -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0

# OpenGL, GLEW, GLFW libraries (ORDER MATTERS!)
LIBS += -lGLEW -lglfw -lGL

# RPath for custom OpenCV runtime
QMAKE_LFLAGS += -Wl,-rpath-link,/usr/local/lib
QMAKE_LFLAGS += -Wl,-rpath-link,/usr/lib/aarch64-linux-gnu/tegra
