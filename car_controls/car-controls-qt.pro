QT = core

CONFIG += c++17 cmdline

# Include Paths (explicit inheritance from root)
INCLUDEPATH += \
	$$PWD/includes \
	$$PWD/includes/inference


# Application Sources
SOURCES += \
	../../ZeroMQ/Publisher.cpp \
	../../ZeroMQ/Subscriber.cpp \
	sources/inference/CameraStreamer.cpp \
	sources/inference/TensorRTInferencer.cpp \
	sources/inference/LanePostProcessor.cpp \
	sources/inference/LaneCurveFitter.cpp \
	sources/objectDetection/LabelManager.cpp \
	sources/objectDetection/YOLOv5TRT.cpp \
	sources/ControlsManager.cpp \
	sources/JoysticksController.cpp \
	sources/EngineController.cpp \
	sources/PeripheralController.cpp \
	sources/main.cpp

HEADERS += \
	../../ZeroMQ/Publisher.hpp \
	../../ZeroMQ/Subscriber.hpp \
	includes/inference/CameraStreamer.hpp \
	includes/inference/TensorRTInferencer.hpp \
	includes/inference/IInferencer.hpp \
	includes/inference/LanePostProcessor.hpp \
	includes/inference/LaneCurveFitter.hpp \
	includes/objectDetection/LabelManager.hpp \
	includes/objectDetection/YOLOv5TRT.hpp \
	includes/ControlsManager.hpp \
	includes/JoysticksController.hpp \
	includes/EngineController.hpp \
	includes/PeripheralController.hpp \
	includes/IPeripheralController.hpp \
	includes/enums.hpp

# Common Libraries
LIBS += -lSDL2 -lrt -lzmq

# Conditionally add paths for cross-compilation
contains(QT_ARCH, arm)|contains(QT_ARCH, arm64)|contains(QT_ARCH, aarch64) {
	LIBS += -L$$[QT_SYSROOT]/usr/lib/aarch64-linux-gnu -lSDL2
	INCLUDEPATH += $$[QT_SYSROOT]/usr/include/SDL2

	message("Building for ARM architecture")

	JETSON_SYSROOT = /home/seame/qtjetson/sysroot

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

	# OpenGL, GLFW, GLEW includes
	INCLUDEPATH += /usr/local/include
	INCLUDEPATH += /usr/include/GL
	INCLUDEPATH += /usr/include/GLFW

	# Library paths
	LIBS += -L$${JETSON_SYSROOT}/usr/local/lib
	LIBS += -L$${JETSON_SYSROOT}/usr/local/cuda-10.2/targets/aarch64-linux/lib
	LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu
	LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra
	LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/openblas

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
}
