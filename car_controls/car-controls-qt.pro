# Configuração para cross-compilation
contains(QT_ARCH, arm)|contains(QT_ARCH, arm64)|contains(QT_ARCH, aarch64) {
    # Configurar sysroot
    QMAKE_SYSROOT = /home/michel/new_qtjetson/sysroot
    
    # Configurar compiladores (usando GCC 9 compatível com Jetson)
    QMAKE_CC = aarch64-linux-gnu-gcc-9
    QMAKE_CXX = aarch64-linux-gnu-g++-9
    QMAKE_LINK = aarch64-linux-gnu-g++-9
    QMAKE_AR = aarch64-linux-gnu-ar
    QMAKE_STRIP = aarch64-linux-gnu-strip
}

QT = core

CONFIG += c++17 cmdline

# Target configuration
TARGET = car-controls-qt
TEMPLATE = app

# Include Paths (explicit inheritance from root)
INCLUDEPATH += \
	$$PWD/includes \
	$$PWD/includes/inference \
	$$PWD/includes/objectDetection

#include ZeroMQ

INCLUDEPATH += $$PWD/../ZeroMQ

# Eigen (header-only)
INCLUDEPATH += /usr/include/eigen3

# OpenCV includes (for host compilation)
INCLUDEPATH += /usr/include/opencv4


# Application Sources
SOURCES += \
	../ZeroMQ/Publisher.cpp \
	../ZeroMQ/Subscriber.cpp \
	sources/inference/CameraStreamer.cpp \
	# sources/inference/ONNXInferencer.cpp \
	sources/inference/TensorRTInferencer.cpp \
	sources/inference/KerasInferencer.cpp \
	sources/inference/InferenceManager.cpp \
	sources/inference/LanePostProcessor.cpp \
	sources/inference/LaneCurveFitter.cpp \
	sources/inference/PolyfitterInferencer.cpp \
	sources/objectDetection/LabelManager.cpp \
	sources/objectDetection/YOLOv5TRT.cpp \
	sources/ControlsManager.cpp \
	sources/JoysticksController.cpp \
	sources/EngineController.cpp \
	sources/PeripheralController.cpp \
	sources/Debugger.cpp \
	sources/main.cpp

SOURCES += \
    sources/MPCOptimizer.cpp \
    sources/MPCPlanner.cpp \
    sources/Polyfitter.cpp

HEADERS += \
	../ZeroMQ/Publisher.hpp \
	../ZeroMQ/Subscriber.hpp \
	includes/inference/CameraStreamer.hpp \
	includes/inference/TensorRTInferencer.hpp \
	# includes/inference/ONNXInferencer.hpp \
	includes/inference/KerasInferencer.hpp \
	includes/inference/InferenceManager.hpp \
	includes/inference/IInferencer.hpp \
	includes/inference/LanePostProcessor.hpp \
	includes/inference/LaneCurveFitter.hpp \
	includes/inference/PolyfitterInferencer.hpp \
	includes/objectDetection/LabelManager.hpp \
	includes/objectDetection/YOLOv5TRT.hpp \
	includes/ControlsManager.hpp \
	includes/JoysticksController.hpp \
	includes/EngineController.hpp \
	includes/PeripheralController.hpp \
	includes/IPeripheralController.hpp \
	includes/Debugger.hpp \
	includes/enums.hpp

HEADERS += \
    includes/CommonTypes.hpp \
    includes/MPCConfig.hpp \
    includes/MPCOptimizer.hpp \
    includes/MPCPlanner.hpp \
    includes/Polyfitter.hpp

# Common Libraries
LIBS += -lSDL2 -lrt -lzmq
# Dependências adicionais
LIBS += -lnlopt -lmlpack
LIBS += -lboost_system -lstdc++fs

# Conditionally add paths for cross-compilation
contains(QT_ARCH, arm)|contains(QT_ARCH, arm64)|contains(QT_ARCH, aarch64) {
	LIBS += -L$$[QT_SYSROOT]/usr/lib/aarch64-linux-gnu -lSDL2
	INCLUDEPATH += $$[QT_SYSROOT]/usr/include/SDL2

	message("Building for ARM architecture")

	JETSON_SYSROOT = /home/michel/new_qtjetson/sysroot

	INCLUDEPATH += $${JETSON_SYSROOT}/usr/include

	# CUDA includes
	INCLUDEPATH += $${JETSON_SYSROOT}/usr/local/cuda-10.2/targets/aarch64-linux/include
	INCLUDEPATH += $${JETSON_SYSROOT}/usr/local/cuda/include
    INCLUDEPATH += $${JETSON_SYSROOT}/usr/local/cuda-10.2/include
    INCLUDEPATH += $${JETSON_SYSROOT}/usr/local/cuda-11.4/include
	
	# TensorRT includes
	INCLUDEPATH += $${JETSON_SYSROOT}/usr/include/aarch64-linux-gnu
	INCLUDEPATH += $${JETSON_SYSROOT}/usr/include/x86_64-linux-gnu
    INCLUDEPATH += $${JETSON_SYSROOT}/usr/include/aarch64-linux-gnu
    INCLUDEPATH += $${JETSON_SYSROOT}/usr/local/include

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
	LIBS += -L$${JETSON_SYSROOT}/usr/local/cuda-10.2/lib64
	LIBS += -L$${JETSON_SYSROOT}/usr/local/cuda-10.2/targets/aarch64-linux/lib
	LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra
	LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/openblas

	# Eigen libraries
	INCLUDEPATH += $${JETSON_SYSROOT}/usr/include/eigen3

	# TensorRT, CUDA, OpenCV
	LIBS += -lcudart -lnvinfer
	LIBS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_highgui -lopencv_calib3d
	LIBS += -lopencv_dnn -lopencv_cudaarithm -lopencv_cudawarping -lopencv_cudaimgproc -lopencv_cudacodec
	LIBS += -lcublasLt -llapack -lblas
	LIBS += -lnvmedia -lnvdla_compiler
	
	# OpenMP from sysroot to avoid GLIBC version conflicts
	LIBS += -L$${JETSON_SYSROOT}/usr/lib/gcc/aarch64-linux-gnu/9
	LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu
	LIBS += $${JETSON_SYSROOT}/usr/lib/gcc/aarch64-linux-gnu/9/libgomp.a

	# GStreamer libraries
	LIBS += -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0

	# OpenGL, GLEW, GLFW libraries (ORDER MATTERS!)
	LIBS += -lGLEW -lglfw -lGL

	# ZeroMQ includes
    INCLUDEPATH += $${JETSON_SYSROOT}/usr/include
    INCLUDEPATH += $${JETSON_SYSROOT}/usr/local/include

	# RPath for custom OpenCV runtime
	QMAKE_LFLAGS += -Wl,-rpath-link,$${JETSON_SYSROOT}/usr/local/lib
	QMAKE_LFLAGS += -Wl,-rpath-link,$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu
	QMAKE_LFLAGS += -Wl,-rpath-link,$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra
	QMAKE_LFLAGS += -Wl,-rpath-link,$${JETSON_SYSROOT}/usr/local/cuda-10.2/lib64
	QMAKE_LFLAGS += -Wl,-rpath-link,$${JETSON_SYSROOT}/usr/lib/gcc/aarch64-linux-gnu/9
	
	# Force using sysroot libraries for glibc compatibility
	QMAKE_LFLAGS += -Wl,-rpath-link,$${JETSON_SYSROOT}/lib/aarch64-linux-gnu
	QMAKE_LFLAGS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu
	QMAKE_LFLAGS += -L$${JETSON_SYSROOT}/lib/aarch64-linux-gnu
	
	# Static link with compatible libstdc++ to avoid glibc version conflicts
	QMAKE_LFLAGS += -static-libstdc++ -static-libgcc
}

# Adicionando flags de compilação para warnings e erros
# QMAKE_CXXFLAGS += -Wall -Werror -Wextra -pedantic

# Debug flags para análise de segfaults
QMAKE_CXXFLAGS += -g
QMAKE_CFLAGS += -g

# OpenMP support
QMAKE_CXXFLAGS += -fopenmp
QMAKE_CFLAGS += -fopenmp
