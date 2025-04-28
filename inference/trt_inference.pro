# Project configuration
TEMPLATE = app
TARGET = trt_inference
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt  # Remove Qt dependency since it's not needed

# Common source files for all architectures
HEADERS += TensorRTInferencer.hpp \
           CameraStreamer.hpp

SOURCES += TensorRTInferencer.cpp \
           CameraStreamer.cpp \
           mainForJetson.cpp  # Always include the Jetson main file for ARM builds

# Common configuration for all builds
QMAKE_CXXFLAGS += -std=c++14

# Platform-specific configuration
contains(QT_ARCH, arm)|contains(QT_ARCH, arm64)|contains(QT_ARCH, aarch64) {
    message("Building for ARM architecture")

    # Define paths for Jetson cross-compilation
    JETSON_SYSROOT = /home/seame/qtjetson/sysroot

    # CUDA includes
    INCLUDEPATH += $${JETSON_SYSROOT}/usr/local/cuda-10.2/targets/aarch64-linux/include

    # TensorRT includes
    INCLUDEPATH += $${JETSON_SYSROOT}/usr/include/aarch64-linux-gnu

    # Prioritize the custom OpenCV include path
    INCLUDEPATH += $${JETSON_SYSROOT}/usr/local/include/opencv4
    # Add system OpenCV as fallback
    INCLUDEPATH += $${JETSON_SYSROOT}/usr/include/opencv4

    # GStreamer includes
    INCLUDEPATH += $${JETSON_SYSROOT}/usr/include/gstreamer-1.0
    INCLUDEPATH += $${JETSON_SYSROOT}/usr/include/glib-2.0
    INCLUDEPATH += $${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/glib-2.0/include

    # Library paths - list custom OpenCV path FIRST
    LIBS += -L$${JETSON_SYSROOT}/usr/local/lib  # Custom OpenCV library path first
    LIBS += -L$${JETSON_SYSROOT}/usr/local/cuda-10.2/targets/aarch64-linux/lib
    LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu
    LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra
    LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/openblas

    # Link against specific 4.0.5 versions
    LIBS += -lcudart -lnvinfer
    LIBS += -l:libopencv_core.so.405 -l:libopencv_imgproc.so.405 -l:libopencv_imgcodecs.so.405 -l:libopencv_videoio.so.405 -l:libopencv_highgui.so.405 -l:libopencv_calib3d.so.405
    LIBS += -l:libopencv_cudaarithm.so.405 -l:libopencv_cudawarping.so.405 -l:libopencv_cudaimgproc.so.405
    LIBS += -lcublasLt -llapack -lblas
    LIBS += -lnvmedia -lnvdla_compiler

    # GStreamer libraries
    LIBS += -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0

    # Add rpath to find custom OpenCV at runtime
    QMAKE_LFLAGS += -Wl,-rpath-link,$${JETSON_SYSROOT}/usr/local/lib
    QMAKE_LFLAGS += -Wl,-rpath-link,$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra

} else {
    message("Building for x86 architecture")

    # Standard include paths for x86
    INCLUDEPATH += /usr/local/cuda/include
    INCLUDEPATH += /usr/include
    INCLUDEPATH += /usr/include/opencv4

    # GStreamer includes for x86
    INCLUDEPATH += /usr/include/gstreamer-1.0
    INCLUDEPATH += /usr/include/glib-2.0
    INCLUDEPATH += /usr/lib/x86_64-linux-gnu/glib-2.0/include

    # Include CUDA modules of OpenCV (this is usually where cudaimgproc.hpp lives)
    INCLUDEPATH += $${JETSON_SYSROOT}/usr/include/opencv4/opencv2

    # Library paths for x86
    LIBS += -L/usr/local/cuda/lib64
    LIBS += -L/usr/lib
    LIBS += -L/usr/lib/x86_64-linux-gnu
    LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/openblas

    # Libraries for x86
    LIBS += -lcudart -lnvinfer -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_highgui -lopencv_calib3d
    LIBS += -lopencv_cudaarithm -lopencv_cudawarping -lopencv_cudaimgproc -lcublasLt -llapack -lblas

    # GStreamer libraries
    LIBS += -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0
}

# Deployment - copy model and configuration files
!isEmpty(target.path): INSTALLS += target
