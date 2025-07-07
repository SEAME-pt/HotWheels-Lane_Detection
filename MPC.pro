# Configuração para cross-compilation para NVIDIA Jetson Nano (ARM64)
contains(QT_ARCH, arm)|contains(QT_ARCH, arm64)|contains(QT_ARCH, aarch64) {
    # Configurar sysroot
    QMAKE_SYSROOT = /home/michel/new_qtjetson/sysroot
}

QT += core gui widgets
CONFIG += c++17 cmdline

# Target configuration
TARGET = main
TEMPLATE = app

# Include Paths (baseado no Makefile)
INCLUDEPATH += \
    $$PWD \
    $$PWD/ZeroMQ \
    $$PWD/car_controls/includes \
    $$PWD/car_controls/includes/inference \
    $$PWD/car_controls/includes/objectDetection \
    $${QMAKE_SYSROOT}/usr/local/include \
    $${QMAKE_SYSROOT}/usr/include \
    $${QMAKE_SYSROOT}/usr/include/eigen3 \
    $${QMAKE_SYSROOT}/usr/include/opencv4 \
    $${QMAKE_SYSROOT}/usr/include/opencv4/opencv2 \
    $${QMAKE_SYSROOT}/usr/local/cuda/include \
    $${QMAKE_SYSROOT}/usr/include/aarch64-linux-gnu \
    /home/michel/new_qtjetson/qt5.15/include \
    /home/michel/new_qtjetson/qt5.15/include/QtCore \
    /home/michel/new_qtjetson/qt5.15/include/QtGui \
    /home/michel/new_qtjetson/qt5.15/include/QtWidgets

# Application Sources (baseado em SOURCES do Makefile)
SOURCES += \
    main.cpp \
    ZeroMQ/Publisher.cpp \
    ZeroMQ/Subscriber.cpp \
    car_controls/sources/Debugger.cpp \
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
    car_controls/sources/inference/PolyfitterInferencer.cpp \
    car_controls/sources/inference/CameraStreamer.cpp \
    car_controls/sources/inference/LanePostProcessor.cpp \
    car_controls/sources/inference/ONNXInferencer.cpp \
    car_controls/sources/MPCOptimizer.cpp \
    car_controls/sources/objectDetection/LabelManager.cpp \
    car_controls/sources/objectDetection/YOLOv5TRT.cpp

# Application Headers (baseado no car-controls-qt.pro)
HEADERS += \
    ZeroMQ/Publisher.hpp \
    ZeroMQ/Subscriber.hpp \
    car_controls/includes/enums.hpp \
    car_controls/includes/Debugger.hpp \
    car_controls/includes/MPCConfig.hpp \
    car_controls/includes/MPCPlanner.hpp \
    car_controls/includes/Polyfitter.hpp \
    car_controls/includes/CommonTypes.hpp \
    car_controls/includes/MPCOptimizer.hpp \
    car_controls/includes/ControlsManager.hpp \
    car_controls/includes/EngineController.hpp \
    car_controls/includes/JoysticksController.hpp \
    car_controls/includes/PeripheralController.hpp \
    car_controls/includes/inference/IInferencer.hpp \
    car_controls/includes/IPeripheralController.hpp \
    car_controls/includes/inference/CameraStreamer.hpp \
    car_controls/includes/inference/KerasInferencer.hpp \
    car_controls/includes/inference/LaneCurveFitter.hpp \
    car_controls/includes/objectDetection/YOLOv5TRT.hpp \
    car_controls/includes/inference/ONNXInferencer.hpp \
    car_controls/includes/inference/InferenceManager.hpp \
    car_controls/includes/inference/LanePostProcessor.hpp \
    car_controls/includes/inference/TensorRTInferencer.hpp \
    car_controls/includes/objectDetection/LabelManager.hpp \
    car_controls/includes/inference/PolyfitterInferencer.hpp

# Compilation flags (otimizações do Makefile)
QMAKE_CXXFLAGS += -std=c++17 -Wall -Wextra -O3 -fopenmp -fPIC \
                  -march=armv8-a -mcpu=cortex-a57 -mtune=cortex-a57 \
                  -DCUDA_AVAILABLE -DJETSON_NANO
QMAKE_CXXFLAGS +=  -g -fdump-rtl-expand 
QMAKE_CFLAGS += -fopenmp

# Library paths (baseado no Makefile)
contains(QT_ARCH, arm)|contains(QT_ARCH, arm64)|contains(QT_ARCH, aarch64) {
    LIBS += -L$${QMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu \
            -L$${QMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra \
            -L$${QMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra-egl \
            -L$${QMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/openblas-pthread \
            -L$${QMAKE_SYSROOT}/usr/lib/gcc/aarch64-linux-gnu/9 \
            -L$${QMAKE_SYSROOT}/usr/local/cuda/lib64 \
            -L$${QMAKE_SYSROOT}/usr/local/lib \
            -L$${QMAKE_SYSROOT}/lib/aarch64-linux-gnu \
            -L/home/michel/new_qtjetson/qt5.15/lib

    # Bibliotecas OpenCV com extensões CUDA (do Makefile)
    LIBS += -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio \
            -lopencv_cudaimgproc -lopencv_cudawarping -lopencv_cudaarithm \
            -lopencv_cudafilters -lopencv_cudacodec -lopencv_cudafeatures2d \
            -lopencv_cudalegacy -lopencv_cudastereo -lopencv_cudabgsegm \
            -lopencv_imgcodecs -lopencv_video -lopencv_objdetect -lopencv_dnn \
            -lopencv_ml -lopencv_features2d -lopencv_flann -lopencv_calib3d -lopencv_photo

    # Bibliotecas matemáticas e outras (do Makefile)
    LIBS += -lmlpack -larmadillo -lopenblasp-r0.3.8 -lcblas -lf77blas -llapack_atlas -lgfortran -lm \
            -lstdc++fs -lzmq -lnvinfer -lnvinfer_plugin -lnvonnxparser \
            -lQt5Core -lQt5Gui -lQt5Widgets \
            -lcuda -lcudart -lcurand -lcufft -lcublas -lcublasLt -lcusolver -lcusparse \
            -lSDL2 -lnlopt -pthread -ldl -lrt
}

# RPath e flags de linking (baseado no Makefile)
contains(QT_ARCH, arm)|contains(QT_ARCH, arm64)|contains(QT_ARCH, aarch64) {
    QMAKE_LFLAGS += --sysroot=$${QMAKE_SYSROOT} \
                    -Wl,-rpath-link,$${QMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu \
                    -Wl,-rpath-link,$${QMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra \
                    -Wl,-rpath-link,$${QMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/openblas-pthread \
                    -Wl,-rpath-link,$${QMAKE_SYSROOT}/lib/aarch64-linux-gnu \
                    -Wl,-rpath-link,/home/michel/new_qtjetson/qt5.15/lib \
                    -Wl,--as-needed -Wl,--gc-sections -Wl,-O1 \
                    -static-libstdc++ -static-libgcc
}
