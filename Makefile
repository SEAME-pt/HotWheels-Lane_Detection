# Makefile.jetson - Cross-compilation MÁXIMA POTÊNCIA para Jetson Nano (ARM64)
# Versão ultra-otimizada com todas as dependências e funcionalidades habilitadas

# Exporta variáveis para garantir que pkg-config use apenas o sysroot
export PKG_CONFIG_SYSROOT_DIR := /home/michel/new_qtjetson/sysroot
export PKG_CONFIG_LIBDIR := /home/michel/new_qtjetson/sysroot/usr/lib/aarch64-linux-gnu/pkgconfig:/home/michel/new_qtjetson/sysroot/usr/lib/pkgconfig:/home/michel/new_qtjetson/sysroot/usr/share/pkgconfig
export PKG_CONFIG_PATH :=

CXX = aarch64-linux-gnu-g++-9
STRIP = aarch64-linux-gnu-strip

# Flags de compilação otimizadas para máxima performance
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -fopenmp -fPIC \
			-march=armv8-a -mcpu=cortex-a57 -mtune=cortex-a57 \
			--sysroot=/home/michel/new_qtjetson/sysroot \
			-DCUDA_AVAILABLE -DJETSON_NANO

# Flags de linking ultra-robustas compatíveis com GLIBC 2.31
LDFLAGS = --sysroot=/home/michel/new_qtjetson/sysroot \
			-L/home/michel/new_qtjetson/sysroot/usr/lib/aarch64-linux-gnu \
			-L/home/michel/new_qtjetson/sysroot/usr/lib/aarch64-linux-gnu/tegra \
			-L/home/michel/new_qtjetson/sysroot/usr/lib/aarch64-linux-gnu/tegra-egl \
			-L/home/michel/new_qtjetson/sysroot/usr/lib/aarch64-linux-gnu/openblas-pthread \
			-L/home/michel/new_qtjetson/sysroot/usr/lib/gcc/aarch64-linux-gnu/9 \
			-L/home/michel/new_qtjetson/sysroot/usr/local/cuda/lib64 \
			-L/home/michel/new_qtjetson/sysroot/usr/local/lib \
			-L/home/michel/new_qtjetson/sysroot/lib/aarch64-linux-gnu \
			-L/home/michel/new_qtjetson/qt5.15/lib \
			-Wl,-rpath-link,/home/michel/new_qtjetson/sysroot/usr/lib/aarch64-linux-gnu \
			-Wl,-rpath-link,/home/michel/new_qtjetson/sysroot/usr/lib/aarch64-linux-gnu/tegra \
			-Wl,-rpath-link,/home/michel/new_qtjetson/sysroot/usr/lib/aarch64-linux-gnu/openblas-pthread \
			-Wl,-rpath-link,/home/michel/new_qtjetson/sysroot/lib/aarch64-linux-gnu \
			-Wl,-rpath-link,/home/michel/new_qtjetson/qt5.15/lib \
			-Wl,--as-needed -Wl,--gc-sections -Wl,-O1

# Caminhos de includes ultra-completos para máxima compatibilidade
INCLUDE_PATHS = -I. \
				-IZeroMQ \
				-Icar_controls/includes \
				-Icar_controls/includes/inference \
				-Icar_controls/includes/objectDetection \
				-I/home/michel/new_qtjetson/sysroot/usr/local/include \
				-I/home/michel/new_qtjetson/sysroot/usr/include \
				-I/home/michel/new_qtjetson/sysroot/usr/include/eigen3 \
				-I/home/michel/new_qtjetson/sysroot/usr/include/opencv4 \
				-I/home/michel/new_qtjetson/sysroot/usr/include/opencv4/opencv2 \
				-I/home/michel/new_qtjetson/sysroot/usr/local/cuda/include \
				-I/home/michel/new_qtjetson/sysroot/usr/include/aarch64-linux-gnu \
				-I/home/michel/new_qtjetson/qt5.15/include \
				-I/home/michel/new_qtjetson/qt5.15/include/QtCore \
				-I/home/michel/new_qtjetson/qt5.15/include/QtGui \
				-I/home/michel/new_qtjetson/qt5.15/include/QtWidgets

# Bibliotecas OpenCV com todas as extensões CUDA habilitadas
OPENCV_LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio \
				-lopencv_cudaimgproc -lopencv_cudawarping -lopencv_cudaarithm \
				-lopencv_cudafilters -lopencv_cudacodec -lopencv_cudafeatures2d \
				-lopencv_cudalegacy -lopencv_cudastereo -lopencv_cudabgsegm \
				-lopencv_imgcodecs -lopencv_video -lopencv_objdetect -lopencv_dnn -lopencv_ml \
				-lopencv_features2d -lopencv_flann -lopencv_calib3d -lopencv_photo

# Bibliotecas matemáticas otimizadas (usando as disponíveis no sysroot)
MATH_LIBS = -lmlpack -larmadillo -lopenblasp-r0.3.8 -lcblas -lf77blas -llapack_atlas -lgfortran -lm
FILESYSTEM_LIBS = -lstdc++fs
ZMQ_LIBS = -lzmq
TENSORRT_LIBS = -lnvinfer -lnvinfer_plugin -lnvonnxparser
QT5_LIBS = -lQt5Core -lQt5Gui -lQt5Widgets
JETSON_LIBS = -lcuda -lcudart -lcurand -lcufft -lcublas -lcublasLt -lcusolver -lcusparse
MULTIMEDIA_LIBS = -lSDL2

# União de todas as bibliotecas essenciais em ordem de dependência otimizada
ALL_LIBS = -lnlopt -pthread $(MATH_LIBS) $(FILESYSTEM_LIBS) $(ZMQ_LIBS) $(TENSORRT_LIBS) \
			$(OPENCV_LIBS) $(QT5_LIBS) $(JETSON_LIBS) $(MULTIMEDIA_LIBS) \
			-ldl -lrt

# Todos os arquivos fonte do projeto para máxima funcionalidade
SOURCES = main.cpp \
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

# Arquivos fonte específicos para enhanced_mpc_main com máxima funcionalidade
ENHANCED_SOURCES = enhanced_mpc_main.cpp \
			ZeroMQ/Publisher.cpp \
			ZeroMQ/Subscriber.cpp \
			car_controls/sources/Debugger.cpp \
			car_controls/sources/EngineController.cpp \
			car_controls/sources/ControlsManager.cpp \
			car_controls/sources/PeripheralController.cpp \
			car_controls/sources/JoysticksController.cpp \
			car_controls/sources/MPCPlanner.cpp \
			car_controls/sources/Polyfitter.cpp \
			car_controls/sources/inference/TensorRTInferencer.cpp \
			car_controls/sources/inference/PolyfitterInferencer.cpp \
			car_controls/sources/inference/LanePostProcessor.cpp \
			car_controls/sources/inference/LaneCurveFitter.cpp \
			car_controls/sources/inference/CameraStreamer.cpp \
			car_controls/sources/inference/InferenceManager.cpp \
			car_controls/sources/inference/KerasInferencer.cpp \
			car_controls/sources/inference/ONNXInferencer.cpp \
			car_controls/sources/MPCOptimizer.cpp \
			car_controls/sources/objectDetection/YOLOv5TRT.cpp \
			car_controls/sources/objectDetection/LabelManager.cpp

SOURCES := $(sort $(SOURCES))
OBJECTS = $(SOURCES:.cpp=.o)

# Arquivos MOC necessários para Qt
MOC_FILES = main.moc \
            car_controls/sources/ControlsManager.moc \
            car_controls/sources/EngineController.moc \
            car_controls/sources/JoysticksController.moc

TARGET = main

# ================================
# REGRAS DE COMPILAÇÃO OTIMIZADAS
# ================================

all: $(TARGET)

# Geração dos arquivos MOC com Qt cross-compilado
%.moc: %.cpp
	/home/michel/new_qtjetson/qt5.15/bin/moc $< -o $@

car_controls/sources/%.moc: car_controls/includes/%.hpp
	/home/michel/new_qtjetson/qt5.15/bin/moc $< -o $@

# Target principal
$(TARGET): $(OBJECTS) $(MOC_FILES)
	@echo "=== Linking main target with maximum optimization ==="
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@ $(LDFLAGS) $(ALL_LIBS)
	$(STRIP) --strip-unneeded $@

# Regras específicas para arquivos com MOC
main.o: main.cpp main.moc
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATHS) -c $< -o $@

car_controls/sources/ControlsManager.o: car_controls/sources/ControlsManager.cpp car_controls/sources/ControlsManager.moc
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATHS) -c $< -o $@

car_controls/sources/EngineController.o: car_controls/sources/EngineController.cpp car_controls/sources/EngineController.moc
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATHS) -c $< -o $@

car_controls/sources/JoysticksController.o: car_controls/sources/JoysticksController.cpp car_controls/sources/JoysticksController.moc
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATHS) -c $< -o $@

# Regra genérica para outros arquivos
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATHS) -c $< -o $@

# ===============================================================
# ENHANCED MPC TARGET - MÁXIMA POTÊNCIA PARA JETSON NANO
# ===============================================================
enhanced_mpc_main: car_controls/sources/ControlsManager.moc car_controls/sources/EngineController.moc car_controls/sources/JoysticksController.moc $(ENHANCED_SOURCES)
	@echo "======================================================"
	@echo "🚀 CROSS-COMPILATION MÁXIMA POTÊNCIA PARA JETSON NANO"
	@echo "🎯 Target: enhanced_mpc_main (ARM64 ultra-otimizado)"
	@echo "🔧 Compiler: $(CXX) with ARM Cortex-A57 optimizations"
	@echo "📚 Libraries: Qt5 + CUDA + TensorRT + OpenCV + MPC"
	@echo "======================================================"
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATHS) $(filter %.cpp,$^) -o $@ $(LDFLAGS) $(ALL_LIBS)
	@echo ""
	@echo "🔍 Verification and optimization:"
	@file enhanced_mpc_main | grep -q ARM && echo "✅ Confirmed ARM64 binary" || echo "❌ ERROR: Not ARM64!"
	@echo -n "📁 Binary size: " && du -h enhanced_mpc_main | cut -f1
	@echo -n "🔗 Dynamic libraries: " && ldd enhanced_mpc_main 2>/dev/null | wc -l || echo "Cross-compiled (can't check on x86)"
	$(STRIP) --strip-unneeded enhanced_mpc_main
	@echo -n "📁 Stripped size: " && du -h enhanced_mpc_main | cut -f1
	@echo ""
	@echo "🎉 ENHANCED MPC CROSS-COMPILATION COMPLETE!"
	@echo "🚀 Ready to deploy on Jetson Nano!"
	@echo "======================================================"

# ===============================================================
# TARGETS AUXILIARES PARA DESENVOLVIMENTO
# ===============================================================

# Criação dos links simbólicos necessários para bibliotecas
fix-links:
	@echo "🔧 Fixing missing library symbolic links..."
	@mkdir -p /home/michel/new_qtjetson/sysroot/etc/alternatives
	@cd /home/michel/new_qtjetson/sysroot/usr/lib/aarch64-linux-gnu/openblas-pthread && \
	 ln -sf libopenblasp-r0.3.8.so libopenblas.so && \
	 ln -sf libopenblasp-r0.3.8.so libopenblas.so.0 && \
	 ln -sf libopenblasp-r0.3.8.a libopenblas.a
	@cd /home/michel/new_qtjetson/sysroot/usr/lib/aarch64-linux-gnu && \
	 rm -f libblas.so libblas.so.3 libblas.a && \
	 ln -sf openblas-pthread/libopenblasp-r0.3.8.so libblas.so && \
	 ln -sf openblas-pthread/libopenblasp-r0.3.8.so libblas.so.3 && \
	 ln -sf openblas-pthread/libopenblasp-r0.3.8.a libblas.a && \
	 rm -f liblapack.so liblapack.so.3 liblapack.a && \
	 ln -sf openblas-pthread/libopenblasp-r0.3.8.so liblapack.so && \
	 ln -sf openblas-pthread/libopenblasp-r0.3.8.so liblapack.so.3 && \
	 ln -sf openblas-pthread/libopenblasp-r0.3.8.a liblapack.a && \
	 ln -sf libgfortran.so.5.0.0 libgfortran.so
	@echo "✅ Library links fixed!"

# Verificação da toolchain e dependências
check-toolchain:
	@echo "🔧 Checking cross-compilation toolchain..."
	@which $(CXX) > /dev/null && echo "✅ Cross-compiler found: $(CXX)" || echo "❌ Cross-compiler not found!"
	@test -d /home/michel/new_qtjetson/sysroot && echo "✅ Sysroot found" || echo "❌ Sysroot not found!"
	@test -f /home/michel/new_qtjetson/qt5.15/bin/moc && echo "✅ Qt MOC found" || echo "❌ Qt MOC not found!"
	@test -f /home/michel/new_qtjetson/sysroot/usr/lib/aarch64-linux-gnu/libQt5Core.so.5 && echo "✅ Qt5Core library found" || echo "❌ Qt5Core library not found!"

# Limpeza completa
clean:
	@echo "🧹 Cleaning all build artifacts..."
	rm -f $(OBJECTS) $(MOC_FILES) $(TARGET) enhanced_mpc_main enhanced_mpc_main.moc lane_detection_video_test
	find . -type f \( -name '*.gcno' -o -name '*.gcda' -o -name '*.gcov' \) -exec rm -f {} +
	@echo "✅ Clean complete!"

# Informações sobre o build
info:
	@echo "📊 BUILD CONFIGURATION INFORMATION"
	@echo "=================================="
	@echo "🔧 Compiler: $(CXX)"
	@echo "🎯 Target Architecture: ARM64 (aarch64)"
	@echo "🖥️  Target Device: NVIDIA Jetson Nano"
	@echo "📚 Features Enabled:"
	@echo "   - CUDA acceleration"
	@echo "   - TensorRT inference"
	@echo "   - Qt5 GUI framework"
	@echo "   - OpenCV with CUDA"
	@echo "   - MPC optimization"
	@echo "   - ZeroMQ messaging"
	@echo "   - Multi-threading"
	@echo "🏗️  Build Type: Release (O3 optimized)"
	@echo "📁 Sources: $(words $(ENHANCED_SOURCES)) files"
	@echo "=================================="

.PHONY: all clean enhanced_mpc_main enhanced_mpc_minimal check-toolchain info fix-links
