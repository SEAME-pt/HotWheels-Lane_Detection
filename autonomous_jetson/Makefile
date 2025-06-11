# Makefile for autonomous_jetson
# Compila todos os arquivos .cpp em src/ e gera o executável main

CXX = c++
CXXFLAGS = -std=c++17 -Wall -Werror -Wextra -Iinc -pedantic -fopenmp -O2
SRC = src/MPCOptimizer.cpp src/MPCPlanner.cpp src/Polyfitter.cpp
OBJ = $(SRC:.cpp=.o)
TARGET = main

# Bibliotecas externas
# OpenCV
# Tenta opencv4, se não existir, tenta opencv
ifeq (, $(shell pkg-config --exists opencv4 && echo 1))
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv)
else
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)
endif
# mlpack
MLPACK_FLAGS = $(shell pkg-config --cflags --libs mlpack)
# Eigen (header-only)
EIGEN_FLAGS = -I/usr/include/eigen3
# NLOpt
NLOPT_FLAGS = -lnlopt

all: $(TARGET)

$(TARGET): main.o $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OPENCV_FLAGS) $(MLPACK_FLAGS) $(EIGEN_FLAGS) $(NLOPT_FLAGS)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(OPENCV_FLAGS) $(EIGEN_FLAGS)

src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(OPENCV_FLAGS) $(MLPACK_FLAGS) $(EIGEN_FLAGS)

clean:
	rm -f $(OBJ) main.o $(TARGET)

.PHONY: all clean
