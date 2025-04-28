#ifndef CAMERA_STREAMER_HPP
#define CAMERA_STREAMER_HPP

#include <iostream>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudawarping.hpp>
#include "TensorRTInferencer.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

class CameraStreamer {
public:
    CameraStreamer(TensorRTInferencer& infer, double scale = 0.5, const std::string& win_name = "CSI Camera", bool show_orig = false);
    ~CameraStreamer();

    void initOpenGL();
    void initUndistortMaps();
    void uploadFrameToTexture(const cv::cuda::GpuMat& gpuFrame);
    void renderTexture();

    void start();

private:
    cv::VideoCapture cap;
    double scale_factor;
    std::string window_name;
    TensorRTInferencer& inferencer;
    bool show_original;

    cv::cuda::GpuMat d_mapx, d_mapy;

    GLFWwindow* window;
    GLuint textureID;
    int window_width, window_height;

    cudaGraphicsResource* cuda_resource;
};

#endif // CAMERA_STREAMER_HPP
