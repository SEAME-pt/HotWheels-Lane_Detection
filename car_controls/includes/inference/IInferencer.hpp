// IInferencer.hpp
#ifndef IINFERENCER_HPP
#define IINFERENCER_HPP

#include <opencv2/core/cuda.hpp>

class IInferencer {
public:
	virtual ~IInferencer() = default;

	// The main method your dependent code will use
	virtual cv::cuda::GpuMat makePrediction(const cv::cuda::GpuMat& gpuImage) = 0;
	virtual void doInference(const cv::Mat& frame) = 0; // Run inference on a given frame
};

#endif // IINFERENCER_HPP
