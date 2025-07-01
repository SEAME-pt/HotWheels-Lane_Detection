#ifndef MOCKINFERENCER_HPP
#define MOCKINFERENCER_HPP

#include "IInferencer.hpp"
#include <gmock/gmock.h>

using ::testing::AtLeast;
using ::testing::NiceMock;

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>

class MockInferencer : public IInferencer {
	public:
		MOCK_METHOD(cv::cuda::GpuMat, makePrediction, (const cv::cuda::GpuMat &), (override));
};

#endif
