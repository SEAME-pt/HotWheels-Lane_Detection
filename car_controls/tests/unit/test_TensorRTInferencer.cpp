// test_TensorRTInferencer.cpp

#include <gtest/gtest.h>
#include "../includes/inference/TensorRTInferencer.hpp"

#include <opencv2/core/cuda.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <thread>

TEST(TensorRTInferencerTest, CanReadEngineFile) {
    TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");
    ASSERT_TRUE(true);  // If it didn't throw, it's OK for this smoke test
}

TEST(TensorRTInferencerTest, WrongEngineFile) {
    try {
        TensorRTInferencer inferencer("/invalid/path/to/engine.engine");
        FAIL() << "Expected std::runtime_error but none was thrown.";
    } catch (const std::runtime_error& e) {
        std::cout << "[INFO] Caught std::runtime_error: " << e.what() << std::endl;
        SUCCEED();
    } catch (...) {
        FAIL() << "Caught unknown exception type.";
    }
}

TEST(TensorRTInferencerTest, PreprocessImageGrayscale) {
    cv::Mat cpuImg(208, 208, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(cpuImg);

    TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");

    cv::cuda::GpuMat processed = inferencer.preprocessImage(gpuImg);

    EXPECT_EQ(processed.type(), CV_32F);
    EXPECT_EQ(processed.size(), cv::Size(208, 208));
}

TEST(TensorRTInferencerTest, ThrowsOnEmptyImage) {
    cv::cuda::GpuMat emptyImg;

    TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");

    //EXPECT_THROW(inferencer.preprocessImage(emptyImg), std::runtime_error);

    try {
        inferencer.preprocessImage(emptyImg);
        FAIL() << "Expected std::runtime_error but none was thrown.";
    } catch (const std::runtime_error& e) {
        std::cout << "[INFO] Caught std::runtime_error: " << e.what() << std::endl;
        SUCCEED();
    } catch (...) {
        FAIL() << "Caught unknown exception type.";
    }
}

TEST(TensorRTInferencerTest, RunInferenceThrowsOnWrongSize) {
    TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");

    cv::Mat smallImg(100, 100, CV_32FC1, cv::Scalar(1.0f));
    cv::cuda::GpuMat gpuInput;
    gpuInput.upload(smallImg);

    //EXPECT_THROW(inferencer.runInference(gpuInput), std::runtime_error);

    try {
        inferencer.runInference(gpuInput);
        FAIL() << "Expected std::runtime_error but none was thrown.";
    } catch (const std::runtime_error& e) {
        std::cout << "[INFO] Caught std::runtime_error: " << e.what() << std::endl;
        SUCCEED();
    } catch (...) {
        FAIL() << "Caught unknown exception type.";
    }
}

TEST(TensorRTInferencerTest, RunInferenceSucceedsOnValidInput) {
    TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");

    cv::Mat validImg(208, 208, CV_32FC1, cv::Scalar(1.0f));
    cv::cuda::GpuMat gpuInput;
    gpuInput.upload(validImg);

    EXPECT_NO_THROW(inferencer.runInference(gpuInput));
}

TEST(TensorRTInferencerTest, RunInferenceFailsOnNullInput) {
    TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");

    cv::cuda::GpuMat nullInput;  // uninitialized
    //EXPECT_THROW(inferencer.runInference(nullInput), std::runtime_error);

    try {
        inferencer.runInference(nullInput);
        FAIL() << "Expected std::runtime_error but none was thrown.";
    } catch (const std::runtime_error& e) {
        std::cout << "[INFO] Caught std::runtime_error: " << e.what() << std::endl;
        SUCCEED();
    } catch (...) {
        FAIL() << "Caught unknown exception type.";
    }
}

TEST(TensorRTInferencerTest, OutputHasNonZeroValuesAfterInference) {
    TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");

    cv::Mat validImg(208, 208, CV_32FC1, cv::Scalar(1.0f));
    cv::cuda::GpuMat gpuInput;
    gpuInput.upload(validImg);

    inferencer.runInference(gpuInput);

    cv::Mat outputCpu(208, 208, CV_32F);
    cudaMemcpy(outputCpu.ptr<float>(), inferencer.getDeviceOutputPtr(),
               outputCpu.total() * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = cv::sum(outputCpu)[0];
    EXPECT_GT(sum, 0.0);
}

TEST(TensorRTInferencerTest, PreprocessGrayscaleNoConvert) {
    TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");

    cv::Mat gray(208, 208, CV_8UC1, cv::Scalar(100));
    cv::cuda::GpuMat gpuGray;
    gpuGray.upload(gray);

    cv::cuda::GpuMat result = inferencer.preprocessImage(gpuGray);

    EXPECT_EQ(result.type(), CV_32F);
}

TEST(TensorRTInferencerTest, PreprocessGrayscaleWithConvert) {
    TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");

    cv::Mat gray(208, 208, CV_8UC1, cv::Scalar(100));
    cv::cuda::GpuMat gpuGray;
    gpuGray.upload(gray);

    cv::cuda::GpuMat result = inferencer.preprocessImage(gpuGray);

    EXPECT_EQ(result.type(), CV_32F);
}

TEST(TensorRTInferencerTest, PreprocessColorImage) {
    TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");

    cv::Mat colorImg(208, 208, CV_8UC3, cv::Scalar(100, 150, 200));
    cv::cuda::GpuMat gpuColorImg;
    gpuColorImg.upload(colorImg);

    cv::cuda::GpuMat result = inferencer.preprocessImage(gpuColorImg);

    EXPECT_EQ(result.type(), CV_32F);
    EXPECT_EQ(result.size(), cv::Size(208, 208));
}

TEST(TensorRTInferencerTest, PreprocessImageWithInvalidType) {
    TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");

    cv::Mat invalidImg(208, 208, CV_8UC4, cv::Scalar(100, 150, 200, 255));
    cv::cuda::GpuMat gpuInvalidImg;
    gpuInvalidImg.upload(invalidImg);

    //EXPECT_THROW(inferencer.preprocessImage(gpuInvalidImg), std::runtime_error);

    try {
        inferencer.preprocessImage(gpuInvalidImg);
        FAIL() << "Expected std::runtime_error but none was thrown.";
    } catch (const std::runtime_error& e) {
        std::cout << "[INFO] Caught std::runtime_error: " << e.what() << std::endl;
        SUCCEED();
    } catch (...) {
        FAIL() << "Caught unknown exception type.";
    }
}

TEST(TensorRTInferencerTest, PreprocessImageWithEmptyGpuMat) {
    TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");

    cv::cuda::GpuMat emptyGpuMat;

    //EXPECT_THROW(inferencer.preprocessImage(emptyGpuMat), std::runtime_error);

    try {
        inferencer.preprocessImage(emptyGpuMat);
        FAIL() << "Expected std::runtime_error but none was thrown.";
    } catch (const std::runtime_error& e) {
        std::cout << "[INFO] Caught std::runtime_error: " << e.what() << std::endl;
        SUCCEED();
    } catch (...) {
        FAIL() << "Caught unknown exception type.";
    }
}

TEST(TensorRTInferencerTest, PreprocessImageWithWrongSize) {
    TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");

    cv::Mat smallImg(100, 100, CV_8UC3, cv::Scalar(100, 150, 200));
    cv::cuda::GpuMat gpuSmallImg;
    gpuSmallImg.upload(smallImg);

    EXPECT_NO_THROW(inferencer.preprocessImage(gpuSmallImg));
}

TEST(TensorRTInferencerTest, PreprocessImageWithValidSize) {
    TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");

    cv::Mat validImg(208, 208, CV_8UC3, cv::Scalar(100, 150, 200));
    cv::cuda::GpuMat gpuValidImg;
    gpuValidImg.upload(validImg);

    EXPECT_NO_THROW(inferencer.preprocessImage(gpuValidImg));
}

TEST(TensorRTInferencerTest, PreprocessImageWithValidSizeAndType) {
    TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");

    cv::Mat validImg(208, 208, CV_8UC3, cv::Scalar(100, 150, 200));
    cv::cuda::GpuMat gpuValidImg;
    gpuValidImg.upload(validImg);

    EXPECT_NO_THROW(inferencer.preprocessImage(gpuValidImg));
}

TEST(TensorRTInferencerTest, PreprocessImageWithValidSizeAndInvalidType) {
    TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");

    cv::Mat validImg(208, 208, CV_32FC1, cv::Scalar(0.5f));
    cv::cuda::GpuMat gpuValidImg;
    gpuValidImg.upload(validImg);

    //EXPECT_THROW(inferencer.preprocessImage(gpuValidImg), std::runtime_error);

    try {
        inferencer.preprocessImage(gpuValidImg);
        FAIL() << "Expected std::runtime_error but none was thrown.";
    } catch (const std::runtime_error& e) {
        std::cout << "[INFO] Caught std::runtime_error: " << e.what() << std::endl;
        SUCCEED();
    } catch (...) {
        FAIL() << "Caught unknown exception type.";
    }
}

TEST(TensorRTInferencerTest, MakePredictionDoesNotThrowOnValidInput) {
    TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");

    cv::Mat input(208, 208, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::cuda::GpuMat gpuInput;
    gpuInput.upload(input);

    EXPECT_NO_THROW(inferencer.makePrediction(gpuInput));
}

TEST(TensorRTInferencerTest, MakePredictionReturnsCorrectSize) {
    TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");

    cv::Mat input(208, 208, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::cuda::GpuMat gpuInput;
    gpuInput.upload(input);

    cv::cuda::GpuMat result = inferencer.makePrediction(gpuInput);

    // Make sure this matches the model’s actual outputDims
    EXPECT_EQ(result.size(), cv::Size(208, 208));
    EXPECT_EQ(result.type(), CV_32F);
}

TEST(TensorRTInferencerTest, MakePredictionOutputNotAllZero) {
    TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");

    cv::Mat input(208, 208, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::cuda::GpuMat gpuInput;
    gpuInput.upload(input);

    cv::cuda::GpuMat result = inferencer.makePrediction(gpuInput);

    // Download to CPU for inspection
    cv::Mat resultCpu;
    result.download(resultCpu);
    float sum = cv::sum(resultCpu)[0];

    EXPECT_GT(sum, 0.0);  // should not be all zeros
}

TEST(TensorRTInferencerTest, MakePredictionReturnsGpuMat) {
    TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");

    cv::Mat cpuImg(208, 208, CV_8UC3, cv::Scalar(120, 120, 120));
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(cpuImg);

    cv::cuda::GpuMat result = inferencer.makePrediction(gpuImg);

    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.type(), CV_32F);
}

TEST(TensorRTInferencerTest, PredictionOutputHasExpectedRange) {
    TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");

    cv::Mat input(208, 208, CV_8UC3, cv::Scalar(120, 120, 120));
    cv::cuda::GpuMat gpuInput;
    gpuInput.upload(input);

    auto output = inferencer.makePrediction(gpuInput);
    cv::Mat outputCpu;
    output.download(outputCpu);

    double minVal, maxVal;
    cv::minMaxLoc(outputCpu, &minVal, &maxVal);

    EXPECT_GE(minVal, 0.0);
    EXPECT_LE(maxVal, 1.0);
}

TEST(TensorRTInferencerTest, ReuseInferenceMultipleTimes) {
    TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");

    for (int i = 0; i < 10; ++i) {
        cv::Mat img(208, 208, CV_8UC3, cv::Scalar(i * 25, i * 25, i * 25));
        cv::cuda::GpuMat gpu;
        gpu.upload(img);
        EXPECT_NO_THROW(inferencer.makePrediction(gpu));
    }
}

TEST(TensorRTInferencerTest, MakePredictionIsDeterministic) {
    TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");

    cv::Mat input(208, 208, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::cuda::GpuMat gpuInput;
    gpuInput.upload(input);

    auto res1 = inferencer.makePrediction(gpuInput);
    auto res2 = inferencer.makePrediction(gpuInput);

    cv::Mat cpu1, cpu2;
    res1.download(cpu1);
    res2.download(cpu2);

    cv::Mat diff;
    cv::absdiff(cpu1, cpu2, diff);
    double maxDiff;
    cv::minMaxLoc(diff, nullptr, &maxDiff);

    EXPECT_LT(maxDiff, 1e-4);  // very small difference
}
