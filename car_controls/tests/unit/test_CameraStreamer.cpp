// test_TensorRTInferencer.cpp

#include <gtest/gtest.h>
#include "../mocks/MockInferencer.hpp"
#include "../includes/inference/CameraStreamer.hpp"

/* TEST(CameraStreamerTest, ConstructorInitializesCamera) {
    auto inferencer = std::make_shared<MockInferencer>();
    EXPECT_NO_THROW({
        CameraStreamer streamer(inferencer, true);
    });
}

TEST(CameraStreamerTest, InitUndistortMapsLoadsFile) {
    auto inferencer = std::make_shared<MockInferencer>();
    CameraStreamer streamer(inferencer, true);
    streamer.initUndistortMaps();
    SUCCEED();
}

TEST(CameraStreamerTest, StartCallsInference) {
    auto inferencer = std::make_shared<NiceMock<MockInferencer>>();
    EXPECT_CALL(*inferencer, makePrediction).Times(AtLeast(1));

    CameraStreamer streamer(inferencer, true);

    // Simulate a short run by calling stop() in another thread
    std::thread stopper([&]() {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        streamer.stop();
    });

    streamer.start();  // this will run for ~1s
    stopper.join();
} */


