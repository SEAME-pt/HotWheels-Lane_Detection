#include <gtest/gtest.h>
#include "../../includes/objectDetection/YOLOv5TRT.hpp"

class YOLOv5TRT_Testable : public YOLOv5TRT {
public:
    YOLOv5TRT_Testable() : YOLOv5TRT("/home/hotweels/cam_calib/models/yolov5m_updated.engine", "/home/hotweels/cam_calib/models/labels.txt") {}

    using YOLOv5TRT::calculateVolume;
    using YOLOv5TRT::postprocess;
};

TEST(YOLOv5TRTTest, CalculateVolume) {
    YOLOv5TRT_Testable yolo;
    nvinfer1::Dims dims;
    dims.nbDims = 3;
    dims.d[0] = 3;
    dims.d[1] = 640;
    dims.d[2] = 640;

    EXPECT_EQ(yolo.calculateVolume(dims), 3 * 640 * 640);
}

TEST(YOLOv5TRTTest, PostprocessDetections) {
    YOLOv5TRT_Testable yolo;

    int num_classes = 3;
    float conf_thresh = 0.25f;
    float nms_thresh = 0.5f;

    std::vector<float> output = {
        100.0f, 200.0f, 50.0f, 80.0f, 0.9f, 0.6f, 0.3f, 0.1f,
        150.0f, 210.0f, 48.0f, 75.0f, 0.85f, 0.5f, 0.4f, 0.2f
    };

    auto detections = yolo.postprocess(output, num_classes, conf_thresh, nms_thresh);
    ASSERT_EQ(detections.size(), 2);
}

TEST(YOLOv5TRTTest, PostprocessNoDetections) {
	YOLOv5TRT_Testable yolo;

	int num_classes = 3;
	float conf_thresh = 0.25f;
	float nms_thresh = 0.5f;

	std::vector<float> output = {}; // No detections

	auto detections = yolo.postprocess(output, num_classes, conf_thresh, nms_thresh);
	ASSERT_EQ(detections.size(), 0);
}

TEST(YOLOv5TRTTest, PostprocessSingleDetection) {
	YOLOv5TRT_Testable yolo;

	int num_classes = 3;
	float conf_thresh = 0.25f;
	float nms_thresh = 0.5f;

	std::vector<float> output = {
		100.0f, 200.0f, 50.0f, 80.0f, 0.9f, 0.6f, 0.3f, 0.1f
	};

	auto detections = yolo.postprocess(output, num_classes, conf_thresh, nms_thresh);
	ASSERT_EQ(detections.size(), 1);
	EXPECT_EQ(detections[0].class_id, 0); // Assuming class_id starts from 0
}

TEST(YOLOv5TRTTest, PostprocessWithLowConfidence) {
	YOLOv5TRT_Testable yolo;

	int num_classes = 3;
	float conf_thresh = 0.25f;
	float nms_thresh = 0.5f;

	std::vector<float> output = {
		100.0f, 200.0f, 50.0f, 80.0f, 0.2f, 0.6f, 0.3f, 0.1f // Low confidence
	};

	auto detections = yolo.postprocess(output, num_classes, conf_thresh, nms_thresh);
	ASSERT_EQ(detections.size(), 0); // Should filter out low confidence detection
}

