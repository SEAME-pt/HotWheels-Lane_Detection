#include "YOLOv5TRT.hpp"
#include <gtest/gtest.h>

class YOLOv5TRT_Testable : public YOLOv5TRT {
	public:
		YOLOv5TRT_Testable()
		    : YOLOv5TRT("/home/jetson/models/object-detection/yolov5m_updated.engine",
		                "/home/jetson/models/object-detection/labels.txt") {}

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
	float conf_thresh = 0.25F;
	float nms_thresh = 0.5F;

	std::vector<float> output = {100.0F, 200.0F, 50.0F, 80.0F, 0.9F,  0.6F, 0.3F, 0.1F,
	                             150.0F, 210.0F, 48.0F, 75.0F, 0.85F, 0.5F, 0.4F, 0.2F};

	auto detections = yolo.postprocess(output, num_classes, conf_thresh, nms_thresh);
	ASSERT_EQ(detections.size(), 2);
}

TEST(YOLOv5TRTTest, PostprocessNoDetections) {
	YOLOv5TRT_Testable yolo;

	int num_classes = 3;
	float conf_thresh = 0.25F;
	float nms_thresh = 0.5F;

	std::vector<float> output = {}; // No detections

	auto detections = yolo.postprocess(output, num_classes, conf_thresh, nms_thresh);
	ASSERT_EQ(detections.size(), 0);
}

TEST(YOLOv5TRTTest, PostprocessSingleDetection) {
	YOLOv5TRT_Testable yolo;

	int num_classes = 3;
	float conf_thresh = 0.25F;
	float nms_thresh = 0.5F;

	std::vector<float> output = {100.0F, 200.0F, 50.0F, 80.0F, 0.9F, 0.6F, 0.3F, 0.1F};

	auto detections = yolo.postprocess(output, num_classes, conf_thresh, nms_thresh);
	ASSERT_EQ(detections.size(), 1);
	EXPECT_EQ(detections[0].class_id, 0); // Assuming class_id starts from 0
}

TEST(YOLOv5TRTTest, PostprocessWithLowConfidence) {
	YOLOv5TRT_Testable yolo;

	int num_classes = 3;
	float conf_thresh = 0.25F;
	float nms_thresh = 0.5F;

	std::vector<float> output = {
	    100.0F, 200.0F, 50.0F, 80.0F, 0.2F, 0.6F, 0.3F, 0.1F // Low confidence
	};

	auto detections = yolo.postprocess(output, num_classes, conf_thresh, nms_thresh);
	ASSERT_EQ(detections.size(), 0); // Should filter out low confidence detection
}
