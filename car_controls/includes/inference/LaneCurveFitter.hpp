#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <optional>

class LaneCurveFitter {
public:
	struct LaneCurve {
		std::vector<cv::Point2f> centroids;
		std::vector<cv::Point2f> curve;
	};

	struct CenterlineResult {
		std::vector<cv::Point2f> blended;
		std::vector<cv::Point2f> c1;
		std::vector<cv::Point2f> c2;
	};

	LaneCurveFitter(float dbscanEps = 5.0f, int dbscanMinSamples = 20, int numWindows = 20, int laneWidthPx = 300);

	std::vector<LaneCurve> fitLanes(const cv::Mat& binaryMask);
	std::optional<CenterlineResult> computeVirtualCenterline(const std::vector<LaneCurve>& lanes, int imgWidth, int imgHeight);

private:
	float dbscanEps;
	int dbscanMinSamples;
	int numWindows;
	int laneWidthPx;

	std::vector<cv::Point> extractLanePoints(const cv::Mat& binaryMask);
	std::vector<int> dbscanCluster(const std::vector<cv::Point>& points, std::vector<int>& uniqueLabels);
	std::pair<std::vector<float>, std::vector<float>> slidingWindowCentroids(const std::vector<cv::Point>& cluster, cv::Size imgSize, bool smooth);
	std::vector<float> fitCurve(const std::vector<float>& y, const std::vector<float>& x, const std::vector<float>& yEval);
	bool hasSignFlip(const std::vector<float>& xValues);
	bool isStraightLine(const std::vector<float>& y, const std::vector<float>& x, float threshold = 0.98f);
};
