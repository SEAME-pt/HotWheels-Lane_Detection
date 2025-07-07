#pragma once

#include <opencv2/opencv.hpp>
#include <optional>
#include <vector>

class LaneCurveFitter {
	public:
		struct LaneCurve {
				std::vector<cv::Point2f> centroids;
				std::vector<cv::Point2f> curve;
		};

		struct CenterlineResult {
				std::vector<cv::Point2f> blended;
				std::vector<cv::Point2f> midpoint;
				std::vector<cv::Point2f> straight;
				std::vector<LaneCurve> lanes;
				bool valid;
				
				CenterlineResult() : valid(false) {}
		};

		LaneCurveFitter(float dbscanEps = 5.0F, int dbscanMinSamples = 20, int numWindows = 20,
		                int laneWidthPx = 80);

		std::vector<LaneCurve> fitLanes(const cv::Mat &binaryMask);
		std::optional<CenterlineResult>
		computeVirtualCenterline(const std::vector<LaneCurve> &lanes, int imgWidth, int imgHeight);
		
		// Main public interface
		std::optional<CenterlineResult> computeCenterline(const cv::Mat& binaryMask);

	private:
		float dbscanEps;
		int dbscanMinSamples;
		int numWindows;
		int laneWidthPx;
		
		// Constants
		static constexpr float STRAIGHT_LINE_THRESHOLD = 0.98F;
		static constexpr float CURVE_THRESHOLD = 0.0012F;

		std::vector<cv::Point> extractLanePoints(const cv::Mat &binaryMask);
		std::vector<int> dbscanCluster(const std::vector<cv::Point> &points,
		                               std::vector<int> &uniqueLabels);
		std::pair<std::vector<float>, std::vector<float>>
		slidingWindowCentroids(const std::vector<cv::Point> &cluster, cv::Size imgSize,
		                       bool smooth);
		std::vector<float> fitCurve(const std::vector<float> &y, const std::vector<float> &x,
		                            const std::vector<float> &yEval);
		std::vector<float> polyfit(const std::vector<float> &x, const std::vector<float> &y, int degree);
		std::vector<float> polyval(const std::vector<float> &coeffs, const std::vector<float> &x);
		std::vector<float> linspace(float start, float end, int num);
		std::vector<float> interp(const std::vector<float> &xNew, const std::vector<float> &x,
		                         const std::vector<float> &y, float leftVal, float rightVal);
		std::pair<LaneCurve*, LaneCurve*> selectRelevantLanes(std::vector<LaneCurve> &lanes, 
		                                                      int imgWidth, int imgHeight);
		bool hasSignFlip(const std::vector<float> &xValues);
		bool isStraightLine(const std::vector<float> &y, const std::vector<float> &x,
		                    float threshold = STRAIGHT_LINE_THRESHOLD);
};
