#ifndef LANE_ANALYZER_HPP
#define LANE_ANALYZER_HPP

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <string>
#include <vector>

struct LaneMetrics {
    float lateralOffsetPx;
    float lateralOffsetMeters;
    float curvature;
    float headingAngleDeg;
    std::string positionStatus;
    bool valid;
};

class LaneAnalyzer {
public:
    LaneAnalyzer(float realLaneWidthMeters = 3.5f);

    LaneMetrics computeMetrics(const cv::Mat& binaryMask);

private:
    float realLaneWidthMeters;

    Eigen::Vector3f fitPolynomial(const std::vector<cv::Point>& points);
    bool extractLanePixels(const cv::Mat& mask, std::vector<cv::Point>& outPixels, bool left);
    float evaluatePoly(const Eigen::Vector3f& coeffs, int y);
    float estimateMetersPerPixel(float xLeft, float xRight);
    std::string classifyPosition(float offsetMeters);
};

#endif
