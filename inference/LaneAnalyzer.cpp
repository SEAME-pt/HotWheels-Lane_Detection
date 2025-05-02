#include "LaneAnalyzer.hpp"
#include <cmath>
#include <iostream>

LaneAnalyzer::LaneAnalyzer(float realLaneWidthMeters)
    : realLaneWidthMeters(realLaneWidthMeters) {}

Eigen::Vector3f LaneAnalyzer::fitPolynomial(const std::vector<cv::Point>& points) {
    Eigen::MatrixXf A(points.size(), 3);
    Eigen::VectorXf X(points.size());

    for (size_t i = 0; i < points.size(); ++i) {
        float y = static_cast<float>(points[i].y);
        A(i, 0) = y * y;
        A(i, 1) = y;
        A(i, 2) = 1.0f;
        X(i) = static_cast<float>(points[i].x);
    }

    return A.colPivHouseholderQr().solve(X);
}

bool LaneAnalyzer::extractLanePixels(const cv::Mat& mask, std::vector<cv::Point>& outPixels, bool left) {
    int halfWidth = mask.cols / 2;
    int xStart = left ? 0 : halfWidth;
    int xEnd   = left ? halfWidth : mask.cols;

    outPixels.clear();

    for (int y = mask.rows / 2; y < mask.rows; ++y) {
        for (int x = xStart; x < xEnd; ++x) {
            if (mask.at<uchar>(y, x) > 0) {
                outPixels.emplace_back(x, y);
            }
        }
    }

    return outPixels.size() >= 50;  // only accept lanes with enough signal
}

float LaneAnalyzer::evaluatePoly(const Eigen::Vector3f& coeffs, int y) {
    return coeffs[0] * y * y + coeffs[1] * y + coeffs[2];
}

float LaneAnalyzer::estimateMetersPerPixel(float xLeft, float xRight) {
    float pixelWidth = std::abs(xRight - xLeft);
    return (pixelWidth > 0.0f) ? realLaneWidthMeters / pixelWidth : 0.05f;
}

std::string LaneAnalyzer::classifyPosition(float offsetMeters) {
    if (std::abs(offsetMeters) < 0.1f) return "Centered";
    if (offsetMeters > 0.1f) return "Right";
    if (offsetMeters < -0.1f) return "Left";
    return "Unknown";
}

LaneMetrics LaneAnalyzer::computeMetrics(const cv::Mat& mask) {
    LaneMetrics metrics = {};
    metrics.valid = false;
    metrics.positionStatus = "Unknown";

    if (mask.empty() || mask.channels() != 1) {
        std::cerr << "[LaneAnalyzer] Invalid input mask." << std::endl;
        return metrics;
    }

    std::vector<cv::Point> leftPixels, rightPixels;
    bool hasLeft = extractLanePixels(mask, leftPixels, true);
    bool hasRight = extractLanePixels(mask, rightPixels, false);

    // Debug print to confirm
    std::cout << "[DEBUG] Left pixels: " << leftPixels.size()
              << ", Right pixels: " << rightPixels.size() << std::endl;

    if (!hasLeft && !hasRight) {
        std::cerr << "[LaneAnalyzer] No valid lanes detected." << std::endl;
        return metrics;
    }

    int y_eval = mask.rows - 1;
    float imageCenterX = mask.cols / 2.0f;
    float x_lane = 0.0f;
    float curvature = 0.0f;
    float headingDeg = 0.0f;
    float metersPerPixel = 0.05f;

    if (hasLeft && hasRight) {
        Eigen::Vector3f coeffLeft = fitPolynomial(leftPixels);
        Eigen::Vector3f coeffRight = fitPolynomial(rightPixels);

        float x_left = evaluatePoly(coeffLeft, y_eval);
        float x_right = evaluatePoly(coeffRight, y_eval);
        x_lane = (x_left + x_right) / 2.0f;

        curvature = (2.0f * coeffLeft[0] + 2.0f * coeffRight[0]) / 2.0f;

        float headingLeft = std::atan(2.0f * coeffLeft[0] * y_eval + coeffLeft[1]);
        float headingRight = std::atan(2.0f * coeffRight[0] * y_eval + coeffRight[1]);
        headingDeg = ((headingLeft + headingRight) / 2.0f) * 180.0f / CV_PI;

        metersPerPixel = estimateMetersPerPixel(x_left, x_right);
    }
    else if (hasLeft) {
        Eigen::Vector3f coeff = fitPolynomial(leftPixels);
        x_lane = evaluatePoly(coeff, y_eval);
        curvature = 2.0f * coeff[0];
        headingDeg = std::atan(2.0f * coeff[0] * y_eval + coeff[1]) * 180.0f / CV_PI;
    }
    else if (hasRight) {
        Eigen::Vector3f coeff = fitPolynomial(rightPixels);
        x_lane = evaluatePoly(coeff, y_eval);
        curvature = 2.0f * coeff[0];
        headingDeg = std::atan(2.0f * coeff[0] * y_eval + coeff[1]) * 180.0f / CV_PI;
    }

    float offset_px = x_lane - imageCenterX;
    float offset_m = offset_px * metersPerPixel;

    metrics.lateralOffsetPx = offset_px;
    metrics.lateralOffsetMeters = offset_m;
    metrics.curvature = curvature;
    metrics.headingAngleDeg = headingDeg;
    metrics.positionStatus = classifyPosition(offset_m);
    metrics.valid = true;

    return metrics;
}

