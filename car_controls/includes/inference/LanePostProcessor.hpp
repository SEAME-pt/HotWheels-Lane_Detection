#ifndef LANEPOSTPROCESSOR_HPP
#define LANEPOSTPROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <vector>

class LanePostProcessor {
public:
    struct LaneInfo {
        int index;
        float angle;
        cv::Point centroid;
        std::vector<cv::Point> targets;
        std::vector<cv::Point> contour;
    };

    LanePostProcessor(int minArea, int minLength, float angleThresh, float mergeDist);

    cv::cuda::GpuMat process(const cv::cuda::GpuMat& rawMaskGpu);

private:
    int minComponentSize;
    int minComponentLength;
    float angleThreshold;
    float mergeDistance;

    std::vector<LaneInfo> extractLaneInfo(const cv::Mat& mask, bool filterSize, bool filterLength);
    bool shouldMerge(const LaneInfo& a, const LaneInfo& b) const;
    std::pair<cv::Point, cv::Point> getExtremities(const std::vector<cv::Point>& contour) const;
    void drawLaneConnections(const std::vector<LaneInfo>& lanes, cv::Mat& canvas) const;
    cv::Mat renderFilteredMask(const std::vector<LaneInfo>& lanes, cv::Size shape) const;
};

#endif


