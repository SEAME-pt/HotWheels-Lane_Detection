#include "../../includes/inference/LanePostProcessor.hpp"
#include <opencv2/cudaarithm.hpp>
#include <cmath>
#include <numeric>

LanePostProcessor::LanePostProcessor(int minArea, int minLength, float angleThresh, float mergeDist)
	: minComponentSize(minArea),
	  minComponentLength(minLength),
	  angleThreshold(angleThresh),
	  mergeDistance(mergeDist) {}

cv::cuda::GpuMat LanePostProcessor::process(const cv::cuda::GpuMat& rawMaskGpu) {
	cv::cuda::GpuMat binaryMaskGpu;
	cv::cuda::threshold(rawMaskGpu, binaryMaskGpu, 0.5, 255.0, cv::THRESH_BINARY);
	binaryMaskGpu.convertTo(binaryMaskGpu, CV_8U);

	cv::Mat binaryMask;
	binaryMaskGpu.download(binaryMask); // CPU copy for contour analysis

	std::vector<LaneInfo> initialLanes = extractLaneInfo(binaryMask, false, true);

	cv::Mat mergedMask;
	cv::cvtColor(binaryMask, mergedMask, cv::COLOR_GRAY2BGR);
	drawLaneConnections(initialLanes, mergedMask);

	cv::Mat grayMerged;
	cv::cvtColor(mergedMask, grayMerged, cv::COLOR_BGR2GRAY);

	std::vector<LaneInfo> finalLanes = extractLaneInfo(grayMerged, true, false);
	cv::Mat filtered = renderFilteredMask(finalLanes, binaryMask.size());

	cv::cuda::GpuMat resultGpu;
	resultGpu.upload(filtered);
	return resultGpu;
}

std::vector<LanePostProcessor::LaneInfo> LanePostProcessor::extractLaneInfo(const cv::Mat& mask, bool filterSize, bool filterLength) {
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	std::vector<LaneInfo> lanes;
	int height = mask.rows, width = mask.cols;

	for (const auto& cnt : contours) {
		double area = cv::contourArea(cnt);
		double length = cv::arcLength(cnt, false);

		if ((!filterSize || area > minComponentSize) &&
			(!filterLength || length > minComponentLength) &&
			cnt.size() >= 2) {

			cv::Vec4f line;
			cv::fitLine(cnt, line, cv::DIST_L2, 0, 0.01, 0.01);
			float vx = line[0], vy = line[1], x0 = line[2], y0 = line[3];
			float angle = std::fmod(std::atan2(vy, vx) * 180.0 / CV_PI, 180.0f);

			cv::Moments M = cv::moments(cnt);
			if (M.m00 == 0) continue;
			cv::Point centroid(M.m10 / M.m00, M.m01 / M.m00);

			std::vector<cv::Point> intersections;
			if (vy != 0) {
				float t = -y0 / vy;
				float x = x0 + vx * t;
				if (x >= 0 && x < width) intersections.emplace_back(x, 0);
				t = (height - 1 - y0) / vy;
				x = x0 + vx * t;
				if (x >= 0 && x < width) intersections.emplace_back(x, height - 1);
			}
			if (vx != 0) {
				float t = -x0 / vx;
				float y = y0 + vy * t;
				if (y >= 0 && y < height) intersections.emplace_back(0, y);
				t = (width - 1 - x0) / vx;
				y = y0 + vy * t;
				if (y >= 0 && y < height) intersections.emplace_back(width - 1, y);
			}

			if (intersections.size() >= 2) {
				lanes.push_back({ static_cast<int>(lanes.size() + 1), angle, centroid, {intersections[0], intersections[1]}, cnt });
			}
		}
	}

	return lanes;
}

bool LanePostProcessor::shouldMerge(const LaneInfo& a, const LaneInfo& b) const {
	float angleDiff = std::fabs(a.angle - b.angle);
	angleDiff = std::min(angleDiff, 180.0f - angleDiff);

	if (angleDiff < angleThreshold) {
		for (const auto& pt1 : a.targets) {
			for (const auto& pt2 : b.targets) {
				if (cv::norm(pt1 - pt2) < mergeDistance)
					return true;
			}
		}
	}
	return false;
}

std::pair<cv::Point, cv::Point> LanePostProcessor::getExtremities(const std::vector<cv::Point>& contour) const {
	double maxDist = 0;
	cv::Point ext1 = contour[0], ext2 = contour[0];

	for (size_t i = 0; i < contour.size(); ++i) {
		for (size_t j = i + 1; j < contour.size(); ++j) {
			double dist = cv::norm(contour[i] - contour[j]);
			if (dist > maxDist) {
				maxDist = dist;
				ext1 = contour[i];
				ext2 = contour[j];
			}
		}
	}

	return {ext1, ext2};
}

void LanePostProcessor::drawLaneConnections(const std::vector<LaneInfo>& lanes, cv::Mat& canvas) const {
	for (size_t i = 0; i < lanes.size(); ++i) {
		for (size_t j = i + 1; j < lanes.size(); ++j) {
			if (shouldMerge(lanes[i], lanes[j])) {
				auto [a1, a2] = getExtremities(lanes[i].contour);
				auto [b1, b2] = getExtremities(lanes[j].contour);

				std::vector<std::pair<cv::Point, cv::Point>> pairs = {
					{a1, b1}, {a1, b2}, {a2, b1}, {a2, b2}
				};

				auto bestPair = *std::min_element(pairs.begin(), pairs.end(), [](const auto& p1, const auto& p2) {
					return cv::norm(p1.first - p1.second) < cv::norm(p2.first - p2.second);
				});

				cv::line(canvas, bestPair.first, bestPair.second, cv::Scalar(255, 255, 255), 3);
			}
		}
	}
}

cv::Mat LanePostProcessor::renderFilteredMask(const std::vector<LaneInfo>& lanes, cv::Size shape) const {
	cv::Mat canvas(shape, CV_8UC3, cv::Scalar(0, 0, 0));
	for (const auto& lane : lanes) {
		cv::drawContours(canvas, std::vector<std::vector<cv::Point>>{lane.contour}, -1, cv::Scalar(255, 255, 255), cv::FILLED);
	}
	return canvas;
}
