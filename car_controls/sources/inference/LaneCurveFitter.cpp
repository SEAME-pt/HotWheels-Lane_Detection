#include "LaneCurveFitter.hpp"
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <algorithm>

LaneCurveFitter::LaneCurveFitter(float eps, int minSamples, int windows, int laneWidthPx)
    : dbscanEps(eps), dbscanMinSamples(minSamples), numWindows(windows), laneWidthPx(laneWidthPx) {}

std::optional<LaneCurveFitter::CenterlineResult>
LaneCurveFitter::computeCenterline(const cv::Mat &binaryMask) {
	if(binaryMask.empty()) {
		return std::nullopt;
	}

	cv::Mat gray;
	if(binaryMask.channels() == 3) {
		cv::cvtColor(binaryMask, gray, cv::COLOR_BGR2GRAY);
	} else {
		gray = binaryMask;
	}

	auto lanes = fitLanes(gray);
	if(lanes.empty()) {
		return std::nullopt;
	}

	auto result = computeVirtualCenterline(lanes, gray.cols, gray.rows);
	if(!result) {
		return std::nullopt;
	}

	result->lanes = lanes;
	return result;
}

std::vector<cv::Point> LaneCurveFitter::extractLanePoints(const cv::Mat &binaryMask) {
	std::vector<cv::Point> points;
	for(int y = 0; y < binaryMask.rows; ++y) {
		for(int x = 0; x < binaryMask.cols; ++x) {
			if(binaryMask.at<uchar>(y, x) > 0)
				points.emplace_back(x, y);
		}
	}
	return points;
}

float interpolateXatY(const std::vector<cv::Point2f> &points, float y_query) {
	if(points.empty())
		return 0.0F;

	for(size_t i = 1; i < points.size(); ++i) {
		float y1 = points[i - 1].y;
		float y2 = points[i].y;

		if((y1 <= y_query && y_query <= y2) || (y2 <= y_query && y_query <= y1)) {
			float t = (y_query - y1) / (y2 - y1 + 1e-6F);
			float x1 = points[i - 1].x;
			float x2 = points[i].x;
			return x1 + t * (x2 - x1);
		}
	}

	// Extrapolate if y_query is outside range
	return points.back().x;
}

// Simple DBSCAN implementation (brute force)
std::vector<int> LaneCurveFitter::dbscanCluster(const std::vector<cv::Point> &points,
                                                std::vector<int> &uniqueLabels) {
	const int n = points.size();
	std::vector<int> labels(n, -1);
	int clusterId = 0;

	for(int i = 0; i < n; ++i) {
		if(labels[i] != -1)
			continue;

		std::vector<int> neighbors;
		for(int j = 0; j < n; ++j) {
			if(cv::norm(points[i] - points[j]) <= dbscanEps)
				neighbors.push_back(j);
		}

		if(neighbors.size() < static_cast<size_t>(dbscanMinSamples))
			continue;

		labels[i] = clusterId;
		std::set<int> seeds(neighbors.begin(), neighbors.end());
		seeds.erase(i);

		while(!seeds.empty()) {
			int current = *seeds.begin();
			seeds.erase(seeds.begin());

			if(labels[current] == -1) {
				labels[current] = clusterId;

				std::vector<int> currentNeighbors;
				for(int j = 0; j < n; ++j) {
					if(cv::norm(points[current] - points[j]) <= dbscanEps)
						currentNeighbors.push_back(j);
				}

				if(currentNeighbors.size() >= static_cast<size_t>(dbscanMinSamples)) {
					seeds.insert(currentNeighbors.begin(), currentNeighbors.end());
				}
			}
		}

		++clusterId;
	}

	uniqueLabels.clear();
	for(int l : labels)
		if(l != -1)
			uniqueLabels.push_back(l);
	std::sort(uniqueLabels.begin(), uniqueLabels.end());
	uniqueLabels.erase(std::unique(uniqueLabels.begin(), uniqueLabels.end()), uniqueLabels.end());

	return labels;
}

std::pair<std::vector<float>, std::vector<float>>
LaneCurveFitter::slidingWindowCentroids(const std::vector<cv::Point> &cluster, cv::Size imgSize,
                                        bool smooth) {
	std::vector<float> cx, cy;
	int h = imgSize.height / numWindows;

	for(int i = 0; i < numWindows; ++i) {
		int yLow = imgSize.height - (i + 1) * h;
		int yHigh = imgSize.height - i * h;

		std::vector<float> xAcc, yAcc;
		for(const auto &pt : cluster) {
			if(pt.y >= yLow && pt.y < yHigh) {
				xAcc.push_back(pt.x);
				yAcc.push_back(pt.y);
			}
		}

		if(!xAcc.empty()) {
			cx.push_back(std::accumulate(xAcc.begin(), xAcc.end(), 0.0F) / xAcc.size());
			cy.push_back(std::accumulate(yAcc.begin(), yAcc.end(), 0.0F) / yAcc.size());
		}
	}

	if(smooth && cx.size() >= 3) {
		for(size_t i = 1; i + 1 < cx.size(); ++i) {
			cx[i] = (cx[i - 1] + cx[i] + cx[i + 1]) / 3.0F;
		}
	}

	return {cy, cx};
}

bool LaneCurveFitter::isStraightLine(const std::vector<float> &y, const std::vector<float> &x,
                                     float threshold) {
	if(x.size() < 4)
		return false;

	float mean_x = std::accumulate(x.begin(), x.end(), 0.0F) / x.size();
	float mean_y = std::accumulate(y.begin(), y.end(), 0.0F) / y.size();

	float num = 0.0F, den_x = 0.0F, den_y = 0.0F;
	for(size_t i = 0; i < x.size(); ++i) {
		num += (x[i] - mean_x) * (y[i] - mean_y);
		den_x += (x[i] - mean_x) * (x[i] - mean_x);
		den_y += (y[i] - mean_y) * (y[i] - mean_y);
	}

	float corr = num / std::sqrt(den_x * den_y + 1e-6F);
	return std::abs(corr) > threshold;
}

bool LaneCurveFitter::hasSignFlip(const std::vector<float> &xVals) {
	std::vector<float> dx2(xVals.size());
	for(size_t i = 1; i + 1 < xVals.size(); ++i)
		dx2[i] = xVals[i + 1] + xVals[i - 1] - 2 * xVals[i];

	for(size_t i = 1; i < dx2.size(); ++i)
		if((dx2[i] > 0) != (dx2[i - 1] > 0))
			return true;
	return false;
}

std::vector<float> LaneCurveFitter::polyfit(const std::vector<float> &x,
                                           const std::vector<float> &y, int degree) {
	int n = x.size();
	int m = degree + 1;

	cv::Mat A(n, m, CV_32F);
	cv::Mat B(n, 1, CV_32F);

	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			A.at<float>(i, j) = std::pow(x[i], j);
		}
		B.at<float>(i, 0) = y[i];
	}

	cv::Mat coeffs;
	if(!cv::solve(A, B, coeffs, cv::DECOMP_SVD)) {
		return std::vector<float>(m, 0.0F);
	}

	std::vector<float> result(m);
	for(int i = 0; i < m; i++) {
		result[m - 1 - i] = coeffs.at<float>(i, 0);
	}
	return result;
}

std::vector<float> LaneCurveFitter::polyval(const std::vector<float> &coeffs,
                                           const std::vector<float> &x) {
	std::vector<float> result(x.size());
	int degree = coeffs.size() - 1;

	for(size_t i = 0; i < x.size(); i++) {
		float val = 0;
		for(int j = 0; j <= degree; j++) {
			val += coeffs[j] * std::pow(x[i], degree - j);
		}
		result[i] = val;
	}
	return result;
}

std::vector<float> LaneCurveFitter::linspace(float start, float end, int num) {
	std::vector<float> result(num);
	float step = (end - start) / (num - 1);
	for(int i = 0; i < num; i++) {
		result[i] = start + i * step;
	}
	return result;
}

std::vector<float> LaneCurveFitter::interp(const std::vector<float> &xNew,
                                          const std::vector<float> &x,
                                          const std::vector<float> &y, 
                                          float leftVal, float rightVal) {
	std::vector<float> result(xNew.size());

	for(size_t i = 0; i < xNew.size(); i++) {
		float xi = xNew[i];

		if(xi <= x[0]) {
			result[i] = leftVal;
		} else if(xi >= x.back()) {
			result[i] = rightVal;
		} else {
			for(size_t j = 0; j < x.size() - 1; j++) {
				if(xi >= x[j] && xi <= x[j + 1]) {
					float t = (xi - x[j]) / (x[j + 1] - x[j]);
					result[i] = y[j] + t * (y[j + 1] - y[j]);
					break;
				}
			}
		}
	}
	return result;
}

std::pair<LaneCurveFitter::LaneCurve*, LaneCurveFitter::LaneCurve*>
LaneCurveFitter::selectRelevantLanes(std::vector<LaneCurve> &lanes, int imgWidth, int imgHeight) {
	float imgCenter = imgWidth / 2.0F;
	LaneCurve *leftLane = nullptr;
	LaneCurve *rightLane = nullptr;

	std::vector<std::pair<float, LaneCurve*>> laneInfos;

	for(auto &lane : lanes) {
		std::vector<float> bottomHalfX;
		for(const auto &point : lane.curve) {
			if(point.y >= imgHeight / 6.0F) {
				bottomHalfX.push_back(point.x);
			}
		}

		if(!bottomHalfX.empty()) {
			float avgX = std::accumulate(bottomHalfX.begin(), bottomHalfX.end(), 0.0F) / bottomHalfX.size();
			laneInfos.push_back({avgX, &lane});
		}
	}

	std::sort(laneInfos.begin(), laneInfos.end());

	for(const auto &[avgX, lane] : laneInfos) {
		if(avgX < imgCenter) {
			leftLane = lane;
		} else if(avgX >= imgCenter && rightLane == nullptr) {
			rightLane = lane;
			break;
		}
	}

	return {leftLane, rightLane};
}

std::vector<float> LaneCurveFitter::fitCurve(const std::vector<float> &y,
                                            const std::vector<float> &x,
                                            const std::vector<float> &yEval) {
	if(y.size() < 3 || x.size() < 3) {
		std::vector<float> fallback(yEval.size(), x.empty() ? 0.0F : x[0]);
		return fallback;
	}

	// Check if it's a straight line
	if(isStraightLine(y, x)) {
		auto coeffs = polyfit(y, x, 1);
		return polyval(coeffs, yEval);
	}

	// Try quadratic fit
	auto coeffs = polyfit(y, x, 2);
	if(coeffs.size() >= 3 && std::abs(coeffs[0]) > CURVE_THRESHOLD && x.size() >= 4) {
		// Use higher degree polynomial for complex curves
		auto splineCoeffs = polyfit(y, x, std::min(3, (int)x.size() - 1));
		return polyval(splineCoeffs, yEval);
	}

	return polyval(coeffs, yEval);
}

std::vector<LaneCurveFitter::LaneCurve> LaneCurveFitter::fitLanes(const cv::Mat &binaryMask) {
	std::vector<LaneCurve> lanes;
	auto points = extractLanePoints(binaryMask);

	std::vector<int> uniqueLabels;
	auto labels = dbscanCluster(points, uniqueLabels);

	for(int label : uniqueLabels) {
		std::vector<cv::Point> cluster;
		for(size_t i = 0; i < labels.size(); ++i)
			if(labels[i] == label)
				cluster.push_back(points[i]);

		auto [cy, cx] = slidingWindowCentroids(cluster, binaryMask.size(), false);
		if(cy.size() < 2)
			continue;

		std::vector<size_t> sortIdx(cy.size());
		std::iota(sortIdx.begin(), sortIdx.end(), 0);
		std::sort(sortIdx.begin(), sortIdx.end(),
		          [&](size_t i, size_t j) { return cy[i] < cy[j]; });

		std::vector<float> y_sorted, x_sorted;
		for(auto i : sortIdx) {
			y_sorted.push_back(cy[i]);
			x_sorted.push_back(cx[i]);
		}

		auto testCurve = fitCurve(y_sorted, x_sorted, y_sorted);
		if(hasSignFlip(testCurve)) {
			std::tie(cy, cx) = slidingWindowCentroids(cluster, binaryMask.size(), true);
			sortIdx = std::vector<size_t>(cy.size());
			std::iota(sortIdx.begin(), sortIdx.end(), 0);
			std::sort(sortIdx.begin(), sortIdx.end(),
			          [&](size_t i, size_t j) { return cy[i] < cy[j]; });

			y_sorted.clear();
			x_sorted.clear();
			for(auto i : sortIdx) {
				y_sorted.push_back(cy[i]);
				x_sorted.push_back(cx[i]);
			}
		}

		float y_min = *std::min_element(y_sorted.begin(), y_sorted.end());
		float y_max = *std::max_element(y_sorted.begin(), y_sorted.end());
		
		auto y_plot = linspace(std::max(0.0F, y_min - 30), 
		                      std::min((float)binaryMask.rows, y_max + 10), 300);
		auto x_plot = fitCurve(y_sorted, x_sorted, y_plot);

		std::vector<cv::Point2f> curve, cents;
		for(size_t i = 0; i < y_plot.size(); ++i)
			curve.emplace_back(x_plot[i], y_plot[i]);
		for(size_t i = 0; i < x_sorted.size(); ++i)
			cents.emplace_back(x_sorted[i], y_sorted[i]);

		lanes.push_back({cents, curve});
	}

	return lanes;
}

std::optional<LaneCurveFitter::CenterlineResult>
LaneCurveFitter::computeVirtualCenterline(const std::vector<LaneCurve> &lanes, int imgWidth,
                                          int imgHeight) {
	auto lanesRef = const_cast<std::vector<LaneCurve>&>(lanes);
	auto [leftLane, rightLane] = selectRelevantLanes(lanesRef, imgWidth, imgHeight);
	float carX = imgWidth / 2.0F;

	CenterlineResult result;

	if(leftLane && rightLane) {
		// Midpoint method
		std::vector<float> xLeft, yLeft, xRight, yRight;
		for(const auto &point : leftLane->curve) {
			xLeft.push_back(point.x);
			yLeft.push_back(point.y);
		}
		for(const auto &point : rightLane->curve) {
			xRight.push_back(point.x);
			yRight.push_back(point.y);
		}

		float yMin = std::max(*std::min_element(yLeft.begin(), yLeft.end()),
		                     *std::min_element(yRight.begin(), yRight.end()));
		float yStart = imgHeight - 1;
		auto yCommon = linspace(yStart, yMin, 300);

		auto xLeftInterp = interp(yCommon, yLeft, xLeft, xLeft[0], xLeft.back());
		auto xRightInterp = interp(yCommon, yRight, xRight, xRight[0], xRight.back());

		for(size_t i = 0; i < yCommon.size(); i++) {
			float xMid = (xLeftInterp[i] + xRightInterp[i]) / 2.0F;
			float w = static_cast<float>(i) / (yCommon.size() - 1);
			float xBlend = w * xMid + (1 - w) * carX;

			result.blended.emplace_back(xBlend, yCommon[i]);
			result.midpoint.emplace_back(xMid, yCommon[i]);
			result.straight.emplace_back(carX, yCommon[i]);
		}
		result.valid = true;

	} else if(leftLane || rightLane) {
		// Offset method
		LaneCurve *lane = leftLane ? leftLane : rightLane;
		float direction = leftLane ? 1.0F : -1.0F;

		std::vector<float> xLane, yLane;
		for(const auto &point : lane->curve) {
			xLane.push_back(point.x);
			yLane.push_back(point.y);
		}

		float yMin = *std::min_element(yLane.begin(), yLane.end());
		float yStart = imgHeight - 1;
		auto yCommon = linspace(yStart, yMin, 300);

		std::vector<float> xOffset;
		for(float x : xLane) {
			xOffset.push_back(x + direction * laneWidthPx / 2.0F);
		}

		auto xOffsetInterp = interp(yCommon, yLane, xOffset, xOffset[0], xOffset.back());

		for(size_t i = 0; i < yCommon.size(); i++) {
			float w = static_cast<float>(i) / (yCommon.size() - 1);
			float xBlend = w * xOffsetInterp[i] + (1 - w) * carX;

			result.blended.emplace_back(xBlend, yCommon[i]);
			result.midpoint.emplace_back(xOffsetInterp[i], yCommon[i]);
			result.straight.emplace_back(carX, yCommon[i]);
		}
		result.valid = true;
	}

	return result.valid ? std::make_optional(result) : std::nullopt;
}
