#include "../../includes/inference/LaneCurveFitter.hpp"
#include <numeric>
#include <cmath>
#include <map>
#include <set>

LaneCurveFitter::LaneCurveFitter(float eps, int minSamples, int windows, int laneWidthPx)
	: dbscanEps(eps), dbscanMinSamples(minSamples), numWindows(windows), laneWidthPx(laneWidthPx) {}

std::vector<cv::Point> LaneCurveFitter::extractLanePoints(const cv::Mat& binaryMask) {
	std::vector<cv::Point> points;
	for (int y = 0; y < binaryMask.rows; ++y) {
		for (int x = 0; x < binaryMask.cols; ++x) {
			if (binaryMask.at<uchar>(y, x) > 0)
				points.emplace_back(x, y);
		}
	}
	return points;
}

float interpolateXatY(const std::vector<cv::Point2f>& points, float y_query) {
	if (points.empty()) return 0.0f;

	for (size_t i = 1; i < points.size(); ++i) {
		float y1 = points[i - 1].y;
		float y2 = points[i].y;

		if ((y1 <= y_query && y_query <= y2) || (y2 <= y_query && y_query <= y1)) {
			float t = (y_query - y1) / (y2 - y1 + 1e-6f);
			float x1 = points[i - 1].x;
			float x2 = points[i].x;
			return x1 + t * (x2 - x1);
		}
	}

	// Extrapolate if y_query is outside range
	return points.back().x;
}

// Simple DBSCAN implementation (brute force)
std::vector<int> LaneCurveFitter::dbscanCluster(const std::vector<cv::Point>& points, std::vector<int>& uniqueLabels) {
	const int n = points.size();
	std::vector<int> labels(n, -1);
	int clusterId = 0;

	for (int i = 0; i < n; ++i) {
		if (labels[i] != -1) continue;

		std::vector<int> neighbors;
		for (int j = 0; j < n; ++j) {
			if (cv::norm(points[i] - points[j]) <= dbscanEps)
				neighbors.push_back(j);
		}

		if (neighbors.size() < dbscanMinSamples)
			continue;

		labels[i] = clusterId;
		std::set<int> seeds(neighbors.begin(), neighbors.end());
		seeds.erase(i);

		while (!seeds.empty()) {
			int current = *seeds.begin();
			seeds.erase(seeds.begin());

			if (labels[current] == -1) {
				labels[current] = clusterId;

				std::vector<int> currentNeighbors;
				for (int j = 0; j < n; ++j) {
					if (cv::norm(points[current] - points[j]) <= dbscanEps)
						currentNeighbors.push_back(j);
				}

				if (currentNeighbors.size() >= dbscanMinSamples) {
					seeds.insert(currentNeighbors.begin(), currentNeighbors.end());
				}
			}
		}

		++clusterId;
	}

	uniqueLabels.clear();
	for (int l : labels)
		if (l != -1)
			uniqueLabels.push_back(l);
	std::sort(uniqueLabels.begin(), uniqueLabels.end());
	uniqueLabels.erase(std::unique(uniqueLabels.begin(), uniqueLabels.end()), uniqueLabels.end());

	return labels;
}

std::pair<std::vector<float>, std::vector<float>> LaneCurveFitter::slidingWindowCentroids(const std::vector<cv::Point>& cluster, cv::Size imgSize, bool smooth) {
	std::vector<float> cx, cy;
	int h = imgSize.height / numWindows;

	for (int i = 0; i < numWindows; ++i) {
		int yLow = imgSize.height - (i + 1) * h;
		int yHigh = imgSize.height - i * h;

		std::vector<float> xAcc, yAcc;
		for (const auto& pt : cluster) {
			if (pt.y >= yLow && pt.y < yHigh) {
				xAcc.push_back(pt.x);
				yAcc.push_back(pt.y);
			}
		}

		if (!xAcc.empty()) {
			cx.push_back(std::accumulate(xAcc.begin(), xAcc.end(), 0.0f) / xAcc.size());
			cy.push_back(std::accumulate(yAcc.begin(), yAcc.end(), 0.0f) / yAcc.size());
		}
	}

	if (smooth && cx.size() >= 3) {
		for (size_t i = 1; i + 1 < cx.size(); ++i) {
			cx[i] = (cx[i - 1] + cx[i] + cx[i + 1]) / 3.0f;
		}
	}

	return {cy, cx};
}

bool LaneCurveFitter::isStraightLine(const std::vector<float>& y, const std::vector<float>& x, float threshold) {
	if (x.size() < 4) return false;

	float mean_x = std::accumulate(x.begin(), x.end(), 0.0f) / x.size();
	float mean_y = std::accumulate(y.begin(), y.end(), 0.0f) / y.size();

	float num = 0.0f, den_x = 0.0f, den_y = 0.0f;
	for (size_t i = 0; i < x.size(); ++i) {
		num += (x[i] - mean_x) * (y[i] - mean_y);
		den_x += (x[i] - mean_x) * (x[i] - mean_x);
		den_y += (y[i] - mean_y) * (y[i] - mean_y);
	}

	float corr = num / std::sqrt(den_x * den_y + 1e-6f);
	return std::abs(corr) > threshold;
}

bool LaneCurveFitter::hasSignFlip(const std::vector<float>& xVals) {
	std::vector<float> dx2(xVals.size());
	for (size_t i = 1; i + 1 < xVals.size(); ++i)
		dx2[i] = xVals[i + 1] + xVals[i - 1] - 2 * xVals[i];

	for (size_t i = 1; i < dx2.size(); ++i)
		if ((dx2[i] > 0) != (dx2[i - 1] > 0))
			return true;
	return false;
}

std::vector<float> LaneCurveFitter::fitCurve(const std::vector<float>& y, const std::vector<float>& x, const std::vector<float>& yEval) {
	if (y.size() < 3 || x.size() < 3) {
		// Fallback: return a straight horizontal line
		std::vector<float> fallback(yEval.size(), x.empty() ? 0.0f : x[0]);
		return fallback;
	}

	cv::Mat A(y.size(), 3, CV_32F);
	cv::Mat X(x);

	for (size_t i = 0; i < y.size(); ++i) {
		A.at<float>(i, 0) = y[i] * y[i];
		A.at<float>(i, 1) = y[i];
		A.at<float>(i, 2) = 1.0f;
	}

	cv::Mat coeffs;
	if (!cv::solve(A, X, coeffs, cv::DECOMP_SVD)) {
		// Fallback in case of failure
		std::vector<float> fallback(yEval.size(), x[0]);
		return fallback;
	}

	std::vector<float> result;
	for (float yv : yEval) {
		result.push_back(coeffs.at<float>(0) * yv * yv + coeffs.at<float>(1) * yv + coeffs.at<float>(2));
	}
	return result;
}


std::vector<LaneCurveFitter::LaneCurve> LaneCurveFitter::fitLanes(const cv::Mat& binaryMask) {
	std::vector<LaneCurve> lanes;
	auto points = extractLanePoints(binaryMask);

	std::vector<int> uniqueLabels;
	auto labels = dbscanCluster(points, uniqueLabels);

	for (int label : uniqueLabels) {
		std::vector<cv::Point> cluster;
		for (size_t i = 0; i < labels.size(); ++i)
			if (labels[i] == label)
				cluster.push_back(points[i]);

		auto [cy, cx] = slidingWindowCentroids(cluster, binaryMask.size(), false);
		if (cy.size() < 2) continue;

		std::vector<size_t> sortIdx(cy.size());
		std::iota(sortIdx.begin(), sortIdx.end(), 0);
		std::sort(sortIdx.begin(), sortIdx.end(), [&](size_t i, size_t j) { return cy[i] < cy[j]; });

		std::vector<float> y_sorted, x_sorted;
		for (auto i : sortIdx) {
			y_sorted.push_back(cy[i]);
			x_sorted.push_back(cx[i]);
		}

		auto testCurve = fitCurve(y_sorted, x_sorted, y_sorted);
		if (hasSignFlip(testCurve)) {
			std::tie(cy, cx) = slidingWindowCentroids(cluster, binaryMask.size(), true);
			sortIdx = std::vector<size_t>(cy.size());
			std::iota(sortIdx.begin(), sortIdx.end(), 0);
			std::sort(sortIdx.begin(), sortIdx.end(), [&](size_t i, size_t j) { return cy[i] < cy[j]; });

			y_sorted.clear(); x_sorted.clear();
			for (auto i : sortIdx) {
				y_sorted.push_back(cy[i]);
				x_sorted.push_back(cx[i]);
			}
		}

		float y_min = *std::min_element(y_sorted.begin(), y_sorted.end());
		float y_max = *std::max_element(y_sorted.begin(), y_sorted.end());
		std::vector<float> y_plot(300);
		float step = (y_max + 10 - (y_min - 30)) / 300.0f;
		for (int i = 0; i < 300; ++i)
			y_plot[i] = y_max + 10 - i * step;

		std::vector<float> x_plot = fitCurve(y_sorted, x_sorted, y_plot);

		std::vector<cv::Point2f> curve, cents;
		for (size_t i = 0; i < y_plot.size(); ++i)
			curve.emplace_back(x_plot[i], y_plot[i]);
		for (size_t i = 0; i < x_sorted.size(); ++i)
			cents.emplace_back(x_sorted[i], y_sorted[i]);

		lanes.push_back({cents, curve});
	}

	return lanes;
}

std::optional<LaneCurveFitter::CenterlineResult> LaneCurveFitter::computeVirtualCenterline(const std::vector<LaneCurve>& lanes, int imgWidth, int imgHeight) {
	const float centerX = imgWidth / 2.0f;
	LaneCurve left, right;
	std::vector<std::pair<float, LaneCurve>> candidates;

	for (const auto& lane : lanes) {
		std::vector<float> bottomXs;
		for (const auto& pt : lane.curve)
			if (pt.y >= imgHeight / 2) bottomXs.push_back(pt.x);
		if (bottomXs.empty()) continue;

		float avgX = std::accumulate(bottomXs.begin(), bottomXs.end(), 0.0f) / bottomXs.size();
		candidates.emplace_back(avgX, lane);
	}

	std::sort(candidates.begin(), candidates.end(), [](auto& a, auto& b) { return a.first < b.first; });

	for (const auto& [avgX, lane] : candidates) {
		if (avgX < centerX)
			left = lane;
		else if (!right.curve.size())
			right = lane;
	}

	std::vector<cv::Point2f> c1, c2, blended;
	if (!left.curve.empty() && !right.curve.empty()) {
		std::vector<float> y_common(300);
		float y_start = imgHeight - 1;
		float y_end = std::max(left.curve.back().y, right.curve.back().y);
		float dy = (y_start - y_end) / 299.0f;
		for (int i = 0; i < 300; ++i)
			y_common[i] = y_start - i * dy;

		std::vector<float> xl(300), xr(300);
		for (int i = 0; i < 300; ++i) {
			xl[i] = interpolateXatY(left.curve, y_common[i]);
			xr[i] = interpolateXatY(right.curve, y_common[i]);
		}

		for (int i = 0; i < 300; ++i) {
			float mid = (xl[i] + xr[i]) / 2.0f;
			float w = static_cast<float>(i) / 299.0f;
			float blendX = w * mid + (1 - w) * centerX;

			c1.emplace_back(mid, y_common[i]);
			c2.emplace_back(centerX, y_common[i]);
			blended.emplace_back(blendX, y_common[i]);
		}
		return CenterlineResult{blended, c1, c2};
	}

	return std::nullopt;
}
