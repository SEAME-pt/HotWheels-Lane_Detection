#include "Polyfitter.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <filesystem>
#include <map>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <mlpack/core.hpp>

namespace fs = std::filesystem;

Point2D::Point2D(double x, double y) : x(x), y(y) {}
CenterlineResult::CenterlineResult() : valid(false) {}
Polyfitter::Polyfitter() {}
Polyfitter::~Polyfitter() {}

std::vector<std::pair<std::string, cv::Mat>> Polyfitter::loadImagesFromFolder(const std::string& folderPath) {
	std::vector<std::pair<std::string, cv::Mat>> images;
	std::vector<std::string> extensions = {".png", ".jpg", ".jpeg"};
	
	if (!fs::exists(folderPath)) {
		std::cerr << "Folder does not exist: " << folderPath << std::endl;
		return images;
	}
	
	std::vector<std::string> filenames;
	for (const auto& entry : fs::directory_iterator(folderPath)) {
		if (entry.is_regular_file()) {
			std::string filename = entry.path().filename().string();
			std::string ext = entry.path().extension().string();
			std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
			
			if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
				filenames.push_back(filename);
			}
		}
	}
	
	std::sort(filenames.begin(), filenames.end());
	
	for (const auto& filename : filenames) {
		std::string filepath = folderPath + "/" + filename;
		cv::Mat img = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
		if (!img.empty()) {
			images.push_back({filename, img});
		}
	}
	
	return images;
}

std::vector<Point2D> Polyfitter::extractLanePoints(const cv::Mat& img) {
	std::vector<Point2D> points;
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			if (img.at<uchar>(y, x) > 0) {
				points.push_back(Point2D(x, y));
			}
		}
	}
	return points;
}


std::pair<std::vector<int>, std::vector<int>> Polyfitter::clusterLanePoints(
	const std::vector<Point2D>& pts) 
{
	const size_t N = pts.size();
	if (N == 0) return {{}, {}};

	// 2 x N matrix: each column is a point [x; y]
	arma::mat dataset(2, N);
	for (size_t i = 0; i < N; ++i) {
		dataset(0, i) = pts[i].x;
		dataset(1, i) = pts[i].y;
	}

	// Output labels and core point flags
	arma::Row<size_t> labels;
	mlpack::dbscan::DBSCAN<> db(EPS, MIN_SAMPLES);
	db.Cluster(dataset, labels);

	// Convert to int and gather unique cluster IDs != SIZE_MAX
	std::vector<int> intLabels(N);
	std::set<int> uniqueIds;
	for (size_t i = 0; i < N; ++i) {
		if (labels[i] == SIZE_MAX) {
			intLabels[i] = -1;
			} else {
			intLabels[i] = (int) labels[i];
			uniqueIds.insert(intLabels[i]);
			}

	}

	std::vector<int> uniques(uniqueIds.begin(), uniqueIds.end());
	return {intLabels, uniques};
}

std::pair<std::vector<double>, std::vector<double>> Polyfitter::slidingWindowCentroids(
	const std::vector<double>& x, const std::vector<double>& y, 
	const cv::Size& imgShape, bool smooth = false) {
	
	int h = imgShape.height / NUM_WINDOWS;
	std::vector<double> cx, cy;
	
	for (int i = 0; i < NUM_WINDOWS; i++) {
		int yLow = imgShape.height - (i + 1) * h;
		int yHigh = imgShape.height - i * h;
		
		std::vector<double> windowX;
		std::vector<double> windowY;
		
		for (size_t j = 0; j < y.size(); j++) {
			if (y[j] >= yLow && y[j] < yHigh) {
				windowX.push_back(x[j]);
				windowY.push_back(y[j]);
			}
		}
		
		if (!windowX.empty()) {
			double meanX = std::accumulate(windowX.begin(), windowX.end(), 0.0) / windowX.size();
			double meanY = std::accumulate(windowY.begin(), windowY.end(), 0.0) / windowY.size();
			cx.push_back(meanX);
			cy.push_back(meanY);
		}
	}
	
	if (smooth && cx.size() >= 3) {
		std::vector<double> smoothedCx = cx;
		for (size_t i = 1; i < cx.size() - 1; i++) {
			smoothedCx[i] = (cx[i-1] + cx[i] + cx[i+1]) / 3.0;
		}
		cx = smoothedCx;
	}
	
	return {cy, cx};
}

bool Polyfitter::hasSignFlip(const std::vector<double>& curve) {
	if (curve.size() < 3) return false;
	
	std::vector<double> firstDeriv(curve.size() - 1);
	for (size_t i = 0; i < firstDeriv.size(); i++) {
		firstDeriv[i] = curve[i+1] - curve[i];
	}
	
	std::vector<double> secondDeriv(firstDeriv.size() - 1);
	for (size_t i = 0; i < secondDeriv.size(); i++) {
		secondDeriv[i] = firstDeriv[i+1] - firstDeriv[i];
	}
	
	for (size_t i = 1; i < secondDeriv.size(); i++) {
		if ((secondDeriv[i] > 0) != (secondDeriv[i-1] > 0)) {
			return true;
		}
	}
	
	return false;
}

bool Polyfitter::isStraightLine(const std::vector<double>& y, const std::vector<double>& x) {
	if (x.size() < 4) return false;
	
	double meanX = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
	double meanY = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
	
	double num = 0, denX = 0, denY = 0;
	for (size_t i = 0; i < x.size(); i++) {
		double dx = x[i] - meanX;
		double dy = y[i] - meanY;
		num += dx * dy;
		denX += dx * dx;
		denY += dy * dy;
	}
	
	if (denX == 0 || denY == 0) return true;
	double corr = num / std::sqrt(denX * denY);
	return std::abs(corr) > STRAIGHT_LINE_THRESHOLD;
}

std::vector<double> Polyfitter::polyfit(const std::vector<double>& x, const std::vector<double>& y, int degree) {
	int n = x.size();
	int m = degree + 1;
	
	cv::Mat A(n, m, CV_64F);
	cv::Mat B(n, 1, CV_64F);
	
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			A.at<double>(i, j) = std::pow(x[i], j);
		}
		B.at<double>(i, 0) = y[i];
	}
	
	cv::Mat coeffs;
	cv::solve(A, B, coeffs, cv::DECOMP_SVD);
	
	std::vector<double> result(m);
	for (int i = 0; i < m; i++) {
		result[m-1-i] = coeffs.at<double>(i, 0);
	}
	
	return result;
}

std::vector<double> Polyfitter::polyval(const std::vector<double>& coeffs, const std::vector<double>& x) {
	std::vector<double> result(x.size());
	int degree = coeffs.size() - 1;
	
	for (size_t i = 0; i < x.size(); i++) {
		double val = 0;
		for (int j = 0; j <= degree; j++) {
			val += coeffs[j] * std::pow(x[i], degree - j);
		}
		result[i] = val;
	}
	
	return result;
}

std::vector<double> Polyfitter::fitLaneCurve(const std::vector<double>& y, const std::vector<double>& x, 
								int imgWidth, const std::vector<double>& yPlot) {
	if (isStraightLine(y, x)) {
		auto coeffs = polyfit(y, x, 1);
		return polyval(coeffs, yPlot);
	}
	
	auto coeffs = polyfit(y, x, 2);
	double a = coeffs[0];
	
	if (std::abs(a) > CURVE_THRESHOLD && x.size() >= 4) {
		// Simple spline approximation using higher degree polynomial
		auto splineCoeffs = polyfit(y, x, std::min(3, (int)x.size() - 1));
		return polyval(splineCoeffs, yPlot);
	}
	
	return polyval(coeffs, yPlot);
}

std::vector<Lane> Polyfitter::fitLanesInImage(const cv::Mat& img) {
	auto points = extractLanePoints(img);
	auto [labels, uniqueLabels] = clusterLanePoints(points);
	
	std::vector<Lane> lanes;
	
	for (int label : uniqueLabels) {
		std::vector<double> x, y;
		for (size_t i = 0; i < points.size(); i++) {
			if (labels[i] == label) {
				x.push_back(points[i].x);
				y.push_back(points[i].y);
			}
		}
		
		auto [centY, centX] = slidingWindowCentroids(x, y, img.size(), false);
		if (centY.size() < 2) continue;
		
		// Sort by y coordinate
		std::vector<size_t> indices(centY.size());
		std::iota(indices.begin(), indices.end(), 0);
		std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
			return centY[a] < centY[b];
		});
		
		std::vector<double> sortedCentY, sortedCentX;
		for (size_t idx : indices) {
			sortedCentY.push_back(centY[idx]);
			sortedCentX.push_back(centX[idx]);
		}
		
		// Check for sign flip
		try {
			auto testCoeffs = polyfit(sortedCentY, sortedCentX, 2);
			auto testCurve = polyval(testCoeffs, sortedCentY);
			if (hasSignFlip(testCurve)) {
				auto [newCentY, newCentX] = slidingWindowCentroids(x, y, img.size(), true);
				std::vector<size_t> newIndices(newCentY.size());
				std::iota(newIndices.begin(), newIndices.end(), 0);
				std::sort(newIndices.begin(), newIndices.end(), [&](size_t a, size_t b) {
					return newCentY[a] < newCentY[b];
				});
				
				sortedCentY.clear();
				sortedCentX.clear();
				for (size_t idx : newIndices) {
					sortedCentY.push_back(newCentY[idx]);
					sortedCentX.push_back(newCentX[idx]);
				}
			}
		} catch (...) {
			continue;
		}
		
		double yMin = *std::min_element(sortedCentY.begin(), sortedCentY.end());
		double yMax = *std::max_element(sortedCentY.begin(), sortedCentY.end());
		
		std::vector<double> yPlot;
		double yStart = std::max(0.0, yMin - 30);
		double yEnd = std::min((double)img.rows, yMax + 10);
		
		for (int i = 0; i < 300; i++) {
			yPlot.push_back(yStart + (yEnd - yStart) * i / 299.0);
		}
		
		auto xPlot = fitLaneCurve(sortedCentY, sortedCentX, img.cols, yPlot);
		
		Lane lane;
		for (size_t i = 0; i < sortedCentX.size(); i++) {
			lane.centroids.push_back(Point2D(sortedCentX[i], sortedCentY[i]));
		}
		for (size_t i = 0; i < xPlot.size(); i++) {
			lane.curve.push_back(Point2D(xPlot[i], yPlot[i]));
		}
		
		lanes.push_back(lane);
	}
	
	return lanes;
}

std::pair<Lane*, Lane*> Polyfitter::selectRelevantLanes(std::vector<Lane>& lanes, int imgWidth, int imgHeight) {
	double imgCenter = imgWidth / 2.0;
	Lane* leftLane = nullptr;
	Lane* rightLane = nullptr;
	
	std::vector<std::pair<double, Lane*>> laneInfos;
	
	for (auto& lane : lanes) {
		std::vector<double> bottomHalfX;
		for (const auto& point : lane.curve) {
			if (point.y >= imgHeight / 2.0) {
				bottomHalfX.push_back(point.x);
			}
		}
		
		if (!bottomHalfX.empty()) {
			double avgX = std::accumulate(bottomHalfX.begin(), bottomHalfX.end(), 0.0) / bottomHalfX.size();
			laneInfos.push_back({avgX, &lane});
		}
	}
	
	std::sort(laneInfos.begin(), laneInfos.end());
	
	for (const auto& [avgX, lane] : laneInfos) {
		if (avgX < imgCenter) {
			leftLane = lane;
		} else if (avgX >= imgCenter && rightLane == nullptr) {
			rightLane = lane;
			break;
		}
	}
	
	return {leftLane, rightLane};
}

std::vector<double> Polyfitter::linspace(double start, double end, int num) {
	std::vector<double> result(num);
	double step = (end - start) / (num - 1);
	for (int i = 0; i < num; i++) {
		result[i] = start + i * step;
	}
	return result;
}

std::vector<double> Polyfitter::interp(const std::vector<double>& xNew, const std::vector<double>& x, 
							const std::vector<double>& y, double leftVal, double rightVal) {
	std::vector<double> result(xNew.size());
	
	for (size_t i = 0; i < xNew.size(); i++) {
		double xi = xNew[i];
		
		if (xi <= x[0]) {
			result[i] = leftVal;
		} else if (xi >= x.back()) {
			result[i] = rightVal;
		} else {
			// Linear interpolation
			for (size_t j = 0; j < x.size() - 1; j++) {
				if (xi >= x[j] && xi <= x[j+1]) {
					double t = (xi - x[j]) / (x[j+1] - x[j]);
					result[i] = y[j] + t * (y[j+1] - y[j]);
					break;
				}
			}
		}
	}
	
	return result;
}

CenterlineResult Polyfitter::computeVirtualCenterline(std::vector<Lane>& lanes, int imgWidth, int imgHeight) {
	bool applyBlending = true;
	auto [leftLane, rightLane] = selectRelevantLanes(lanes, imgWidth, imgHeight);
	double carX = imgWidth / 2.0;
	
	CenterlineResult result;
	
	if (leftLane && rightLane) {
		// Midpoint method
		std::vector<double> xLeft, yLeft, xRight, yRight;
		for (const auto& point : leftLane->curve) {
			xLeft.push_back(point.x);
			yLeft.push_back(point.y);
		}
		for (const auto& point : rightLane->curve) {
			xRight.push_back(point.x);
			yRight.push_back(point.y);
		}
		
		double yMin = std::max(*std::min_element(yLeft.begin(), yLeft.end()),
								*std::min_element(yRight.begin(), yRight.end()));
		double yStart = imgHeight - 1;
		auto yCommon = linspace(yStart, yMin, 300);
		
		auto xLeftInterp = interp(yCommon, yLeft, xLeft, xLeft[0], xLeft.back());
		auto xRightInterp = interp(yCommon, yRight, xRight, xRight[0], xRight.back());
		
		std::vector<double> xC1(yCommon.size());
		std::vector<double> xC2(yCommon.size(), carX);
		
		for (size_t i = 0; i < yCommon.size(); i++) {
			xC1[i] = (xLeftInterp[i] + xRightInterp[i]) / 2.0;
		}
		
		if (!applyBlending) {
			for (size_t i = 0; i < yCommon.size(); i++) {
				result.blend.push_back(Point2D(xC1[i], yCommon[i]));
				result.c1.push_back(Point2D(xC1[i], yCommon[i]));
				result.c2.push_back(Point2D(xC2[i], yCommon[i]));
			}
		} else {
			for (size_t i = 0; i < yCommon.size(); i++) {
				double w = (yCommon[0] - yCommon[i]) / (yCommon[0] - yCommon.back());
				double xBlend = w * xC1[i] + (1 - w) * xC2[i];
				
				result.blend.push_back(Point2D(xBlend, yCommon[i]));
				result.c1.push_back(Point2D(xC1[i], yCommon[i]));
				result.c2.push_back(Point2D(xC2[i], yCommon[i]));
			}
		}
		
		result.valid = true;
		
	} else if (leftLane || rightLane) {
		// Offset method
		Lane* lane = leftLane ? leftLane : rightLane;
		double direction = leftLane ? 1.0 : -1.0;
		
		std::vector<double> xLane, yLane;
		for (const auto& point : lane->curve) {
			xLane.push_back(point.x);
			yLane.push_back(point.y);
		}
		
		std::vector<double> xC1;
		for (double x : xLane) {
			xC1.push_back(x + direction * LANE_WIDTH_PX / 2.0);
		}
		
		double yMin = *std::min_element(yLane.begin(), yLane.end());
		double yStart = imgHeight - 1;
		auto yCommon = linspace(yStart, yMin, 300);
		
		auto xC1Interp = interp(yCommon, yLane, xC1, xC1[0], xC1.back());
		std::vector<double> xC2(yCommon.size(), carX);
		
		if (!applyBlending) {
			for (size_t i = 0; i < yCommon.size(); i++) {
				result.blend.push_back(Point2D(xC1Interp[i], yCommon[i]));
				result.c1.push_back(Point2D(xC1Interp[i], yCommon[i]));
				result.c2.push_back(Point2D(xC2[i], yCommon[i]));
			}
		} else {
			for (size_t i = 0; i < yCommon.size(); i++) {
				double w = (yCommon[0] - yCommon[i]) / (yCommon[0] - yCommon.back());
				double xBlend = w * xC1Interp[i] + (1 - w) * xC2[i];
				
				result.blend.push_back(Point2D(xBlend, yCommon[i]));
				result.c1.push_back(Point2D(xC1Interp[i], yCommon[i]));
				result.c2.push_back(Point2D(xC2[i], yCommon[i]));
			}
		}
		
		result.valid = true;
	}
	
	return result;
}

void Polyfitter::displayImagesWithPolyfit(const std::vector<std::pair<std::string, cv::Mat>>& images, int cols = 4) {
	if (images.empty()) return;
	
	int numImages = images.size();
	int rows = (numImages + cols - 1) / cols;
	
	// Calculate individual image display size
	int imgDisplayWidth = 300;
	int imgDisplayHeight = 200;
	
	// Create a large canvas to hold all images
	int canvasWidth = cols * imgDisplayWidth;
	int canvasHeight = rows * imgDisplayHeight;
	cv::Mat canvas = cv::Mat::zeros(canvasHeight, canvasWidth, CV_8UC3);
	
	std::vector<cv::Scalar> colors = {
		cv::Scalar(0, 0, 255),    // Red
		cv::Scalar(255, 0, 0),    // Blue
		cv::Scalar(0, 255, 255),  // Yellow
		cv::Scalar(128, 0, 128),  // Purple
		cv::Scalar(0, 255, 0)     // Green
	};
	
	for (int idx = 0; idx < numImages; idx++) {
		int row = idx / cols;
		int col = idx % cols;
		
		const auto& [filename, img] = images[idx];
		
		// Resize image to fit in the grid
		cv::Mat resizedImg;
		cv::resize(img, resizedImg, cv::Size(imgDisplayWidth, imgDisplayHeight));
		
		// Convert grayscale to color for drawing
		cv::Mat colorImg;
		cv::cvtColor(resizedImg, colorImg, cv::COLOR_GRAY2BGR);
		
		// Calculate scale factors for drawing
		double scaleX = (double)imgDisplayWidth / img.cols;
		double scaleY = (double)imgDisplayHeight / img.rows;
		
		// Fit lanes
		auto lanes = fitLanesInImage(img);
		
		// Draw lane curves
		for (size_t i = 0; i < lanes.size(); i++) {
			const auto& lane = lanes[i];
			cv::Scalar color = colors[i % colors.size()];
			
			// Draw centroids
			for (const auto& centroid : lane.centroids) {
				int x = (int)(centroid.x * scaleX);
				int y = (int)(centroid.y * scaleY);
				if (x >= 0 && x < imgDisplayWidth && y >= 0 && y < imgDisplayHeight) {
					cv::circle(colorImg, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
					cv::circle(colorImg, cv::Point(x, y), 3, cv::Scalar(0, 0, 0), 1);
				}
			}
			
			// Draw curve
			std::vector<cv::Point> curvePoints;
			for (const auto& point : lane.curve) {
				int x = (int)(point.x * scaleX);
				int y = (int)(point.y * scaleY);
				if (x >= 0 && x < imgDisplayWidth && y >= 0 && y < imgDisplayHeight) {
					curvePoints.push_back(cv::Point(x, y));
				}
			}
			
			for (size_t j = 1; j < curvePoints.size(); j++) {
				cv::line(colorImg, curvePoints[j-1], curvePoints[j], color, 2);
			}
		}
		
		// Compute and draw centerline
		auto centerlineResult = computeVirtualCenterline(lanes, img.cols, img.rows);
		if (centerlineResult.valid) {
			// Draw blended centerline
			std::vector<cv::Point> centerlinePoints;
			for (const auto& point : centerlineResult.blend) {
				int x = (int)(point.x * scaleX);
				int y = (int)(point.y * scaleY);
				if (x >= 0 && x < imgDisplayWidth && y >= 0 && y < imgDisplayHeight) {
					centerlinePoints.push_back(cv::Point(x, y));
				}
			}
			
			for (size_t j = 1; j < centerlinePoints.size(); j++) {
				cv::line(colorImg, centerlinePoints[j-1], centerlinePoints[j], cv::Scalar(0, 165, 255), 2); // Orange
			}
		}
		
		// Add title
		cv::putText(colorImg, filename, cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
		
		// Copy to canvas
		int startX = col * imgDisplayWidth;
		int startY = row * imgDisplayHeight;
		cv::Rect roi(startX, startY, imgDisplayWidth, imgDisplayHeight);
		colorImg.copyTo(canvas(roi));
	}
	
	cv::namedWindow("Lane Detection Results", cv::WINDOW_AUTOSIZE);
	cv::imshow("Lane Detection Results", canvas);
	cv::waitKey(0);
	cv::destroyAllWindows();
}