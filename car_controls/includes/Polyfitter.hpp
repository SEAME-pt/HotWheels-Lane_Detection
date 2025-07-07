#ifndef POLYFITTER_HPP
#define POLYFITTER_HPP

#include "CommonTypes.hpp"
#include "Debugger.hpp"
#include "Publisher.hpp"
#include "Subscriber.hpp"
#include <NvInfer.h>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <experimental/filesystem>
#include <iostream>
#include <map>
#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <numeric>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

// === ESTRUTURAS MELHORADAS ===
struct Lane {
		std::vector<Point2D> centroids;
		std::vector<Point2D> curve;
};

struct CenterlineResult {
		std::vector<Point2D> blend;
		std::vector<Point2D> c1;
		std::vector<Point2D> c2;
		bool valid;

		CenterlineResult() : valid(false) {}
};

class Polyfitter {
	private:
		// === PARÂMETROS OTIMIZADOS (ALTERADOS) ===
		static constexpr double EPS = 5.0;     // Era 8.0
		static constexpr int MIN_SAMPLES = 5;  // Era 15
		static constexpr int NUM_WINDOWS = 40; // Era 25
		static constexpr double STRAIGHT_LINE_THRESHOLD = 0.98;
		static constexpr double CURVE_THRESHOLD = 0.0012;
		static constexpr int LANE_WIDTH_PX = 300; // Era 120

		
		private:
		// === NOVO: Método de publicação ===
		void publishLaneData(const LaneInfo &laneInfo, const cv::Mat &binaryMask);
		std::string serializeLaneInfo(const LaneInfo &laneInfo);
		
		public:
		bool m_zeromq_enabled;
		Publisher *m_publisherLaneData;
		Polyfitter();
		~Polyfitter();

		// === MÉTODOS EXISTENTES (MANTIDOS) ===
		std::vector<std::pair<std::string, cv::Mat>>
		loadImagesFromFolder(const std::string &folderPath);
		std::vector<Point2D> extractLanePoints(const cv::Mat &img);
		std::pair<std::vector<int>, std::vector<int>>
		clusterLanePoints(const std::vector<Point2D> &pts);
		std::pair<std::vector<double>, std::vector<double>>
		slidingWindowCentroids(const std::vector<double> &x, const std::vector<double> &y,
		                       const cv::Size &imgShape, bool smooth = false);
		std::vector<double> polyfit(const std::vector<double> &x, const std::vector<double> &y,
		                            int degree) const;
		std::vector<double> polyval(const std::vector<double> &coeffs,
		                            const std::vector<double> &x);
		std::vector<double> fitLaneCurve(const std::vector<double> &y, const std::vector<double> &x,
		                                 int imgWidth, const std::vector<double> &yPlot);
		void displayImagesWithPolyfit(const std::vector<std::pair<std::string, cv::Mat>> &images,
		                              int cols = 4);
		void enableZeroMQPublishing(bool enable = true);
		void setZeroMQPort(int port);

		// === MÉTODOS NOVOS (ADICIONADOS) ===
		bool hasSignFlip(const std::vector<double> &curve);
		std::pair<Lane *, Lane *> selectRelevantLanes(std::vector<Lane> &lanes, int imgWidth,
		                                              int imgHeight);
		CenterlineResult computeVirtualCenterline(std::vector<Lane> &lanes, int imgWidth,
		                                          int imgHeight);
		std::vector<Lane> fitLanesInImage(const cv::Mat &img);
		std::vector<double> linspace(double start, double end, int num);
		std::vector<double> interp(const std::vector<double> &xNew, const std::vector<double> &x,
		                           const std::vector<double> &y, double leftVal, double rightVal);

		// === MÉTODOS MELHORADOS (OTIMIZADOS) ===
		bool isStraightLine(const std::vector<double> &y, const std::vector<double> &x) const;

		// === MÉTODOS PARA MPC (MANTIDOS) ===
		double calculateCTE(const std::vector<double> &polyCoeffs, double x, double y) const;
		double calculateEPSI(const std::vector<double> &polyCoeffs, double x, double psi) const;
		std::vector<double> getPolynomialCoeffs(const std::vector<Point2D> &trajectory) const;
		std::vector<Point2D> convertImagePointsToWorld(const std::vector<int> &center_x,
		                                               const std::vector<int> &center_y,
		                                               const VehicleTransform &vehicle_transform,
		                                               int img_width, int img_height) const;

		// === MÉTODO PRINCIPAL MELHORADO ===
		LaneInfo processFrame(const cv::Mat &mask);
};

#endif // POLYFITTER_HPP
