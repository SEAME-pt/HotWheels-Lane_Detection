
#include "Polyfitter.hpp"

namespace fs = std::experimental::filesystem;

Polyfitter::Polyfitter () : m_zeromq_enabled (false), m_publisherLaneData (nullptr) {
	// Inicialização sem ZeroMQ por padrão
}

Polyfitter::~Polyfitter () {
	if (m_publisherLaneData) {
		delete m_publisherLaneData;
		m_publisherLaneData = nullptr;
	}
}

void Polyfitter::enableZeroMQPublishing (bool enable) {
	m_zeromq_enabled = enable;

	if (enable && !m_publisherLaneData) {
		// Create Publisher using public factory method or public constructor
		try {
			m_publisherLaneData = Publisher::instance (5558);
			INFO_LOG ("Polyfitter", "ZeroMQ publishing enabled on port 5558");
		} catch (const std::exception &e) {
			ERROR_STREAM ("Polyfitter") << "Failed to create Publisher: " << e.what ();
			m_publisherLaneData = nullptr;
			m_zeromq_enabled = false;
		}
	} else if (!enable && m_publisherLaneData) {
		delete m_publisherLaneData;
		m_publisherLaneData = nullptr;
		INFO_LOG ("Polyfitter", "ZeroMQ publishing disabled");
	}
}

LaneInfo Polyfitter::processFrame (const cv::Mat &mask) {
	// Usar novo algoritmo de fitting
	std::vector<Lane> lanes = fitLanesInImage (mask);

	// Calcular centerline virtual com blending
	CenterlineResult centerline = computeVirtualCenterline (lanes, mask.cols, mask.rows);

	LaneInfo result;
	if (centerline.valid) {
		// Fix: center_line is a double, not a vector
		if (!centerline.blend.empty ()) {
			// Use the y-coordinate of the first blend point as center_line
			result.center_line = centerline.blend[0].y;
			// Calculate lateral offset from center
			result.lateral_offset = centerline.blend[0].x - (mask.cols / 2.0);
		} else {
			result.center_line = mask.cols / 2.0; // Default to image center
			result.lateral_offset = 0.0;
		}

		// Calculate boundaries if lanes exist
		if (lanes.size () >= 2) {
			result.left_boundary = lanes[0].centroids.empty () ? 0.0 : lanes[0].centroids[0].x;
			result.right_boundary =
			    lanes[1].centroids.empty () ? mask.cols : lanes[1].centroids[0].x;
		} else {
			result.left_boundary = 0.0;
			result.right_boundary = mask.cols;
		}

		result.yaw_error = 0.0; // Calculate based on trajectory if needed
		result.isValid = true;

		// === PUBLICAR DADOS PARA APLICAÇÕES EXTERNAS ===
		if (m_zeromq_enabled && m_publisherLaneData) {
			publishLaneData (result, mask);
		}
	} else {
		result.isValid = false;
		result.left_boundary = 0.0;
		result.right_boundary = mask.cols;
		result.center_line = mask.cols / 2.0;
		result.lateral_offset = 0.0;
		result.yaw_error = 0.0;
	}

	return result;
}

void Polyfitter::publishLaneData (const LaneInfo &laneInfo, const cv::Mat &binaryMask) {
	if (!m_publisherLaneData) return;

	try {
		// 1. Publicar informações da lane
		std::string laneData = serializeLaneInfo (laneInfo);
		m_publisherLaneData->publish ("lane_info", laneData);

		// 2. Publicar máscara binária processada
		std::vector<uchar> mask_buffer;
		cv::imencode (".png", binaryMask, mask_buffer);
		std::string mask_data (mask_buffer.begin (), mask_buffer.end ());
		m_publisherLaneData->publish ("processed_mask", mask_data);

	} catch (const std::exception &e) {
		ERROR_STREAM ("Polyfitter") << "Error publishing ZeroMQ data: " << e.what ();
	}
}

std::string Polyfitter::serializeLaneInfo (const LaneInfo &laneInfo) {
	std::ostringstream oss;
	oss << laneInfo.left_boundary << "," << laneInfo.right_boundary << "," << laneInfo.center_line
	    << "," << laneInfo.lateral_offset << "," << laneInfo.yaw_error << ","
	    << (laneInfo.isValid ? 1 : 0);
	return oss.str ();
}

std::vector<std::pair<std::string, cv::Mat>>
Polyfitter::loadImagesFromFolder (const std::string &folderPath) {
	std::vector<std::pair<std::string, cv::Mat>> images;
	std::vector<std::string> extensions = {".png", ".jpg", ".jpeg"};

	if (!fs::exists (folderPath)) {
		ERROR_STREAM ("Polyfitter") << "Folder does not exist: " << folderPath;
		return images;
	}

	std::vector<std::string> filenames;
	for (const auto &entry : fs::directory_iterator (folderPath)) {
		if (fs::is_regular_file (entry.path ())) {
			std::string filename = entry.path ().filename ().string ();
			std::string ext = entry.path ().extension ().string ();
			std::transform (ext.begin (), ext.end (), ext.begin (), ::tolower);

			if (std::find (extensions.begin (), extensions.end (), ext) != extensions.end ()) {
				filenames.push_back (filename);
			}
		}
	}

	std::sort (filenames.begin (), filenames.end ());

	for (const auto &filename : filenames) {
		std::string filepath = folderPath + "/" + filename;
		cv::Mat img = cv::imread (filepath, cv::IMREAD_GRAYSCALE);
		if (!img.empty ()) {
			images.push_back ({filename, img});
		}
	}

	return images;
}

std::vector<Point2D> Polyfitter::extractLanePoints (const cv::Mat &img) {
	std::vector<Point2D> points;
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			if (img.at<uchar> (y, x) > 0) {
				points.push_back (Point2D (x, y));
			}
		}
	}
	return points;
}

std::pair<std::vector<int>, std::vector<int>>
Polyfitter::clusterLanePoints (const std::vector<Point2D> &pts) {
	const size_t N = pts.size ();
	if (N == 0) return {{}, {}};

	// 2 x N matrix: each column is a point [x; y]
	arma::mat dataset (2, N);
	for (size_t i = 0; i < N; ++i) {
		dataset (0, i) = pts[i].x;
		dataset (1, i) = pts[i].y;
	}

	// Output labels and core point flags
	arma::Row<size_t> labels;
	mlpack::dbscan::DBSCAN<> db (EPS, MIN_SAMPLES);
	db.Cluster (dataset, labels);

	// Convert to int and gather unique cluster IDs != SIZE_MAX
	std::vector<int> intLabels (N);
	std::set<int> uniqueIds;
	for (size_t i = 0; i < N; ++i) {
		if (labels[i] == SIZE_MAX) {
			intLabels[i] = -1;
		} else {
			intLabels[i] = (int)labels[i];
			uniqueIds.insert (intLabels[i]);
		}
	}

	std::vector<int> uniques (uniqueIds.begin (), uniqueIds.end ());
	return {intLabels, uniques};
}

std::pair<std::vector<double>, std::vector<double>>
Polyfitter::slidingWindowCentroids (const std::vector<double> &x, const std::vector<double> &y,
                                    const cv::Size &imgShape, bool smooth) {

	int h = imgShape.height / NUM_WINDOWS;
	std::vector<double> cx, cy;

	for (int i = 0; i < NUM_WINDOWS; i++) {
		int yLow = imgShape.height - (i + 1) * h;
		int yHigh = imgShape.height - i * h;

		std::vector<double> windowX;
		std::vector<double> windowY;

		for (size_t j = 0; j < y.size (); j++) {
			if (y[j] >= yLow && y[j] < yHigh) {
				windowX.push_back (x[j]);
				windowY.push_back (y[j]);
			}
		}

		if (!windowX.empty ()) {
			double meanX =
			    std::accumulate (windowX.begin (), windowX.end (), 0.0) / windowX.size ();
			double meanY =
			    std::accumulate (windowY.begin (), windowY.end (), 0.0) / windowY.size ();
			cx.push_back (meanX);
			cy.push_back (meanY);
		}
	}

	if (smooth && cx.size () >= 3) {
		std::vector<double> smoothedCx = cx;
		for (size_t i = 1; i < cx.size () - 1; i++) {
			smoothedCx[i] = (cx[i - 1] + cx[i] + cx[i + 1]) / 3.0;
		}
		cx = smoothedCx;
	}

	return {cy, cx};
}

bool Polyfitter::hasSignFlip (const std::vector<double> &curve) {
	if (curve.size () < 3) return false;

	std::vector<double> firstDeriv (curve.size () - 1);
	for (size_t i = 0; i < firstDeriv.size (); i++) {
		firstDeriv[i] = curve[i + 1] - curve[i];
	}

	std::vector<double> secondDeriv (firstDeriv.size () - 1);
	for (size_t i = 0; i < secondDeriv.size (); i++) {
		secondDeriv[i] = firstDeriv[i + 1] - firstDeriv[i];
	}

	for (size_t i = 1; i < secondDeriv.size (); i++) {
		if ((secondDeriv[i] > 0) != (secondDeriv[i - 1] > 0)) {
			return true;
		}
	}
	return false;
}

bool Polyfitter::isStraightLine (const std::vector<double> &y, const std::vector<double> &x) const {
	if (x.size () < 4) return false;

	double meanX = std::accumulate (x.begin (), x.end (), 0.0) / x.size ();
	double meanY = std::accumulate (y.begin (), y.end (), 0.0) / y.size ();

	double num = 0, denX = 0, denY = 0;
	for (size_t i = 0; i < x.size (); i++) {
		double dx = x[i] - meanX;
		double dy = y[i] - meanY;
		num += dx * dy;
		denX += dx * dx;
		denY += dy * dy;
	}

	if (denX == 0 || denY == 0) return true;
	double corr = num / std::sqrt (denX * denY);
	return std::abs (corr) > STRAIGHT_LINE_THRESHOLD;
}

std::vector<double> Polyfitter::polyfit (const std::vector<double> &x, const std::vector<double> &y,
                                         int degree) const {
	int n = x.size ();
	int m = degree + 1;

	// Validation checks
	if (n < m) {
		// Not enough points for the degree - return empty vector
		ERROR_STREAM ("Polyfitter")
		    << "Not enough points (" << n << ") for degree " << degree << " polynomial";
		return std::vector<double> ();
	}

	if (n == 0 || x.size () != y.size ()) {
		ERROR_STREAM ("Polyfitter") << "Invalid input data for polyfit";
		return std::vector<double> ();
	}

	// Check for duplicate x values which can cause singular matrix
	std::vector<double> x_sorted = x;
	std::sort (x_sorted.begin (), x_sorted.end ());
	for (size_t i = 1; i < x_sorted.size (); i++) {
		if (std::abs (x_sorted[i] - x_sorted[i - 1]) < 1e-10) {
			// Duplicate x values detected - use simpler fitting
			WARNING_STREAM ("Polyfitter") << "Duplicate x values detected, using mean y value";
			double mean_y = std::accumulate (y.begin (), y.end (), 0.0) / y.size ();
			return std::vector<double>{mean_y}; // Return constant polynomial
		}
	}

	cv::Mat A (n, m, CV_64F);
	cv::Mat B (n, 1, CV_64F);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			A.at<double> (i, j) = std::pow (x[i], j);
		}
		B.at<double> (i, 0) = y[i];
	}

	cv::Mat coeffs;

	try {
		// Use SVD decomposition which is more robust for ill-conditioned matrices
		bool success = cv::solve (A, B, coeffs, cv::DECOMP_SVD);

		if (!success) {
			ERROR_STREAM ("Polyfitter") << "SVD solve failed, using QR decomposition";
			success = cv::solve (A, B, coeffs, cv::DECOMP_QR);
		}

		if (!success) {
			ERROR_STREAM ("Polyfitter") << "All solve methods failed, returning constant fit";
			double mean_y = std::accumulate (y.begin (), y.end (), 0.0) / y.size ();
			return std::vector<double>{mean_y};
		}
	} catch (const cv::Exception &e) {
		ERROR_STREAM ("Polyfitter") << "OpenCV exception in polyfit: " << e.what ();
		// Fallback to constant polynomial
		double mean_y = std::accumulate (y.begin (), y.end (), 0.0) / y.size ();
		return std::vector<double>{mean_y};
	}

	std::vector<double> result (m);
	for (int i = 0; i < m; i++) {
		result[m - 1 - i] = coeffs.at<double> (i, 0);
	}

	return result;
}

std::vector<double> Polyfitter::polyval (const std::vector<double> &coeffs,
                                         const std::vector<double> &x) {
	std::vector<double> result (x.size ());
	int degree = coeffs.size () - 1;

	for (size_t i = 0; i < x.size (); i++) {
		double val = 0;
		for (int j = 0; j <= degree; j++) {
			val += coeffs[j] * std::pow (x[i], degree - j);
		}
		result[i] = val;
	}

	return result;
}

std::vector<double> Polyfitter::fitLaneCurve (const std::vector<double> &y,
                                              const std::vector<double> &x, int imgWidth,
                                              const std::vector<double> &yPlot) {
	(void)imgWidth; // Suppress unused parameter warning

	// Input validation
	if (x.empty () || y.empty () || x.size () != y.size () || yPlot.empty ()) {
		ERROR_STREAM ("Polyfitter") << "Invalid input data for fitLaneCurve";
		return std::vector<double> (yPlot.size (), 0.0); // Return zeros
	}

	if (x.size () < 2) {
		WARNING_STREAM ("Polyfitter") << "Not enough points for lane curve fitting";
		// Return constant value based on first x point
		return std::vector<double> (yPlot.size (), x.empty () ? 0.0 : x[0]);
	}

	if (isStraightLine (y, x)) {
		auto coeffs = polyfit (y, x, 1);
		if (coeffs.empty ()) {
			// Fallback to constant
			double mean_x = std::accumulate (x.begin (), x.end (), 0.0) / x.size ();
			return std::vector<double> (yPlot.size (), mean_x);
		}
		return polyval (coeffs, yPlot);
	}

	auto coeffs = polyfit (y, x, 2);
	if (coeffs.empty ()) {
		// Fallback to linear fit
		coeffs = polyfit (y, x, 1);
		if (coeffs.empty ()) {
			// Final fallback to constant
			double mean_x = std::accumulate (x.begin (), x.end (), 0.0) / x.size ();
			return std::vector<double> (yPlot.size (), mean_x);
		}
		return polyval (coeffs, yPlot);
	}

	double a = coeffs.size () > 0 ? coeffs[0] : 0.0;

	if (std::abs (a) > CURVE_THRESHOLD && x.size () >= 4) {
		// Simple spline approximation using higher degree polynomial
		auto splineCoeffs = polyfit (y, x, std::min (3, (int)x.size () - 1));
		if (!splineCoeffs.empty ()) {
			return polyval (splineCoeffs, yPlot);
		}
	}

	return polyval (coeffs, yPlot);
}

std::vector<Lane> Polyfitter::fitLanesInImage (const cv::Mat &img) {
	// Implementação usando algoritmo melhorado do polyfit.cpp.txt
	std::vector<Lane> lanes;

	// Extrair pontos de lane
	auto lanePoints = extractLanePoints (img);

	// Clustering DBSCAN melhorado
	auto [labels, uniqueLabels] = clusterLanePoints (lanePoints);

	// Para cada cluster, criar uma lane
	for (int label : uniqueLabels) {
		if (label == -1) continue; // Ruído

		Lane lane;
		std::vector<double> clusterX, clusterY;

		for (size_t i = 0; i < lanePoints.size (); i++) {
			if (labels[i] == label) {
				clusterX.push_back (lanePoints[i].x);
				clusterY.push_back (lanePoints[i].y);
			}
		}

		if (clusterX.size () >= MIN_SAMPLES) {
			// Sliding window centroids
			auto [centroidX, centroidY] = slidingWindowCentroids (clusterX, clusterY, img.size ());

			// Converter para Point2D
			for (size_t i = 0; i < centroidX.size (); i++) {
				lane.centroids.push_back (Point2D (centroidX[i], centroidY[i]));
			}

			// Fit curve usando parâmetros otimizados
			auto yPlot = linspace (0, img.rows - 1, img.rows);
			auto xFitted = fitLaneCurve (centroidY, centroidX, img.cols, yPlot);

			for (size_t i = 0; i < yPlot.size (); i++) {
				lane.curve.push_back (Point2D (xFitted[i], yPlot[i]));
			}

			lanes.push_back (lane);
		}
	}

	return lanes;
}

std::pair<Lane *, Lane *> Polyfitter::selectRelevantLanes (std::vector<Lane> &lanes, int imgWidth,
                                                           int imgHeight) {
	double imgCenter = imgWidth / 2.0;
	Lane *leftLane = nullptr;
	Lane *rightLane = nullptr;

	std::vector<std::pair<double, Lane *>> laneInfos;

	for (auto &lane : lanes) {
		std::vector<double> bottomHalfX;
		for (const auto &point : lane.curve) {
			if (point.y >= imgHeight / 3.0) {
				bottomHalfX.push_back (point.x);
			}
		}

		if (!bottomHalfX.empty ()) {
			double avgX = std::accumulate (bottomHalfX.begin (), bottomHalfX.end (), 0.0) /
			              bottomHalfX.size ();
			laneInfos.push_back ({avgX, &lane});
		}
	}

	std::sort (laneInfos.begin (), laneInfos.end ());

	for (const auto &[avgX, lane] : laneInfos) {
		if (avgX < imgCenter) {
			leftLane = lane;
		} else if (avgX >= imgCenter && rightLane == nullptr) {
			rightLane = lane;
			break;
		}
	}

	return {leftLane, rightLane};
}

std::vector<double> Polyfitter::linspace (double start, double end, int num) {
	std::vector<double> result (num);
	double step = (end - start) / (num - 1);
	for (int i = 0; i < num; i++) {
		result[i] = start + i * step;
	}
	return result;
}

std::vector<double> Polyfitter::interp (const std::vector<double> &xNew,
                                        const std::vector<double> &x, const std::vector<double> &y,
                                        double leftVal, double rightVal) {
	std::vector<double> result (xNew.size ());

	for (size_t i = 0; i < xNew.size (); i++) {
		double xi = xNew[i];

		if (xi <= x[0]) {
			result[i] = leftVal;
		} else if (xi >= x.back ()) {
			result[i] = rightVal;
		} else {
			for (size_t j = 0; j < x.size () - 1; j++) {
				if (xi >= x[j] && xi <= x[j + 1]) {
					double t = (xi - x[j]) / (x[j + 1] - x[j]);
					result[i] = y[j] + t * (y[j + 1] - y[j]);
					break;
				}
			}
		}
	}

	return result;
}

CenterlineResult Polyfitter::computeVirtualCenterline (std::vector<Lane> &lanes, int imgWidth,
                                                       int imgHeight) {
	auto [leftLane, rightLane] = selectRelevantLanes (lanes, imgWidth, imgHeight);
	double carX = imgWidth / 2.0;

	CenterlineResult result;

	if (leftLane && rightLane) {
		// Método midpoint melhorado
		std::vector<double> xLeft, yLeft, xRight, yRight;
		for (const auto &point : leftLane->curve) {
			xLeft.push_back (point.x);
			yLeft.push_back (point.y);
		}
		for (const auto &point : rightLane->curve) {
			xRight.push_back (point.x);
			yRight.push_back (point.y);
		}

		double yMin = std::max (*std::min_element (yLeft.begin (), yLeft.end ()),
		                        *std::min_element (yRight.begin (), yRight.end ()));
		double yStart = imgHeight - 1;
		auto yCommon = linspace (yStart, yMin, 300);

		auto xLeftInterp = interp (yCommon, yLeft, xLeft, xLeft[0], xLeft.back ());
		auto xRightInterp = interp (yCommon, yRight, xRight, xRight[0], xRight.back ());

		std::vector<double> xC1 (yCommon.size ());
		std::vector<double> xC2 (yCommon.size (), carX);

		for (size_t i = 0; i < yCommon.size (); i++) {
			xC1[i] = (xLeftInterp[i] + xRightInterp[i]) / 2.0;
		}

		// Aplicar blending para suavizar transição
		for (size_t i = 0; i < yCommon.size (); i++) {
			double w = (yCommon[0] - yCommon[i]) / (yCommon[0] - yCommon.back ());
			double xBlend = w * xC1[i] + (1 - w) * xC2[i];

			result.blend.push_back (Point2D (xBlend, yCommon[i]));
			result.c1.push_back (Point2D (xC1[i], yCommon[i]));
			result.c2.push_back (Point2D (xC2[i], yCommon[i]));
		}

		result.valid = true;
	}

	return result;
}

void Polyfitter::displayImagesWithPolyfit (
    const std::vector<std::pair<std::string, cv::Mat>> &images, int cols) {
	if (images.empty ()) return;

	int numImages = images.size ();
	int rows = (numImages + cols - 1) / cols;

	// Calculate individual image display size
	int imgDisplayWidth = 300;
	int imgDisplayHeight = 200;

	// Create a large canvas to hold all images
	int canvasWidth = cols * imgDisplayWidth;
	int canvasHeight = rows * imgDisplayHeight;
	cv::Mat canvas = cv::Mat::zeros (canvasHeight, canvasWidth, CV_8UC3);

	std::vector<cv::Scalar> colors = {
	    cv::Scalar (0, 0, 255),   // Red
	    cv::Scalar (255, 0, 0),   // Blue
	    cv::Scalar (0, 255, 255), // Yellow
	    cv::Scalar (128, 0, 128), // Purple
	    cv::Scalar (0, 255, 0)    // Green
	};

	for (int idx = 0; idx < numImages; idx++) {
		int row = idx / cols;
		int col = idx % cols;

		const auto &[filename, img] = images[idx];

		// Resize image to fit in the grid
		cv::Mat resizedImg;
		cv::resize (img, resizedImg, cv::Size (imgDisplayWidth, imgDisplayHeight));

		// Convert grayscale to color for drawing
		cv::Mat colorImg;
		cv::cvtColor (resizedImg, colorImg, cv::COLOR_GRAY2BGR);

		// Calculate scale factors for drawing
		double scaleX = (double)imgDisplayWidth / img.cols;
		double scaleY = (double)imgDisplayHeight / img.rows;

		// Fit lanes
		auto lanes = fitLanesInImage (img);

		// Draw lane curves
		for (size_t i = 0; i < lanes.size (); i++) {
			const auto &lane = lanes[i];
			cv::Scalar color = colors[i % colors.size ()];

			// Draw centroids
			for (const auto &centroid : lane.centroids) {
				int x = (int)(centroid.x * scaleX);
				int y = (int)(centroid.y * scaleY);
				if (x >= 0 && x < imgDisplayWidth && y >= 0 && y < imgDisplayHeight) {
					cv::circle (colorImg, cv::Point (x, y), 2, cv::Scalar (0, 255, 0), -1);
					cv::circle (colorImg, cv::Point (x, y), 3, cv::Scalar (0, 0, 0), 1);
				}
			}

			// Draw curve
			std::vector<cv::Point> curvePoints;
			for (const auto &point : lane.curve) {
				int x = (int)(point.x * scaleX);
				int y = (int)(point.y * scaleY);
				if (x >= 0 && x < imgDisplayWidth && y >= 0 && y < imgDisplayHeight) {
					curvePoints.push_back (cv::Point (x, y));
				}
			}

			for (size_t j = 1; j < curvePoints.size (); j++) {
				cv::line (colorImg, curvePoints[j - 1], curvePoints[j], color, 2);
			}
		}

		// Compute and draw centerline
		auto centerlineResult = computeVirtualCenterline (lanes, img.cols, img.rows);
		if (centerlineResult.valid) {
			// Draw blended centerline
			std::vector<cv::Point> centerlinePoints;
			for (const auto &point : centerlineResult.blend) {
				int x = (int)(point.x * scaleX);
				int y = (int)(point.y * scaleY);
				if (x >= 0 && x < imgDisplayWidth && y >= 0 && y < imgDisplayHeight) {
					centerlinePoints.push_back (cv::Point (x, y));
				}
			}

			for (size_t j = 1; j < centerlinePoints.size (); j++) {
				cv::line (colorImg, centerlinePoints[j - 1], centerlinePoints[j],
				          cv::Scalar (0, 165, 255), 2); // Orange
			}
		}

		// Add title
		cv::putText (colorImg, filename, cv::Point (5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
		             cv::Scalar (255, 255, 255), 1);

		// Copy to canvas
		int startX = col * imgDisplayWidth;
		int startY = row * imgDisplayHeight;
		cv::Rect roi (startX, startY, imgDisplayWidth, imgDisplayHeight);
		colorImg.copyTo (canvas (roi));
	}

	cv::namedWindow ("Lane Detection Results", cv::WINDOW_AUTOSIZE);
	cv::imshow ("Lane Detection Results", canvas);
	cv::waitKey (0);
	cv::destroyAllWindows ();
}

double Polyfitter::calculateCTE (const std::vector<double> &polyCoeffs, double x, double y) const {
	if (polyCoeffs.empty ()) return 0.0;

	// Avaliar polinômio no ponto x para obter y_ref
	double y_ref = 0.0;
	int degree = polyCoeffs.size () - 1;

	for (int i = 0; i <= degree; i++) {
		y_ref += polyCoeffs[i] * std::pow (x, degree - i);
	}

	// CTE = y_atual - y_referencia
	return y - y_ref;
}

double Polyfitter::calculateEPSI (const std::vector<double> &polyCoeffs, double x,
                                  double psi) const {
	if (polyCoeffs.size () < 2) return 0.0;

	// Calcular derivada do polinômio para obter psi_des
	double psi_des = 0.0;
	int degree = polyCoeffs.size () - 1;

	// Derivada: d/dx[a*x^n + b*x^(n-1) + ... ] = n*a*x^(n-1) + (n-1)*b*x^(n-2) + ...
	for (int i = 0; i < degree; i++) {
		int power = degree - i - 1;
		if (power >= 0) {
			psi_des += (degree - i) * polyCoeffs[i] * std::pow (x, power);
		}
	}

	// psi_des = arctan(derivada)
	psi_des = std::atan (psi_des);

	// EPSI = psi_atual - psi_desejado
	return psi - psi_des;
}

std::vector<double> Polyfitter::getPolynomialCoeffs (const std::vector<Point2D> &trajectory) const {
	if (trajectory.size () < 2) return {};

	std::vector<double> x, y;
	for (const auto &point : trajectory) {
		x.push_back (point.x);
		y.push_back (point.y);
	}

	// Detectar variação lateral significativa
	if (y.size () >= 3) {
		double y_min = *std::min_element (y.begin (), y.end ());
		double y_max = *std::max_element (y.begin (), y.end ());
		double lateral_variation = y_max - y_min;

		// Se há variação lateral > 0.1m, usar curva quadrática
		if (lateral_variation > 0.1) {
			std::cout << "[Polyfitter] Lateral variation detected: " << lateral_variation
			          << "m -> using quadratic fit" << std::endl;
			return polyfit (x, y, 2); // Curva quadrática
		}
	}

	// Determinar se é linha reta ou curva baseado na correlação
	if (isStraightLine (y, x)) {
		return polyfit (x, y, 1); // Linha reta
	} else {
		return polyfit (x, y, 2); // Curva quadrática
	}
}

std::vector<Point2D> Polyfitter::convertImagePointsToWorld (
    const std::vector<int> &center_x, const std::vector<int> &center_y,
    const VehicleTransform &vehicle_transform, int img_width, int img_height) const {
	std::vector<Point2D> waypoints_world;
	if (center_y.empty ()) return waypoints_world;

	int center_x_img = img_width / 2;
	double real_height_m = 8.0; // Ajuste conforme sua câmera
	double escala_m_por_pixel = real_height_m / img_height;

	// Encontrar ponto de partida (mais próximo do veículo)
	int start_idx = 0;
	int max_y = center_y[0];
	for (size_t i = 1; i < center_y.size (); ++i) {
		if (center_y[i] > max_y) {
			max_y = center_y[i];
			start_idx = i;
		}
	}

	// Converter pontos de imagem para coordenadas mundo
	int N = std::min (10, (int)center_y.size ()); // Limitar a 10 pontos
	for (int i = 0; i < N; ++i) {
		int idx = start_idx - i;
		if (idx < 0) break;

		int x_img = center_x[idx];
		int y_img = center_y[idx];

		// Converter para coordenadas locais do veículo (sistema NED)
		double distance_ahead = (img_height - y_img) * escala_m_por_pixel;
		double lateral_offset = (x_img - center_x_img) * escala_m_por_pixel;

		// Transformar para coordenadas globais
		double cos_yaw = std::cos (vehicle_transform.yaw);
		double sin_yaw = std::sin (vehicle_transform.yaw);

		double world_x = vehicle_transform.x + distance_ahead * cos_yaw - lateral_offset * sin_yaw;
		double world_y = vehicle_transform.y + distance_ahead * sin_yaw + lateral_offset * cos_yaw;

		waypoints_world.emplace_back (world_x, world_y);
	}

	return waypoints_world;
}

LaneInfo Polyfitter::processMask (const cv::cuda::GpuMat &maskGpu) {
	cv::Mat binaryMask;
	maskGpu.download (binaryMask);

	if (binaryMask.type () == CV_32F) {
		cv::threshold (binaryMask, binaryMask, 0.5, 255, cv::THRESH_BINARY);
		binaryMask.convertTo (binaryMask, CV_8U);
	}
	return processFrame (binaryMask); // processFrame já retorna LaneInfo
}