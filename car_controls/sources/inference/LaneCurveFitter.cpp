#include "LaneCurveFitter.hpp"

std::optional<LaneCurveFitter::CenterlineResult> LaneCurveFitter::computeCenterline(const cv::Mat& binaryMask) {
    if (binaryMask.empty()) {
        return std::nullopt;
    }

    cv::Mat gray;
    if (binaryMask.channels() == 3) {
        cv::cvtColor(binaryMask, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = binaryMask;
    }

    
    // Fit lanes from the binary mask
    auto lanes = fitLanes(gray);
    
    if (lanes.empty()) {
        return std::nullopt;
    }
    
    // Compute virtual centerline
    auto result = computeVirtualCenterline(lanes, gray.cols, gray.rows);
    
    if (!result.valid) {
        return std::nullopt;
    }
    
    result.lanes = lanes;

    return result;
}

std::vector<LaneCurveFitter::Point2D> LaneCurveFitter::extractLanePoints(const cv::Mat& img) {
    std::vector<Point2D> points;

    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            if (img.at<uchar>(y, x) > 0) {
                points.emplace_back(x, y);
            }
        }
    }

    return points;
}

std::pair<std::vector<int>, std::vector<int>> LaneCurveFitter::clusterLanePoints(const std::vector<Point2D>& pts) {
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

std::pair<std::vector<double>, std::vector<double>> LaneCurveFitter::slidingWindowCentroids(
    const std::vector<double>& x, const std::vector<double>& y, 
    const cv::Size& imgShape, bool smooth) {
    
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

bool LaneCurveFitter::hasSignFlip(const std::vector<double>& curve) {
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

bool LaneCurveFitter::isStraightLine(const std::vector<double>& y, const std::vector<double>& x) {
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

std::vector<double> LaneCurveFitter::polyfit(const std::vector<double>& x, const std::vector<double>& y, int degree) {
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

std::vector<double> LaneCurveFitter::polyval(const std::vector<double>& coeffs, const std::vector<double>& x) {
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

std::vector<double> LaneCurveFitter::fitLaneCurve(const std::vector<double>& y, const std::vector<double>& x, 
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

void visualizeOverlay(const cv::Mat& maskGray, const cv::Mat& debugVis, const std::string& windowName) {
    // std::cout << "maskCPU (binary): " << maskGray.size() << std::endl;
    // std::cout << "debugVis:         " << debugVis.size() << std::endl;

    CV_Assert(maskGray.type() == CV_8UC1);                // Must be grayscale
    CV_Assert(debugVis.type() == CV_8UC3);                // Must be color
    CV_Assert(maskGray.size() == debugVis.size());        // Must match sizes

    cv::Mat maskColor;
    cv::cvtColor(maskGray, maskColor, cv::COLOR_GRAY2BGR);

    cv::Mat overlay;
    cv::addWeighted(maskColor, 0.5, debugVis, 0.5, 0.0, overlay);

    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::imshow(windowName, overlay);
}



std::vector<LaneCurveFitter::LaneCurve> LaneCurveFitter::fitLanes(const cv::Mat& img) {
    auto points = extractLanePoints(img);
    auto [labels, uniqueLabels] = clusterLanePoints(points);

    if (uniqueLabels.empty()) return {};

    std::vector<LaneCurve> lanes;

    // Optional visual debug
    bool enableDebugVis = false;
    cv::Mat debugVis(img.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
        cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255)
    };

    for (int label : uniqueLabels) {
        std::vector<double> x, y;

        for (size_t i = 0; i < points.size(); i++) {
            cv::Point pt(points[i].x, points[i].y);
            if (enableDebugVis)
                cv::circle(debugVis, pt, 1, colors[label % colors.size()], -1);
            if (labels[i] == label) {
                // x.push_back(points[i].x);
                x.push_back(pt.x);
                y.push_back(pt.y);
                // y.push_back(points[i].y);
            }
        }

        if (x.empty()) continue;

        // Use cluster-based Y range for windowing
        double yMin = *std::min_element(y.begin(), y.end());
        double yMax = *std::max_element(y.begin(), y.end());
        double span = yMax - yMin;

        if (span < 10) {
            // std::cout << " -> Skipping: lane span too small (" << span << ").\n";
            continue;
        }

        double h = span / NUM_WINDOWS;
        std::vector<double> centX, centY;

        for (int i = 0; i < NUM_WINDOWS; i++) {
            double yLow = yMax - (i + 1) * h;
            double yHigh = yMax - i * h;
            std::vector<double> winX, winY;

            for (size_t j = 0; j < y.size(); j++) {
                if (y[j] >= yLow && y[j] < yHigh) {
                    winX.push_back(x[j]);
                    winY.push_back(y[j]);
                }
            }

            if (!winX.empty()) {
                double meanX = std::accumulate(winX.begin(), winX.end(), 0.0) / winX.size();
                double meanY = std::accumulate(winY.begin(), winY.end(), 0.0) / winY.size();
                centX.push_back(meanX);
                centY.push_back(meanY);
            }
        }

        if (centY.size() < 2) {
            // std::cout << " -> Skipping: not enough centroids.\n";
            continue;
        }

        // Sort centroids by increasing Y
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

        try {
            auto testCoeffs = polyfit(sortedCentY, sortedCentX, 2);
            auto testCurve = polyval(testCoeffs, sortedCentY);
            if (hasSignFlip(testCurve)) {
                std::tie(centY, centX) = slidingWindowCentroids(x, y, img.size(), true);
                sortedCentY.clear(); sortedCentX.clear();
                std::vector<size_t> newIndices(centY.size());
                std::iota(newIndices.begin(), newIndices.end(), 0);
                std::sort(newIndices.begin(), newIndices.end(), [&](size_t a, size_t b) {
                    return centY[a] < centY[b];
                });
                for (size_t idx : newIndices) {
                    sortedCentY.push_back(centY[idx]);
                    sortedCentX.push_back(centX[idx]);
                }
            }
        } catch (...) {
            // std::cout << " -> Skipping: curve fitting exception.\n";
            continue;
        }

        double yStart = std::max(0.0, *std::min_element(sortedCentY.begin(), sortedCentY.end()) - 30);
        double yEnd = std::min((double)img.rows, *std::max_element(sortedCentY.begin(), sortedCentY.end()) + 10);
        auto yPlot = linspace(yStart, yEnd, 300);
        auto xPlot = fitLaneCurve(sortedCentY, sortedCentX, img.cols, yPlot);

        LaneCurve lane;
        for (size_t i = 0; i < sortedCentX.size(); i++)
            lane.centroids.emplace_back(sortedCentX[i], sortedCentY[i]);
        for (size_t i = 0; i < xPlot.size(); i++)
            lane.curve.emplace_back(xPlot[i], yPlot[i]);

        lanes.push_back(lane);
    }

    if (enableDebugVis) {
        cv::Mat maskGray;
        if (img.channels() == 3) {
            cv::cvtColor(img, maskGray, cv::COLOR_BGR2GRAY);
        } else {
            maskGray = img;
        }

        visualizeOverlay(maskGray, debugVis, "Cluster Debug Overlay");
        cv::waitKey(0);
    }



    return lanes;
}



std::pair<LaneCurveFitter::LaneCurve*, LaneCurveFitter::LaneCurve*> LaneCurveFitter::selectRelevantLanes(
    std::vector<LaneCurve>& lanes, int imgWidth, int imgHeight) {
    double imgCenter = imgWidth / 2.0;
    LaneCurve* leftLane = nullptr;
    LaneCurve* rightLane = nullptr;
    
    std::vector<std::pair<double, LaneCurve*>> laneInfos;
    
    for (auto& lane : lanes) {
        std::vector<double> bottomHalfX;
        for (const auto& point : lane.curve) {
            if (point.y >= imgHeight / 6.0) {
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

std::vector<double> LaneCurveFitter::linspace(double start, double end, int num) {
    std::vector<double> result(num);
    double step = (end - start) / (num - 1);
    for (int i = 0; i < num; i++) {
        result[i] = start + i * step;
    }
    return result;
}

std::vector<double> LaneCurveFitter::interp(const std::vector<double>& xNew, const std::vector<double>& x, 
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

LaneCurveFitter::CenterlineResult LaneCurveFitter::computeVirtualCenterline(
    std::vector<LaneCurve>& lanes, int imgWidth, int imgHeight) {
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
                result.blended.push_back(Point2D(xC1[i], yCommon[i]));
                result.midpoint.push_back(Point2D(xC1[i], yCommon[i]));
                result.straight.push_back(Point2D(xC2[i], yCommon[i]));
            }
        } else {
            for (size_t i = 0; i < yCommon.size(); i++) {
                double w = (yCommon[0] - yCommon[i]) / (yCommon[0] - yCommon.back());
                double xBlend = w * xC1[i] + (1 - w) * xC2[i];
                
                result.blended.push_back(Point2D(xBlend, yCommon[i]));
                result.midpoint.push_back(Point2D(xC1[i], yCommon[i]));
                result.straight.push_back(Point2D(xC2[i], yCommon[i]));
            }
        }
        
        result.valid = true;
        
    } else if (leftLane || rightLane) {
        // Offset method
        LaneCurve* lane = leftLane ? leftLane : rightLane;
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
                result.blended.push_back(Point2D(xC1Interp[i], yCommon[i]));
                result.midpoint.push_back(Point2D(xC1Interp[i], yCommon[i]));
                result.straight.push_back(Point2D(xC2[i], yCommon[i]));
            }
        } else {
            for (size_t i = 0; i < yCommon.size(); i++) {
                double w = (yCommon[0] - yCommon[i]) / (yCommon[0] - yCommon.back());
                double xBlend = w * xC1Interp[i] + (1 - w) * xC2[i];
                
                result.blended.push_back(Point2D(xBlend, yCommon[i]));
                result.midpoint.push_back(Point2D(xC1Interp[i], yCommon[i]));
                result.straight.push_back(Point2D(xC2[i], yCommon[i]));
            }
        }
        
        result.valid = true;
    }
    
    return result;
}