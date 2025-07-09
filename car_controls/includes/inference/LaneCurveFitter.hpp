#ifndef LANECURVEFITTER_HPP
#define LANECURVEFITTER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <optional>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <mlpack/core.hpp>

class LaneCurveFitter {
public:
    struct Point2D {
        double x, y;
        Point2D(double x = 0, double y = 0) : x(x), y(y) {}
        
        // Conversion to cv::Point for drawing
        operator cv::Point() const {
            return cv::Point(static_cast<int>(x), static_cast<int>(y));
        }
    };
    
    struct LaneCurve {
        std::vector<Point2D> centroids;
        std::vector<Point2D> curve;
    };
    
    struct CenterlineResult {
        std::vector<Point2D> blended;
        std::vector<Point2D> midpoint;
        std::vector<Point2D> straight;
        std::vector<LaneCurve> lanes;
        bool valid;
        
        CenterlineResult() : valid(false) {}
    };

    // Public interface - only method that should be called externally
    std::optional<CenterlineResult> computeCenterline(const cv::Mat& binaryMask);

private:
    // Constants
    static constexpr double EPS = 5.0;
    static constexpr int MIN_SAMPLES = 1;
    static constexpr int NUM_WINDOWS = 20;
    static constexpr double STRAIGHT_LINE_THRESHOLD = 0.98;
    static constexpr double CURVE_THRESHOLD = 0.0012;
    static constexpr int LANE_WIDTH_PX = 80;

    // Private helper methods
    std::vector<Point2D> extractLanePoints(const cv::Mat& img);
    std::pair<std::vector<int>, std::vector<int>> clusterLanePoints(const std::vector<Point2D>& pts);
    std::pair<std::vector<double>, std::vector<double>> slidingWindowCentroids(
        const std::vector<double>& x, const std::vector<double>& y, 
        const cv::Size& imgShape, bool smooth = false);
    
    bool hasSignFlip(const std::vector<double>& curve);
    bool isStraightLine(const std::vector<double>& y, const std::vector<double>& x);
    
    std::vector<double> polyfit(const std::vector<double>& x, const std::vector<double>& y, int degree);
    std::vector<double> polyval(const std::vector<double>& coeffs, const std::vector<double>& x);
    
    std::vector<double> fitLaneCurve(const std::vector<double>& y, const std::vector<double>& x, 
                                   int imgWidth, const std::vector<double>& yPlot);
    
    std::vector<LaneCurve> fitLanes(const cv::Mat& img);
    std::pair<LaneCurve*, LaneCurve*> selectRelevantLanes(std::vector<LaneCurve>& lanes, int imgWidth, int imgHeight);
    
    std::vector<double> linspace(double start, double end, int num);
    std::vector<double> interp(const std::vector<double>& xNew, const std::vector<double>& x, 
                             const std::vector<double>& y, double leftVal, double rightVal);
    
    CenterlineResult computeVirtualCenterline(std::vector<LaneCurve>& lanes, int imgWidth, int imgHeight);
};

#endif