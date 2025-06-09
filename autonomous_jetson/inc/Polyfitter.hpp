#ifndef POLYFITTER_HPP
# define POLYFITTER_HPP

// #include <opencv2/opencv.hpp>
# include <vector>
# include <string>
# include <utility>
# include "CommonTypes.hpp"

struct Lane {
    std::vector<Point2D> centroids;
    std::vector<Point2D> curve;
};

struct CenterlineResult {
    std::vector<Point2D> blend;
    std::vector<Point2D> c1;
    std::vector<Point2D> c2;
    bool valid;
    CenterlineResult();
};

class Polyfitter {
private:
    static constexpr double EPS = 5.0;
    static constexpr int MIN_SAMPLES = 5;
    static constexpr int NUM_WINDOWS = 40;
    static constexpr double STRAIGHT_LINE_THRESHOLD = 0.98;
    static constexpr double CURVE_THRESHOLD = 0.0012;
    static constexpr int LANE_WIDTH_PX = 300;

public:
    Polyfitter();
    ~Polyfitter();

    std::vector<std::pair<std::string, cv::Mat>> loadImagesFromFolder(const std::string& folderPath);
    std::vector<Point2D> extractLanePoints(const cv::Mat& img);
    std::pair<std::vector<int>, std::vector<int>> clusterLanePoints(const std::vector<Point2D>& pts);
    std::pair<std::vector<double>, std::vector<double>> slidingWindowCentroids(const std::vector<double>& x, const std::vector<double>& y, const cv::Size& imgShape, bool smooth = false);
    bool hasSignFlip(const std::vector<double>& curve);
    bool isStraightLine(const std::vector<double>& y, const std::vector<double>& x);
    std::vector<double> polyfit(const std::vector<double>& x, const std::vector<double>& y, int degree);
    std::vector<double> polyval(const std::vector<double>& coeffs, const std::vector<double>& x);
    std::vector<double> fitLaneCurve(const std::vector<double>& y, const std::vector<double>& x, int imgWidth, const std::vector<double>& yPlot);
    std::vector<Lane> fitLanesInImage(const cv::Mat& img);
    std::pair<Lane*, Lane*> selectRelevantLanes(std::vector<Lane>& lanes, int imgWidth, int imgHeight);
    std::vector<double> linspace(double start, double end, int num);
    std::vector<double> interp(const std::vector<double>& xNew, const std::vector<double>& x, const std::vector<double>& y, double leftVal, double rightVal);
    CenterlineResult computeVirtualCenterline(std::vector<Lane>& lanes, int imgWidth, int imgHeight);
};

#endif // POLYFITTER_HPP
