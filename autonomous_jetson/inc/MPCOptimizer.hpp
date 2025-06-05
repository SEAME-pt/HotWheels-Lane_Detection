#ifndef MPCOPTIMIZER_HPP
# define MPCOPTIMIZER_HPP

# include "MPCConfig.hpp"
# include <nlopt.h>
# include <string>
# include <filesystem>
# include <vector>

struct LaneInfo {
    double left_boundary;
    double right_boundary;
    double center_line;
};


struct Point2D {
    double x, y;
    Point2D(double x = 0.0, double y = 0.0) : x(x), y(y) {}
};

class MPCOptimizer{

	private:
		MPCConfig _mpc;

	public:
		MPCOptimizer(void);
		MPCOptimizer(const MPCOptimizer &orign);
		MPCOptimizer &operator=(const MPCOptimizer &orign);
		~MPCOptimizer(void);

		std::pair<double, double> solve(double x0, double y0, double yaw0, double v0,
                                             const std::vector<Point2D>& reference,
                                             const LaneInfo* lane_info = nullptr);
		double MPCOptimizer::costFunction(const std::vector<double>& u, const std::vector<double>& state,
										const std::vector<Point2D>& reference, const LaneInfo* lane_info);
};

#endif /* !MPCOPTIMIZER */