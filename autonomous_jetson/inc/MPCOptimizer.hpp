#ifndef MPCOPTIMIZER_HPP
# define MPCOPTIMIZER_HPP

# include "MPCConfig.hpp"
# include "CommonTypes.hpp"
# include <cmath>
# include <filesystem>
# include <nlopt.hpp>
# include <numeric>
# include <stdexcept>
# include <string>
# include <vector>

class MPCOptimizer{

	private:
		MPCConfig _mpc;
		std::vector<double> _current_state;
		std::vector<Point2D> _current_reference;
		const LaneInfo* _current_lane_info;

	public:
		MPCOptimizer(void);
		MPCOptimizer(const MPCOptimizer &orign);
		MPCOptimizer &operator=(const MPCOptimizer &orign);
		~MPCOptimizer(void);

		// Getters
		const std::vector<double>& getCurrentState(void) const { return _current_state; }
		const std::vector<Point2D>& getCurrentReference(void) const { return _current_reference; }
		const LaneInfo* getCurrentLaneInfo(void) const { return _current_lane_info; }

	 // Declaração da função solve
		std::pair<double, double> solve(double x0, double y0, double yaw0, double v0, const std::vector<Point2D>& reference, const LaneInfo* lane_info = nullptr);

	// Função de custo para o otimizador
		double _costFunction(const std::vector<double>& u, const std::vector<double>& state, const std::vector<Point2D>& reference, const LaneInfo* lane_info) const;
	
		void _kinematicModel(double& x, double& y, double& yaw, double& v, double throttle, double steer) const;

		double _normalizeAngle(double angle) const;
		double _calculatePathCurvature(const std::vector<Point2D>& reference) const;
		double _calculateCurveCurvature(const std::vector<double>& x_coords, const std::vector<double>& y_coords) const;

};

#endif /* !MPCOPTIMIZER */