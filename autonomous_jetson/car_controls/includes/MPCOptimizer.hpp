#ifndef MPCOPTIMIZER_HPP
#define MPCOPTIMIZER_HPP

#include "CommonTypes.hpp"
#include "MPCConfig.hpp"
#include <cmath>
#include <ctime>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <nlopt.hpp>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::experimental::filesystem;

class MPCOptimizer {

	private:
		MPCConfig _mpc;
		std::vector<double> _current_state;
		std::vector<Point2D> _current_reference;
		const LaneInfo *_current_lane_info;
		std::vector<double> _current_poly_coeffs; // Add this line
		std::vector<Point2D> _predicted_trajectory;

		// Debug system (simplified - uses centralized Debugger)
		bool debug_enabled;
		bool output_debug_to_file;
		int debug_counter;

	public:
		MPCOptimizer(void);
		MPCOptimizer(const MPCConfig &config);
		MPCOptimizer(const MPCOptimizer &orign);
		MPCOptimizer &operator=(const MPCOptimizer &orign);
		~MPCOptimizer(void);

		// Getters
		const std::vector<double> &getCurrentState(void) const {
			return _current_state;
		}
		const std::vector<Point2D> &getCurrentReference(void) const {
			return _current_reference;
		}
		const LaneInfo *getCurrentLaneInfo(void) const {
			return _current_lane_info;
		}
		const std::vector<Point2D> &getPredictedTrajectory() const {
			return _predicted_trajectory;
		}

		// Debug methods (simplified - uses centralized Debugger)
		void enableDebug(bool enable = true) {
			debug_enabled = enable;
		}
		void setDebugOutputToFile(bool enable = true) {
			output_debug_to_file = enable;
		}

		// Declaração da função solve
		std::pair<double, double> solve(double x0, double y0, double yaw0, double v0,
		                                const std::vector<Point2D> &reference,
		                                const LaneInfo *lane_info = nullptr);

		// Função de custo para o otimizador
		double _costFunction(const std::vector<double> &u, const std::vector<double> &state,
		                     const std::vector<Point2D> &reference,
		                     const LaneInfo *lane_info) const;

		void _kinematicModel(double &x, double &y, double &yaw, double &v, double &cte,
		                     double &epsi, double throttle, double steer,
		                     const std::vector<double> &poly_coeffs) const;

		double _normalizeAngle(double angle) const;
		double _calculatePathCurvature(const std::vector<Point2D> &reference) const;
		double _calculateCurveCurvature(const std::vector<double> &x_coords,
		                                const std::vector<double> &y_coords) const;

		// Adicionar método para tratar latência
		std::vector<double> _predictStateWithLatency(double x0, double y0, double yaw0, double v0,
		                                             double throttle, double steer,
		                                             double latency) const;
};

#endif /* !MPCOPTIMIZER */