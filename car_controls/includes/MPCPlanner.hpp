#ifndef MPCPLANNER_HPP
#define MPCPLANNER_HPP

#include "MPCOptimizer.hpp"
#include <Eigen/Dense>
#include <cstddef>
#include <vector>

class MPCPlanner {

	private:
		std::vector<Eigen::Vector2d>
		_prepareReference(const VehicleState &state,
		                  const std::vector<Eigen::Vector2d> &global_waypoints) const;

		MPCOptimizer _optimizer;

		// Adicionar método para mapear comandos para hardware
		ControlCommand _mapCommandsToHardware(double throttle, double steer) const;

	public:
		MPCPlanner(void);
		MPCPlanner(const MPCPlanner &orign);
		MPCPlanner &operator=(const MPCPlanner &orign);
		~MPCPlanner(void);

		MPCPlanner(const MPCOptimizer &optimizer);

		ControlCommand plan(const VehicleState &current_state,
		                    const std::vector<Point2D> &global_waypoints,
		                    const LaneInfo *lane_info = NULL);

		std::vector<Point2D> convertImagePointsToWorld(const std::vector<int> &center_x,
		                                               const std::vector<int> &center_y,
		                                               const VehicleTransform &vehicle_transform,
		                                               int img_width, int img_height) const;

		// Novo método para acessar a trajetória prevista do MPC
		const std::vector<Point2D>& getPredictedTrajectory() const { return _optimizer.getPredictedTrajectory(); }
};

#endif /* !MPCPlanner */