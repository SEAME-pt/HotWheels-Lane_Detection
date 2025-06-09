#ifndef MPCPLANNER_HPP
# define MPCPLANNER_HPP

#include <vector>
#include <Eigen/Dense>
#include "MPCOptimizer.hpp"
#include "MPCConfig.hpp"

struct VehicleState {
    double x;
    double y;
    double yaw;
    double velocity;
};

struct VehicleTransform {
    double x, y;      // Posição
    double yaw;       // Orientação em radianos
};

struct ControlCommand {
    double throttle;
    double steer;
};

class MPCPlanner{

	private:
		std::vector<Point2D> _prepareReference(const VehicleState& state, const std::vector<Eigen::Vector2d>& global_waypoints) const;
    
		MPCConfig _config;
		MPCOptimizer _optimizer;

	public:
		MPCPlanner(void);
		MPCPlanner(const MPCPlanner &orign);
		MPCPlanner &operator=(const MPCPlanner &orign);
		~MPCPlanner(void);

		MPCPlanner(const MPCConfig& config, const MPCOptimizer& optimizer);
    
		ControlCommand plan(const VehicleState& current_state, const std::vector<Eigen::Vector2d>& global_waypoints, const LaneInfo* lane_info = nullptr);
		
		std::vector<Point2D> convertImagePointsToWorld(const std::vector<int>& center_x, const std::vector<int>& center_y, const VehicleTransform& vehicle_transform, int img_width, int img_height) const;
};

#endif /* !MPCPlanner */