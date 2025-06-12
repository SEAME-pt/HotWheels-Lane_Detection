#include "MPCPlanner.hpp"

MPCPlanner::MPCPlanner(void) {
	
}

MPCPlanner::MPCPlanner(const MPCPlanner &origin) {
	*this = origin;
}

MPCPlanner &MPCPlanner::operator=(const MPCPlanner &origin) {
	if (this != &origin)
		*this = origin;
	return *this;
}

MPCPlanner::~MPCPlanner(void) {
	
}

MPCPlanner::MPCPlanner(const MPCConfig& config, const MPCOptimizer& optimizer)
	: _config(config), _optimizer(optimizer) {}

ControlCommand MPCPlanner::plan(const VehicleState& current_state,
							  const std::vector<Eigen::Vector2d>& global_waypoints,
							  const LaneInfo* lane_info) {
	// Validação básica
	if (global_waypoints.empty()) {
		throw std::invalid_argument("Waypoints list cannot be empty");
	}

	// Prepara referência no frame local
	std::vector<Point2D> local_ref = _prepareReference(current_state, global_waypoints);

	// Chama o otimizador
	auto [throttle, steer] = _optimizer.solve(
		current_state.x,
		current_state.y,
		current_state.yaw,
		std::max(3.0, current_state.velocity), // Garante velocidade mínima
		local_ref,
		lane_info
	);

	return {throttle, steer};
}

std::vector<Point2D> MPCPlanner::_prepareReference(const VehicleState& state,
			const std::vector<Eigen::Vector2d>& global_waypoints) const {
	
	std::vector<Point2D> local_points;
	const double cos_yaw = cos(-state.yaw);
	const double sin_yaw = sin(-state.yaw);

	// Matriz de rotação
	Eigen::Matrix2d rotation_matrix;
	rotation_matrix << cos_yaw, -sin_yaw,
					  sin_yaw,  cos_yaw;

	for (const auto& wp : global_waypoints) {
		// Translação
		Eigen::Vector2d translated = wp - Eigen::Vector2d(state.x, state.y);
		
		// Rotação
		Eigen::Vector2d local = rotation_matrix * translated;
		
		local_points.emplace_back(local.x(), local.y());
		
		// Limita ao horizonte
		if (local_points.size() >= _config.horizon) break;
	}

	return local_points;
}

std::vector<Point2D> MPCPlanner::convertImagePointsToWorld(const std::vector<int>& center_x,
                                                  const std::vector<int>& center_y,
                                                  const VehicleTransform& vehicle_transform,
                                                  int img_width,
                                                  int img_height) const {
        std::vector<Point2D> waypoints_world;
        if (center_y.empty()) return waypoints_world;

        int center_x_img = img_width / 2;
        double real_height_m = 8.0;
        double escala_m_por_pixel = real_height_m / img_height;

        // Índice do ponto mais próximo do fundo da imagem (maior y)
        int start_idx = 0;
        int max_y = center_y[0];
        for (size_t i = 1; i < center_y.size(); ++i) {
            if (center_y[i] > max_y) {
                max_y = center_y[i];
                start_idx = i;
            }
        }

        int N = 10; // número de pontos amostrados
        for (int i = 0; i < N; ++i) {
            int idx = start_idx - i;
            if (idx < 0) break;

            int x_img = center_x[idx];
            int y_img = center_y[idx];

            double distance_ahead = (img_height - y_img) * escala_m_por_pixel;
            double lateral_offset = (center_x_img - x_img) * escala_m_por_pixel;

            double cos_yaw = std::cos(vehicle_transform.yaw);
            double sin_yaw = std::sin(vehicle_transform.yaw);

            double world_x = vehicle_transform.x + distance_ahead * cos_yaw - lateral_offset * sin_yaw;
            double world_y = vehicle_transform.y + distance_ahead * sin_yaw + lateral_offset * cos_yaw;

            waypoints_world.emplace_back(world_x, world_y);
        }

        return waypoints_world;
    }