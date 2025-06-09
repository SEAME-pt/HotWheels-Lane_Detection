#include "MPCOptimizer.hpp"

MPCOptimizer::MPCOptimizer(void) {
	
}

MPCOptimizer::MPCOptimizer(const MPCOptimizer &origin) {
	*this = origin;
}

MPCOptimizer &MPCOptimizer::operator=(const MPCOptimizer &origin) {
	if (this != &origin)
		*this = origin;
	return *this;
}

MPCOptimizer::~MPCOptimizer(void) {
	
}

static double costWrapper(unsigned n, const double* x, double* grad, void* data) {
    MPCOptimizer* self = static_cast<MPCOptimizer*>(data);
    // Convertendo x para std::vector<double>
    std::vector<double> u(x, x + n);
    // Chamando a função de custo da classe
    return self->_costFunction(u, self->getCurrentState(), self->getCurrentReference(), self->getCurrentLaneInfo());
}

std::pair<double, double> MPCOptimizer::solve(double x0, double y0, double yaw0, double v0,
                                             const std::vector<Point2D>& reference,
                                             const LaneInfo* lane_info) {
	// This function should implement the MPC algorithm to compute the optimal control inputs
	// based on the current state and reference trajectory.
	std::vector<double> state = {0.0, 0.0, 0.0, v0}; // x, y, yaw, velocity
	// Configuração do otimizador
    nlopt::opt optimizer(nlopt::LD_MMA, 2 * MPCConfig::horizon);
	
	// Limites das variáveis de controle
	std::vector<double> lb(2 * MPCConfig::horizon);
	std::vector<double> ub(2 * MPCConfig::horizon);

	for (int i = 0; i < MPCConfig::horizon; ++i) {
		lb[2*i] = MPCConfig::throttle_limits[0]; // 0.0
		ub[2*i] = MPCConfig::throttle_limits[1]; // max_throttle
		lb[2*i + 1] = MPCConfig::steering_limits[0]; // -max_steer
		ub[2*i + 1] = MPCConfig::steering_limits[1]; // max_steer
	}
	optimizer.set_lower_bounds(lb);
	optimizer.set_upper_bounds(ub);

	// Movimento inicial
	std::vector<double> u0(2 * MPCConfig::horizon, 0.0);
	for (size_t i = 0; i < u0.size(); i += 2) {
		u0[i] = 0.5; // throttle initial guess
	}

	// Configuração do otimizador usando lambda
	optimizer.set_min_objective(costWrapper, this);

	optimizer.set_maxeval(MPCConfig::max_iter);
    optimizer.set_xtol_rel(1e-6);
    optimizer.set_ftol_rel(1e-6);

	double min_cost;
	try {
        optimizer.optimize(u0, min_cost);
    } catch(const std::exception& e) {
        throw std::runtime_error("Falha na otimização: " + std::string(e.what()));
    }

    return {u0[0], u0[1]}; // Retorna primeiro par de controles
}

double MPCOptimizer::_normalizeAngle(double angle) const {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

double MPCOptimizer::_calculatePathCurvature(const std::vector<Point2D>& reference) const {
    if (reference.size() < 3) {
        return 0.0;
    }
    
    // Usar os primeiros 3 pontos
    const Point2D& p1 = reference[0];
    const Point2D& p2 = reference[1];
    const Point2D& p3 = reference[2];
    
    // Calcular distâncias entre pontos
    double a = std::sqrt(std::pow(p2.x - p1.x, 2) + std::pow(p2.y - p1.y, 2));
    double b = std::sqrt(std::pow(p3.x - p2.x, 2) + std::pow(p3.y - p2.y, 2));
    double c = std::sqrt(std::pow(p3.x - p1.x, 2) + std::pow(p3.y - p1.y, 2));
    
    if (a * b * c == 0.0) {
        return 0.0;
    }
    
    // Semi-perímetro
    double s = (a + b + c) / 2.0;
    // Área do triângulo (fórmula de Heron)
    double area = std::sqrt(std::max(0.0, s * (s - a) * (s - b) * (s - c)));
    
    // Curvatura = 4 * área / (a * b * c)
    double curvature = (a * b * c > 0.0) ? 4.0 * area / (a * b * c) : 0.0;
    
    return curvature;
}

double MPCOptimizer::_costFunction(const std::vector<double>& u, 
                                 const std::vector<double>& state,
                                 const std::vector<Point2D>& reference,
                                 const LaneInfo* lane_info) const {
    double cost = 0.0;
    double x = state[0], y = state[1], yaw = state[2], v = state[3];

    double curvature = _calculatePathCurvature(reference);
    bool is_curve = std::abs(curvature) > 0.05;

	double w_cte, w_etheta, w_velocity, w_throttle, w_steer, target_speed;
    if (is_curve) {
        w_cte = _mpc.w_cte_curve;
        w_etheta = _mpc.w_etheta_curve;
        w_velocity = _mpc.w_vel_curve;
        w_throttle = _mpc.w_throttle_curve;
        w_steer = _mpc.w_steer_curve;
        target_speed = std::max(_mpc.target_speed_curve_base, 
                               _mpc.target_speed_straight - std::abs(curvature) * _mpc.target_speed_curve_factor);
    } else {
        w_cte = _mpc.w_cte_straight;
        w_etheta = _mpc.w_etheta_straight;
        w_velocity = _mpc.w_vel_straight;
        w_throttle = _mpc.w_throttle_straight;
        w_steer = _mpc.w_steer_straight;
        target_speed = _mpc.target_speed_straight;
    }

    for (int t = 0; t < MPCConfig::horizon; ++t) {
        double throttle = u[2*t];
        double steer = u[2*t+1];
        
        // Atualiza estado
        v += throttle * MPCConfig::dt;
        v = std::max(0.5, std::min(v, 10.0)); // Limites de velocidade
        
        x += v * std::cos(yaw) * MPCConfig::dt;
        y += v * std::sin(yaw) * MPCConfig::dt;
        yaw += (v / MPCConfig::wheelbase) * std::tan(steer) * MPCConfig::dt;
        yaw = _normalizeAngle(yaw);
        
        // CTE (soma dos quadrados das diferenças, igual ao Python)
        if (t < (int)reference.size()) {
            double dx = x - reference[t].x;
            double dy = y - reference[t].y;
            double cte = dx*dx + dy*dy;
            cost += w_cte * cte;
            
            // Erro de direção
            double desired_heading = std::atan2(reference[t].y - y, reference[t].x - x);
            double heading_error = _normalizeAngle(yaw - desired_heading);
            cost += w_etheta * heading_error * heading_error;
        }
        
        // Erro de velocidade
        double speed_error = v - target_speed;
        cost += w_velocity * speed_error * speed_error;
        
        // Esforço de controle
        cost += w_throttle * throttle * throttle;
        cost += w_steer * steer * steer;
        
        // Taxa de mudança do steer
        if (t > 0) {
            double prev_steer = u[2*(t-1)+1];
            double steer_rate = (steer - prev_steer) / MPCConfig::dt;
            cost += 0.5 * steer_rate * steer_rate;
        }
    }
    return cost;
}


void MPCOptimizer::_kinematicModel(double& x, double& y, double& yaw, double& v,
                                 double throttle, double steer) const {
    x += v * std::cos(yaw) * MPCConfig::dt;
    y += v * std::sin(yaw) * MPCConfig::dt;
    yaw += v * std::tan(steer) / MPCConfig::wheelbase * MPCConfig::dt;
    v += throttle * MPCConfig::dt;
}

double MPCOptimizer::_calculateCurveCurvature(const std::vector<double>& x_coords,
                                   const std::vector<double>& y_coords) const {
        if (x_coords.size() < 3 || y_coords.size() < 3) return 0.0;

        std::vector<double> dx(x_coords.size()), dy(y_coords.size());
        std::vector<double> ddx(x_coords.size()), ddy(y_coords.size());

        // Cálculo das derivadas (gradiente simples)
        for (size_t i = 1; i < x_coords.size() - 1; ++i) {
            dx[i] = (x_coords[i+1] - x_coords[i-1]) / 2.0;
            dy[i] = (y_coords[i+1] - y_coords[i-1]) / 2.0;
        }
        dx[0] = dx[1]; dx.back() = dx[dx.size()-2];
        dy[0] = dy[1]; dy.back() = dy[dy.size()-2];

        for (size_t i = 1; i < dx.size() - 1; ++i) {
            ddx[i] = (dx[i+1] - dx[i-1]) / 2.0;
            ddy[i] = (dy[i+1] - dy[i-1]) / 2.0;
        }
        ddx[0] = ddx[1]; ddx.back() = ddx[ddx.size()-2];
        ddy[0] = ddy[1]; ddy.back() = ddy[ddy.size()-2];

        std::vector<double> curvature(x_coords.size());
        for (size_t i = 0; i < x_coords.size(); ++i) {
            double numerator = std::abs(dx[i] * ddy[i] - dy[i] * ddx[i]);
            double denom = std::pow(dx[i]*dx[i] + dy[i]*dy[i], 1.5);
            if (denom < 1e-6) denom = 1e-6;  // evitar divisão por zero
            curvature[i] = numerator / denom;
        }

        // Retorna a média da curvatura
        double sum = std::accumulate(curvature.begin(), curvature.end(), 0.0);
        return sum / curvature.size();
    }