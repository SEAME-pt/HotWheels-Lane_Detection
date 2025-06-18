#include "MPCPlanner.hpp"

MPCPlanner::MPCPlanner(void) {}

MPCPlanner::MPCPlanner(const MPCPlanner &origin) { *this = origin; }

MPCPlanner &MPCPlanner::operator=(const MPCPlanner &origin) {
  if (this != &origin)
    *this = origin;
  return *this;
}

MPCPlanner::~MPCPlanner(void) {}

MPCPlanner::MPCPlanner(const MPCConfig &config, const MPCOptimizer &optimizer)
    : _config(config), _optimizer(optimizer) {}

ControlCommand MPCPlanner::plan(const VehicleState &current_state,
                                const std::vector<Point2D> &global_waypoints,
                                const LaneInfo *lane_info) {
  if (global_waypoints.empty()) {
    throw std::invalid_argument("Waypoints list cannot be empty");
  }
  
  // Convert Point2D to Eigen::Vector2d
  std::vector<Eigen::Vector2d> global_waypoints_eigen;
  global_waypoints_eigen.reserve(global_waypoints.size());
  for (const auto &pt : global_waypoints) {
    global_waypoints_eigen.emplace_back(pt.x, pt.y);
  }
  
  // Get local reference in Eigen format
  std::vector<Eigen::Vector2d> local_ref_eigen =
      _prepareReference(current_state, global_waypoints_eigen);
      
  // Convert local_ref_eigen to std::vector<Point2D>
  std::vector<Point2D> local_ref;
  local_ref.reserve(local_ref_eigen.size());
  for (const auto &pt : local_ref_eigen) {
    local_ref.emplace_back(pt.x(), pt.y());
  }
  
  // Garantir velocidade mínima para estabilidade
  double current_velocity = std::max(0.5, current_state.velocity);
  
  // Calcular polinômio da referência local
  std::vector<double> ref_x, ref_y;
  for (const auto& pt : local_ref) {
    ref_x.push_back(pt.x);
    ref_y.push_back(pt.y);
  }
  std::vector<double> poly_coeffs;
  if (ref_x.size() >= 2) {
    // poly_coeffs = Polyfitter::polyfit(ref_x, ref_y, 2);
    // Suponha polyfit disponível
  }
  double f0 = 0.0, psides0 = 0.0;
  if (!poly_coeffs.empty()) {
    for (size_t i = 0; i < poly_coeffs.size(); ++i)
      f0 += poly_coeffs[i] * std::pow(0.0, poly_coeffs.size() - 1 - i);
    double df0 = 0.0;
    for (size_t i = 0; i < poly_coeffs.size() - 1; ++i)
      df0 += (poly_coeffs.size() - 1 - i) * poly_coeffs[i] * std::pow(0.0, poly_coeffs.size() - 2 - i);
    psides0 = std::atan(df0);
  }
  double cte0 = f0 - 0.0;
  double epsi0 = current_state.yaw - psides0;

  // Tratar latência
  double latency = 0.1;
  double steer0 = 0.0, throttle0 = 0.0;
  std::vector<double> state_with_latency = _optimizer._predictStateWithLatency(
      0.0, 0.0, current_state.yaw, current_velocity, throttle0, steer0, latency);
  state_with_latency.push_back(cte0);
  state_with_latency.push_back(epsi0);

  auto [throttle, steer] = _optimizer.solve(
      state_with_latency[0], state_with_latency[1], state_with_latency[2],
      state_with_latency[3], local_ref, lane_info);
  
  return _mapCommandsToHardware(throttle, steer);
}

ControlCommand MPCPlanner::_mapCommandsToHardware(double throttle, double steer) const {
  // Mapear throttle para o range do seu hardware
  // Exemplo: converter de [-1, 1] para [1000, 2000] PWM se necessário
  double mapped_throttle = throttle;
  
  // Aplicar curva de resposta não-linear se necessário
  if (throttle > 0) {
    mapped_throttle = std::min(1.0, throttle * 1.2);  // Aumentar sensibilidade
  }
  
  // Mapear steering com possível offset de calibração
  double mapped_steer = steer;
  
  // Aplicar deadzone e limitação
  if (std::abs(mapped_steer) < 0.03) {
    mapped_steer = 0.0;
  }
  
  mapped_steer = std::max(MPCConfig::steering_limits[0], 
                         std::min(MPCConfig::steering_limits[1], mapped_steer));
  
  return ControlCommand{mapped_throttle, mapped_steer};
}

std::vector<Eigen::Vector2d> MPCPlanner::_prepareReference(
    const VehicleState &state,
    const std::vector<Eigen::Vector2d> &global_waypoints) const {

  std::vector<Eigen::Vector2d> local_points;
  const double cos_yaw = cos(-state.yaw);  // Negativo para transformação inversa
  const double sin_yaw = sin(-state.yaw);

  for (const auto &wp : global_waypoints) {
    // 1. Translação (mover origem para posição do veículo)
    double dx = wp.x() - state.x;
    double dy = wp.y() - state.y;

    // 2. Rotação (transformar para referencial do veículo)
    double local_x = dx * cos_yaw - dy * sin_yaw;
    double local_y = dx * sin_yaw + dy * cos_yaw;

    // 3. Filtrar pontos atrás do veículo (x < 0)
    if (local_x > 0.0) {
      local_points.emplace_back(local_x, local_y);
    }

    // Limita ao horizonte
    if (local_points.size() >= static_cast<size_t>(_config.horizon))
      break;
  }

  // Se não temos pontos suficientes, gerar referência reta
  if (local_points.size() < 3) {
    local_points.clear();
    for (int i = 1; i <= _config.horizon; ++i) {
      local_points.emplace_back(i * 2.0, 0.0);  // Pontos a cada 2m à frente
    }
  }

  return local_points;
}

std::vector<Point2D>
MPCPlanner::convertImagePointsToWorld(const std::vector<int> &center_x,
                                      const std::vector<int> &center_y,
                                      const VehicleTransform &vehicle_transform,
                                      int img_width, int img_height) const {
  std::vector<Point2D> waypoints_world;
  if (center_y.empty())
    return waypoints_world;

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
    if (idx < 0)
      break;

    int x_img = center_x[idx];
    int y_img = center_y[idx];

    double distance_ahead = (img_height - y_img) * escala_m_por_pixel;
    double lateral_offset = (center_x_img - x_img) * escala_m_por_pixel;

    double cos_yaw = std::cos(vehicle_transform.yaw);
    double sin_yaw = std::sin(vehicle_transform.yaw);

    double world_x = vehicle_transform.x + distance_ahead * cos_yaw -
                     lateral_offset * sin_yaw;
    double world_y = vehicle_transform.y + distance_ahead * sin_yaw +
                     lateral_offset * cos_yaw;

    waypoints_world.emplace_back(world_x, world_y);
  }

  return waypoints_world;
}