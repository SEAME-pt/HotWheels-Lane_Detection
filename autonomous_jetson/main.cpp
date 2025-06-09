#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <stdexcept>
#include "CommonTypes.hpp"
#include "MPCPlanner.hpp"
#include "Polyfitter.hpp"

// Placeholder para a classe de controle do carro (substituir por CarControls)
class CarControls {
public:
    // Métodos placeholder (serão implementados conforme sua classe CarControls)
    void applyControl(double throttle, double steer, double brake = 0.0) {
        std::cout << "[CarControls] Applying control: throttle=" << throttle 
                  << ", steer=" << steer << ", brake=" << brake << std::endl;
    }
    void updateSpectator() {
        std::cout << "[CarControls] Updating spectator camera" << std::endl;
    }
    void cleanup() {
        std::cout << "[CarControls] Cleaning up" << std::endl;
    }
    // ... outros métodos conforme necessário
};

// Placeholder para o detector de faixas (substituir conforme sua implementação)
class KerasLaneDetector {
public:
    KerasLaneDetector(const std::string& model_path) {
        std::cout << "[KerasLaneDetector] Initialized with model: " << model_path << std::endl;
    }
    cv::Mat predict(const cv::Mat& img) {
        // Simulação: retorna uma máscara binária
        cv::Mat mask(img.size(), CV_8UC1, cv::Scalar(0));
        // Exemplo: retorna uma linha central simples
        cv::line(mask, cv::Point(0, mask.rows-1), cv::Point(mask.cols/2, mask.rows/2), cv::Scalar(255), 2);
        return mask;
    }
};

// Função de debug
void print_debug_info(const std::string& text) {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::cout << "[DEBUG] " << std::ctime(&now_c) << " - " << text << std::endl;
}

// Placeholder para a função de extração de trajetória central
std::pair<std::vector<double>, std::vector<double>> extract_center_curve_with_lanes_improved(const cv::Mat& binary_mask) {
    // Simulação: retorna uma trajetória central simples
    std::vector<double> x, y;
    for (int i = 0; i < 10; ++i) {
        x.push_back(binary_mask.cols * 0.5 + std::sin(i * 0.3) * 50);
        y.push_back(binary_mask.rows - i * 20);
    }
    return {x, y};
}

// Placeholder para a função de extração de informações da faixa
LaneInfo extract_lane_info(const cv::Mat& mask, const std::pair<std::vector<double>, std::vector<double>>& center_curve) {
    if (center_curve.first.empty()) return LaneInfo();
    // Simulação: calcula desvio lateral simples
    int img_width = mask.cols;
    int car_center_x = img_width / 2;
    int idx = 0; // Ponto mais próximo do carro (maior y)
    double lane_center_x = center_curve.first[idx];
    double real_height_m = 8.0;
    double escala_m_por_pixel = real_height_m / mask.rows;
    double lateral_offset = (car_center_x - lane_center_x) * escala_m_por_pixel;

    // Simulação: erro de orientação simples
    double yaw_error = 0.0;
    if (center_curve.first.size() > 5) {
        double x1 = center_curve.first[center_curve.first.size()-5];
        double y1 = center_curve.second[center_curve.first.size()-5];
        double x2 = center_curve.first.back();
        double y2 = center_curve.second.back();
        if (x2 != x1)
            yaw_error = std::atan2(y2 - y1, x2 - x1) - M_PI/2;
    }
    return LaneInfo(lateral_offset, yaw_error);
}

// Placeholder para a função de cálculo de curvatura
double calculate_curve_curvature(const std::vector<double>& x_coords, const std::vector<double>& y_coords) {
    if (x_coords.size() < 5) return 0.0;
    // Simulação: calcula curvatura simples (pode ser substituída pela sua função)
    return 0.0;
}

// Placeholder para a função de conversão de pontos da imagem para o mundo
std::vector<Point2D> convert_image_points_to_world(
    const std::pair<std::vector<double>, std::vector<double>>& center_curve,
    CarControls& car_controls) {
    // Simulação: retorna waypoints simples à frente do carro
    std::vector<Point2D> waypoints;
    for (size_t i = 0; i < center_curve.first.size(); ++i) {
        waypoints.emplace_back(center_curve.first[i], center_curve.second[i]);
    }
    return waypoints;
}

// Placeholder para a função de obtenção de waypoints à frente (fallback)
std::vector<Point2D> get_waypoints_ahead(double distance, int count, const VehicleState& state) {
    std::vector<Point2D> waypoints;
    for (int i = 1; i <= count; ++i) {
        double dx = i * distance / count;
        waypoints.emplace_back(state.x + dx, state.y);
    }
    return waypoints;
}

// Placeholder para a função de obtenção do estado do veículo
VehicleState get_vehicle_state(CarControls& car_controls) {
    // Simulação: retorna um estado simples
    static double x = 0.0, y = 0.0, yaw = 0.0, velocity = 1.0;
    x += 0.1;
    y += 0.05 * std::sin(x * 0.1);
    yaw = 0.05 * std::sin(x * 0.2);
    velocity = 2.0 + 0.5 * std::sin(x * 0.1);
    return VehicleState{x, y, yaw, velocity};
}

// Placeholder para a função de obtenção da imagem da câmera
cv::Mat get_camera_image(CarControls& car_controls) {
    // Simulação: retorna uma imagem preta com uma linha central
    cv::Mat img(640, 640, CV_8UC1, cv::Scalar(0));
    cv::line(img, cv::Point(0, img.rows-1), cv::Point(img.cols/2, img.rows/2), cv::Scalar(255), 2);
    return img;
}

// Função principal
int main() {
    try {
        // Configurações
        const double CONTROL_RATE = 20.0; // Hz
        const double CONTROL_PERIOD = 1.0 / CONTROL_RATE; // s
        bool DEBUG = true;

        // Inicialização dos sistemas
        CarControls car_controls;
        KerasLaneDetector detector("models/lane_detector_combined_v2.keras");
        MPCConfig mpc_config;
        MPCOptimizer mpc_optimizer(mpc_config);
        MPCPlanner mpc_planner(mpc_config, mpc_optimizer);
        Polyfitter polyfitter;

        // Controle de posições anteriores para detecção de travamento
        std::vector<std::pair<double, double>> last_positions;
        const int max_position_history = 5;
        bool is_in_recovery = false;
        auto recovery_end_time = std::chrono::system_clock::now();
        auto last_control_time = std::chrono::system_clock::now();

        // Impulso inicial
        car_controls.applyControl(0.3, 0.0, 0.0);
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));

        // Loop principal de controle
        while (true) {
            auto now = std::chrono::system_clock::now();
            if (std::chrono::duration<double>(now - last_control_time).count() < CONTROL_PERIOD)
                continue;
            last_control_time = now;

            // Obter estado atual do veículo
            VehicleState current_state = get_vehicle_state(car_controls);
            if (current_state.velocity < 0.1) {
                // Controle manual de teste (carro parado)
                print_debug_info("Carro parado, aplicando controle manual de teste");
                car_controls.applyControl(0.4, 0.0, 0.0);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }

            print_debug_info("Vehicle state: Position (" + std::to_string(current_state.x) + ", " +
                            std::to_string(current_state.y) + "), Velocity: " +
                            std::to_string(current_state.velocity) + " m/s, Yaw: " +
                            std::to_string(current_state.yaw));

            // Obter imagem da câmera
            cv::Mat image = get_camera_image(car_controls);
            LaneInfo lane_info(0.0, 0.0);
            std::vector<Point2D> waypoints;
            bool waypoints_from_lanes = false;

            if (!image.empty()) {
                try {
                    cv::Mat binary_mask = detector.predict(image);
                    if (!binary_mask.empty() && cv::countNonZero(binary_mask) > 0) {
                        auto center_curve = extract_center_curve_with_lanes_improved(binary_mask);
                        if (!center_curve.first.empty()) {
                            // Extrair informações da faixa
                            lane_info = extract_lane_info(binary_mask, center_curve);
                            // Converter para waypoints do mundo
                            waypoints = convert_image_points_to_world(center_curve, car_controls);
                            if (waypoints.size() > 5) {
                                waypoints_from_lanes = true;
                                print_debug_info("Usando " + std::to_string(waypoints.size()) + " waypoints da trajetória central");
                            } else {
                                // Fallback: waypoints simples à frente
                                waypoints = get_waypoints_ahead(2.0, 20, current_state);
                                print_debug_info("Usando waypoints simples à frente");
                            }
                        } else {
                            print_debug_info("Polyfit não retornou resultado válido");
                            waypoints = get_waypoints_ahead(2.0, 20, current_state);
                        }
                    } else {
                        print_debug_info("Máscara binária inválida");
                        waypoints = get_waypoints_ahead(2.0, 20, current_state);
                    }
                } catch (const std::exception& e) {
                    print_debug_info("Erro no processamento da imagem: " + std::string(e.what()));
                    waypoints = get_waypoints_ahead(2.0, 20, current_state);
                }
            } else {
                waypoints = get_waypoints_ahead(2.0, 20, current_state);
            }

            // Validar waypoints
            if (waypoints.empty()) {
                print_debug_info("WARNING: Nenhum waypoint disponível!");
                waypoints.emplace_back(current_state.x + 5, current_state.y);
            } else {
                // Verificar distância dos waypoints
                double avg_distance = 0.0;
                for (int i = 0; i < std::min(3, (int)waypoints.size()); ++i) {
                    double dx = waypoints[i].x - current_state.x;
                    double dy = waypoints[i].y - current_state.y;
                    avg_distance += std::sqrt(dx*dx + dy*dy);
                }
                avg_distance /= std::min(3, (int)waypoints.size());
                print_debug_info("Distância média dos waypoints: " + std::to_string(avg_distance) + "m");
                if (avg_distance > 20.0) {
                    print_debug_info("Waypoints muito distantes, usando waypoints simples");
                    waypoints = get_waypoints_ahead(2.0, 3, current_state);
                }
            }

            // Gerar comandos de controle
            try {
                // Converter waypoints para o formato esperado pelo MPCPlanner
                std::vector<Eigen::Vector2d> global_waypoints;
                for (const auto& wp : waypoints) {
                    global_waypoints.emplace_back(wp.x, wp.y);
                }
                ControlCommand control = mpc_planner.plan(current_state, global_waypoints, &lane_info);

                print_debug_info("MPC output - Throttle: " + std::to_string(control.throttle) +
                                ", Steer: " + std::to_string(control.steer));

                // Aplicar controle adaptativo baseado na curvatura
                double throttle_value = control.throttle;
                double steer_value = control.steer;
                bool is_curve = false; // Simulação: pode ser calculado conforme sua lógica

                // Forçar throttle mínimo adaptativo
                if (std::abs(throttle_value) < 0.1) {
                    if (is_curve) {
                        throttle_value = 0.2;
                        print_debug_info("Forçando throttle mínimo para CURVA: " + std::to_string(throttle_value));
                    } else {
                        throttle_value = 0.3;
                        print_debug_info("Forçando throttle mínimo para RETA: " + std::to_string(throttle_value));
                    }
                }

                // Aplicar limites adaptativos
                double max_throttle = is_curve ? 0.25 : 0.4;
                double max_steer = is_curve ? 0.4 : 0.3;
                throttle_value = std::clamp(throttle_value, 0.0, max_throttle);
                steer_value = std::clamp(steer_value, -max_steer, max_steer);

                print_debug_info("Controle adaptativo - Throttle: " + std::to_string(throttle_value) +
                                ", Steer: " + std::to_string(steer_value) +
                                ", Tipo: " + (is_curve ? "CURVA" : "RETA"));

                car_controls.applyControl(throttle_value, steer_value, 0.0);
            } catch (const std::exception& e) {
                print_debug_info("Erro no MPC: " + std::string(e.what()));
                // Controle de emergência - parar o carro
                car_controls.applyControl(0.0, 0.0, 0.5);
            }

            // Atualizar câmera do espectador
            car_controls.updateSpectator();

            // Detecção de travamento
            auto current_pos = std::make_pair(current_state.x, current_state.y);
            last_positions.push_back(current_pos);
            if (last_positions.size() > max_position_history)
                last_positions.erase(last_positions.begin());

            if (!is_in_recovery && last_positions.size() >= 3) {
                double total_distance = 0.0;
                for (size_t i = 1; i < last_positions.size(); ++i) {
                    double dx = last_positions[i].first - last_positions[i-1].first;
                    double dy = last_positions[i].second - last_positions[i-1].second;
                    total_distance += std::sqrt(dx*dx + dy*dy);
                }
                double avg_movement = total_distance / (last_positions.size() - 1);
                bool is_stuck = (avg_movement < 0.05 && throttle_value > 0.2 && current_state.velocity < 0.5);

                if (is_stuck) {
                    print_debug_info("Car is stuck! Avg movement: " + std::to_string(avg_movement) +
                                    "m, Speed: " + std::to_string(current_state.velocity) + "m/s");
                    is_in_recovery = true;
                    recovery_end_time = now + std::chrono::seconds(2);
                    print_debug_info("Applying gentle reverse for 2 seconds...");
                }
            }

            // Modo de recuperação
            if (is_in_recovery) {
                if (now < recovery_end_time) {
                    double progress = std::chrono::duration<double>(now - (recovery_end_time - std::chrono::seconds(2))).count() / 2.0;
                    double reverse_throttle = -0.4 * (1.0 - progress * 0.5);
                    double reverse_steer = -steer_value * 0.7;
                    if (progress > 0.7)
                        reverse_steer *= (1.0 - (progress - 0.7) / 0.3);
                    print_debug_info("Recovery - progress: " + std::to_string(progress) +
                                    ", reverse: " + std::to_string(reverse_throttle) +
                                    ", steer: " + std::to_string(reverse_steer));
                    car_controls.applyControl(reverse_throttle, reverse_steer, 0.0);
                } else {
                    is_in_recovery = false;
                    print_debug_info("Recovery completed, resuming normal control");
                    last_positions.clear();
                }
            }

            // Dormir para manter a taxa de controle
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(CONTROL_PERIOD * 1000)));
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return 1;
    }
    return 0;
}
