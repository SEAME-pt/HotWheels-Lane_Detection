#pragma once

#include <array>

struct MPCConfig {
    // Parâmetros do MPC
    static constexpr double dt = 0.1;
    static constexpr int horizon = 10;
    static constexpr double wheelbase = 2.7;

    // Pesos padrão (para retas)
    static constexpr double w_cte_straight = 8.0;
    static constexpr double w_etheta_straight = 4.0;
    static constexpr double w_vel_straight = 1.5;
    static constexpr double w_throttle_straight = 0.1;
    static constexpr double w_steer_straight = 0.1;
    static constexpr double target_speed_straight = 6.0;

    // Pesos para curvas
    static constexpr double w_cte_curve = 15.0;
    static constexpr double w_etheta_curve = 8.0;
    static constexpr double w_vel_curve = 0.3;
    static constexpr double w_throttle_curve = 0.05;
    static constexpr double w_steer_curve = 0.02;
    static constexpr double target_speed_curve_base = 2.0;
    static constexpr double target_speed_curve_factor = 20.0;

    // Restrições de controle
    static constexpr double max_steer = 0.52;
    static constexpr double max_throttle = 0.8;
    static constexpr double max_brake = 0.5;
    static constexpr int max_iter = 500;

    static constexpr std::array<double, 2> steering_limits = {-0.52, 0.52};
    static constexpr std::array<double, 2> throttle_limits = {0.0, 0.8};

    // Outros parâmetros
    static constexpr double desired_speed = 5.0; // Opcional, pode ser usado como referência
    static constexpr double control_frequency = 0.1;
    static constexpr int camera_width = 320;
    static constexpr int camera_height = 320;
};