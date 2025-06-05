#pragma once

#include <array>

struct MPCConfig {
    // MPC parameters
    static constexpr double dt = 0.1;
    static constexpr int horizon = 10;
    static constexpr double wheelbase = 2.7;

    // Cost function weights
    static constexpr double w_cte = 10.0;
    static constexpr double w_etheta = 5.0;
    static constexpr double w_vel = 1.0;
    static constexpr double w_steer = 0.1;
    static constexpr double w_accel = 0.05;
    static constexpr double w_steer_rate = 0.3;
    static constexpr double w_lane = 8.0;

    // Control constraints
    static constexpr double max_steer = 0.52;
    static constexpr double max_throttle = 0.8;
    static constexpr double max_brake = 0.5;
    static constexpr int max_iter = 500;

    static constexpr std::array<double, 2> steering_limits = {-0.52, 0.52};
    static constexpr std::array<double, 2> throttle_limits = {0.0, 0.8};

    // Control loop frequency
    static constexpr double control_frequency = 0.1;

    // Camera config (suggestion)
    static constexpr int camera_width = 320;
    static constexpr int camera_height = 320;
};