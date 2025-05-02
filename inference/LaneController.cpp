#include "LaneController.hpp"
#include <algorithm>
#include <cmath>

LaneController::LaneController(float kp_offset, float ka_angle)
    : kp(kp_offset), ka(ka_angle) {}

/* float LaneController::computeSteering(const LaneMetrics& metrics) {
    if (!metrics.valid) return 0.0f;

    // Steering logic (simple weighted sum)
    float steering = kp * metrics.lateralOffset + ka * metrics.headingAngleDeg;

    // Clamp to [-1, 1] range
    steering = std::clamp(steering, -1.0f, 1.0f);
    return steering;
} */
