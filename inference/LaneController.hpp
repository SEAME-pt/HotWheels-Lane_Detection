#ifndef LANE_CONTROLLER_HPP
#define LANE_CONTROLLER_HPP

#include "LaneAnalyzer.hpp"

class LaneController {
public:
    LaneController(float kp_offset = 0.005f, float ka_angle = 0.02f);

    // Returns a steering value between -1 and 1
    float computeSteering(const LaneMetrics& metrics);

private:
    float kp;  // Gain for lateral offset
    float ka;  // Gain for heading angle
};

#endif
