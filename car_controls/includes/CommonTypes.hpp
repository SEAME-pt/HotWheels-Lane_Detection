#ifndef COMMON_TYPES_HPP
#define COMMON_TYPES_HPP

#include <vector>
#include <array>
#include <string>

// Estrutura básica para coordenadas 2D
struct Point2D {
    double x, y;
    Point2D(double x_ = 0.0, double y_ = 0.0) : x(x_), y(y_) {}
};

// Estado atual do veículo
struct VehicleState {
    double x, y, yaw, velocity;
};

// Transformação do veículo (posição e orientação)
struct VehicleTransform {
    double x, y, yaw;
};

// Comando de controle (throttle, steer)
struct ControlCommand {
    double throttle, steer;
};

// Informações da faixa (para controle de faixa)
struct LaneInfo {
    double left_boundary, right_boundary, center_line;
    double lateral_offset, yaw_error;
    LaneInfo(double lo = 0.0, double ye = 0.0)
        : left_boundary(0.0), right_boundary(0.0), center_line(0.0), lateral_offset(lo), yaw_error(ye) {}
    LaneInfo() : left_boundary(0.0), right_boundary(0.0), center_line(0.0), lateral_offset(0.0), yaw_error(0.0) {}
};

// Outros tipos comuns podem ser adicionados aqui

#endif // COMMON_TYPES_HPP
