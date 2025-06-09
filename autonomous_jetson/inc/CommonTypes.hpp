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
};

// Outros tipos comuns podem ser adicionados aqui

#endif // COMMON_TYPES_HPP
