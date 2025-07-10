#ifndef COMMON_TYPES_HPP
#define COMMON_TYPES_HPP

#include <array>
#include <string>
#include <vector>

// Estrutura básica para coordenadas 2D
struct Point2D {
		double x, y;
		Point2D (double x_ = 0.0, double y_ = 0.0) : x (x_), y (y_) {}
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

		// Constructors
		ControlCommand () : throttle (0.0), steer (0.0) {}
		ControlCommand (double t, double s) : throttle (t), steer (s) {}
};

// Informações da faixa (para controle de faixa)
struct LaneInfo {
		double left_boundary, right_boundary, center_line;
		double lateral_offset, yaw_error;
		bool isValid; // Adicionado caso precise de verificar se a faixa é válida
		LaneInfo (double lo = 0.0, double ye = 0.0)
		    : left_boundary (0.0), right_boundary (0.0), center_line (0.0), lateral_offset (lo),
		      yaw_error (ye) {}
};

// Diagnósticos do estimador de estado
struct StateEstimatorDiagnostics {
		double position_accuracy;   // Precisão da posição estimada
		double velocity_confidence; // Confiança na velocidade
		double yaw_stability;       // Estabilidade da orientação
		bool sensor_health;         // Estado dos sensores
		double last_update_time;    // Timestamp da última atualização
		int num_sensor_failures;    // Contador de falhas de sensor

		StateEstimatorDiagnostics ()
		    : position_accuracy (0.0), velocity_confidence (0.0), yaw_stability (0.0),
		      sensor_health (true), last_update_time (0.0), num_sensor_failures (0) {}
};

// Outros tipos comuns podem ser adicionados aqui

#endif // COMMON_TYPES_HPP
