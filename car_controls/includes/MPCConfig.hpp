#pragma once

#include <array>

struct MPCConfig
{
	// Parâmetros do MPC ajustados para Jetracer (26cmX21cm, entre-eixo 15cm)
	static constexpr double dt = 0.05;		  // Menor timestep para melhor controle em escala pequena
	static constexpr int horizon = 8;		  // Horizonte menor para pista de 12m²
	static constexpr double wheelbase = 0.15; // Entre-eixo real: 15cm

	// Pesos padrão ajustados para escala pequena (para retas)
	static constexpr double w_cte_straight = 12.0;		 // Maior peso para tracking de trajetória
	static constexpr double w_etheta_straight = 6.0;	 // Mais controle de orientação
	static constexpr double w_vel_straight = 0.8;		 // Menor peso na velocidade
	static constexpr double w_throttle_straight = 0.2;	 // Suavidade de throttle
	static constexpr double w_steer_straight = 0.15;	 // Suavidade de steering
	static constexpr double target_speed_straight = 1.0; // Velocidade reduzida: 1 m/s (3.6 km/h)

	// Pesos para curvas ajustados para Jetracer
	static constexpr double w_cte_curve = 20.0;				  // Muito importante para curvas apertadas
	static constexpr double w_etheta_curve = 12.0;			  // Controle crítico de orientação
	static constexpr double w_vel_curve = 0.2;				  // Velocidade muito controlada em curvas
	static constexpr double w_throttle_curve = 0.1;			  // Throttle suave
	static constexpr double w_steer_curve = 0.05;			  // Steering muito suave
	static constexpr double target_speed_curve_base = 0.3;	  // Velocidade mínima em curvas: 0.3 m/s
	static constexpr double target_speed_curve_factor = 15.0; // Fator de redução por curvatura

	// Restrições de controle ajustadas para servo motor real
	static constexpr double max_steer = 0.35;	// ±20 graus (servo limitado)
	static constexpr double max_throttle = 0.6; // 60% do throttle máximo para segurança
	static constexpr double max_brake = 0.3;	// Frenagem suave
	static constexpr int max_iter = 300;		// Menos iterações para tempo real

	// Limites ajustados para hardware real
	static constexpr std::array<double, 2> steering_limits = {-0.35, 0.35}; // ±20 graus
	static constexpr std::array<double, 2> throttle_limits = {-0.3, 0.6};	// Reverse limitado, forward controlado

	// Parâmetros específicos para Jetracer
	static constexpr double desired_speed = 0.8;	  // Velocidade de cruzeiro: 0.8 m/s
	static constexpr double control_frequency = 0.05; // 20 Hz de controle
	static constexpr int camera_width = 320;		  // Resolução reduzida para performance
	static constexpr int camera_height = 240;		  // Resolução reduzida para performance

	// Parâmetros de segurança para ambiente real
	static constexpr double min_safe_distance = 0.2;		// 20cm de distância mínima de obstáculos
	static constexpr double emergency_brake_distance = 0.1; // 10cm para freio de emergência
	static constexpr double max_lateral_acceleration = 2.0; // 2 m/s² máximo lateral
	static constexpr double track_width_estimate = 0.4;		// Largura estimada da pista: 40cm
};