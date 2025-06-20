#include "MPCOptimizer.hpp"
#include "Polyfitter.hpp"

MPCOptimizer::MPCOptimizer(void)
{
    MPCConfig mpc_config;
    _mpc = mpc_config;
}

MPCOptimizer::MPCOptimizer(const MPCConfig &config)
    : _mpc(config), _current_state(), _current_reference(), _current_lane_info(nullptr)
{
}

MPCOptimizer::MPCOptimizer(const MPCOptimizer &origin)
{
    *this = origin;
}

MPCOptimizer &MPCOptimizer::operator=(const MPCOptimizer &origin)
{
    if (this != &origin)
        *this = origin;
    return *this;
}

MPCOptimizer::~MPCOptimizer(void)
{
}

static double costWrapper(unsigned n, const double *x, double *grad, void *data)
{
    (void)grad; // Suppress unused parameter warning
    MPCOptimizer *self = static_cast<MPCOptimizer *>(data);
    // Convertendo x para std::vector<double>
    std::vector<double> u(x, x + n);
    // Chamando a função de custo da classe
    return self->_costFunction(u, self->getCurrentState(), self->getCurrentReference(),
                               self->getCurrentLaneInfo());
}

std::pair<double, double> MPCOptimizer::solve(double x0, double y0, double yaw0, double v0,
                                              const std::vector<Point2D> &reference,
                                              const LaneInfo *lane_info)
{
    // Calcular polinômio da trajetória de referência usando Polyfitter
    Polyfitter polyfitter;
    std::vector<double> poly_coeffs = polyfitter.getPolynomialCoeffs(reference);

    // Calcular cte e epsi iniciais
    double cte0 = 0.0, epsi0 = 0.0;
    if (!poly_coeffs.empty())
    {
        cte0 = polyfitter.calculateCTE(poly_coeffs, x0, y0);
        epsi0 = polyfitter.calculateEPSI(poly_coeffs, x0, yaw0);
    }

    // Tratar latência
    double latency = 0.1;
    double steer0 = 0.0, throttle0 = 0.0;
    std::vector<double> state_with_latency =
        _predictStateWithLatency(x0, y0, yaw0, v0, throttle0, steer0, latency);
    // Adicionar cte e epsi ao estado
    state_with_latency.push_back(cte0);
    state_with_latency.push_back(epsi0);

    _current_state = state_with_latency;
    _current_reference = reference;
    _current_lane_info = lane_info;
    _current_poly_coeffs = poly_coeffs; // Store coefficients

    // Configuração do otimizador (melhor para MPC)
    nlopt::opt optimizer(nlopt::LD_SLSQP, 2 * MPCConfig::horizon); // SLSQP é melhor para MPC

    // Limites das variáveis de controle
    std::vector<double> lb(2 * MPCConfig::horizon);
    std::vector<double> ub(2 * MPCConfig::horizon);

    for (int i = 0; i < MPCConfig::horizon; ++i)
    {
        lb[2 * i] = MPCConfig::throttle_limits[0];
        ub[2 * i] = MPCConfig::throttle_limits[1];
        lb[2 * i + 1] = MPCConfig::steering_limits[0];
        ub[2 * i + 1] = MPCConfig::steering_limits[1];
    }
    optimizer.set_lower_bounds(lb);
    optimizer.set_upper_bounds(ub);

    // Inicialização mais inteligente
    std::vector<double> u0(2 * MPCConfig::horizon, 0.0);
    for (int i = 0; i < MPCConfig::horizon; ++i)
    {
        u0[2 * i] = 0.3;     // throttle moderado
        u0[2 * i + 1] = 0.0; // steering neutro
    }

    optimizer.set_min_objective(costWrapper, this);

    // Configurações mais robustas
    optimizer.set_maxeval(MPCConfig::max_iter);
    optimizer.set_xtol_rel(1e-4); // Menos restritivo para tempo real
    optimizer.set_ftol_rel(1e-4);
    optimizer.set_maxtime(0.05); // Timeout para garantir tempo real

    double min_cost;
    try
    {
        nlopt::result result = optimizer.optimize(u0, min_cost);
        if (result < 0)
        {
            // Fallback: retornar controles seguros
            return {0.2, 0.0};
        }
    }
    catch (const std::exception &e)
    {
        // Fallback: retornar controles seguros
        return {0.2, 0.0};
    }

    return {u0[0], u0[1]}; // Retorna primeiro par de controles
}

double MPCOptimizer::_normalizeAngle(double angle) const
{
    while (angle > M_PI)
        angle -= 2.0 * M_PI;
    while (angle < -M_PI)
        angle += 2.0 * M_PI;
    return angle;
}

double MPCOptimizer::_calculatePathCurvature(const std::vector<Point2D> &reference) const
{
    if (reference.size() < 3)
    {
        return 0.0;
    }

    // Usar os primeiros 3 pontos
    const Point2D &p1 = reference[0];
    const Point2D &p2 = reference[1];
    const Point2D &p3 = reference[2];

    // Calcular distâncias entre pontos
    double a = std::sqrt(std::pow(p2.x - p1.x, 2) + std::pow(p2.y - p1.y, 2));
    double b = std::sqrt(std::pow(p3.x - p2.x, 2) + std::pow(p3.y - p2.y, 2));
    double c = std::sqrt(std::pow(p3.x - p1.x, 2) + std::pow(p3.y - p1.y, 2));

    if (a * b * c == 0.0)
    {
        return 0.0;
    }

    // Semi-perímetro
    double s = (a + b + c) / 2.0;
    // Área do triângulo (fórmula de Heron)
    double area = std::sqrt(std::max(0.0, s * (s - a) * (s - b) * (s - c)));

    // Curvatura = 4 * área / (a * b * c)
    double curvature = (a * b * c > 0.0) ? 4.0 * area / (a * b * c) : 0.0;

    return curvature;
}

double MPCOptimizer::_costFunction(const std::vector<double> &u, const std::vector<double> &state,
                                   const std::vector<Point2D> &reference,
                                   const LaneInfo *lane_info) const
{
    (void)lane_info;
    double cost = 0.0;
    // Estado inicial
    double x = state[0], y = state[1], yaw = state[2], v = state[3], cte = state[4],
           epsi = state[5];

    double curvature = _calculatePathCurvature(reference);
    bool is_curve = std::abs(curvature) > 0.05;

    // Seleção de pesos baseada na curvatura
    double w_cte, w_etheta, w_velocity, w_throttle, w_steer, target_speed;
    if (is_curve)
    {
        w_cte = _mpc.w_cte_curve;
        w_etheta = _mpc.w_etheta_curve;
        w_velocity = _mpc.w_vel_curve;
        w_throttle = _mpc.w_throttle_curve;
        w_steer = _mpc.w_steer_curve;
        target_speed = std::max(_mpc.target_speed_curve_base,
                                _mpc.target_speed_straight -
                                    std::abs(curvature) * _mpc.target_speed_curve_factor);
    }
    else
    {
        w_cte = _mpc.w_cte_straight;
        w_etheta = _mpc.w_etheta_straight;
        w_velocity = _mpc.w_vel_straight;
        w_throttle = _mpc.w_throttle_straight;
        w_steer = _mpc.w_steer_straight;
        target_speed = _mpc.target_speed_straight;
    }

    for (int t = 0; t < MPCConfig::horizon; ++t)
    {
        double throttle = u[2 * t];
        double steer = u[2 * t + 1];
        // Modelo 6 estados usando coeficientes armazenados
        _kinematicModel(x, y, yaw, v, cte, epsi, throttle, steer, _current_poly_coeffs);
        // Penalizar cte e epsi explicitamente
        cost += w_cte * cte * cte;
        cost += w_etheta * epsi * epsi;

        // 3. Velocity Error
        double v_error = v - target_speed;
        cost += w_velocity * v_error * v_error;

        // 4. Actuator Use (minimize control effort)
        cost += w_throttle * throttle * throttle;
        cost += w_steer * steer * steer;

        // 5. Actuator Rate (smoothness)
        if (t > 0)
        {
            double prev_throttle = u[2 * (t - 1)];
            double prev_steer = u[2 * (t - 1) + 1];
            double throttle_rate = throttle - prev_throttle;
            double steer_rate = steer - prev_steer;

            cost += 0.1 * throttle_rate * throttle_rate; // Suavidade do throttle
            cost += 0.5 * steer_rate * steer_rate;       // Suavidade do steering
        }
    }

    return cost;
}

// Atualize o modelo cinemático para 6 estados
void MPCOptimizer::_kinematicModel(double &x, double &y, double &yaw, double &v, double &cte,
                                   double &epsi, double throttle, double steer,
                                   const std::vector<double> &poly_coeffs) const
{
    double f = 0.0, psides = 0.0;
    if (!poly_coeffs.empty())
    {
        for (size_t i = 0; i < poly_coeffs.size(); ++i)
            f += poly_coeffs[i] * std::pow(x, poly_coeffs.size() - 1 - i);
        double df = 0.0;
        for (size_t i = 0; i < poly_coeffs.size() - 1; ++i)
            df += (poly_coeffs.size() - 1 - i) * poly_coeffs[i] *
                  std::pow(x, poly_coeffs.size() - 2 - i);
        psides = std::atan(df);
    }
    x += v * std::cos(yaw) * MPCConfig::dt;
    y += v * std::sin(yaw) * MPCConfig::dt;
    yaw += (v / MPCConfig::wheelbase) * std::tan(steer) * MPCConfig::dt;
    v += throttle * MPCConfig::dt;
    cte = f - y + v * std::sin(epsi) * MPCConfig::dt;
    epsi = yaw - psides + (v / MPCConfig::wheelbase) * std::tan(steer) * MPCConfig::dt;
    yaw = _normalizeAngle(yaw);
    epsi = _normalizeAngle(epsi);
    v = std::max(0.0, std::min(v, 10.0));
}

double MPCOptimizer::_calculateCurveCurvature(const std::vector<double> &x_coords,
                                              const std::vector<double> &y_coords) const
{
    if (x_coords.size() < 3 || y_coords.size() < 3)
        return 0.0;

    std::vector<double> dx(x_coords.size()), dy(y_coords.size());
    std::vector<double> ddx(x_coords.size()), ddy(y_coords.size());

    // Cálculo das derivadas (gradiente simples)
    for (size_t i = 1; i < x_coords.size() - 1; ++i)
    {
        dx[i] = (x_coords[i + 1] - x_coords[i - 1]) / 2.0;
        dy[i] = (y_coords[i + 1] - y_coords[i - 1]) / 2.0;
    }
    dx[0] = dx[1];
    dx.back() = dx[dx.size() - 2];
    dy[0] = dy[1];
    dy.back() = dy[dy.size() - 2];

    for (size_t i = 1; i < dx.size() - 1; ++i)
    {
        ddx[i] = (dx[i + 1] - dx[i - 1]) / 2.0;
        ddy[i] = (dy[i + 1] - dy[i - 1]) / 2.0;
    }
    ddx[0] = ddx[1];
    ddx.back() = ddx[ddx.size() - 2];
    ddy[0] = ddy[1];
    ddy.back() = ddy[ddy.size() - 2];

    std::vector<double> curvature(x_coords.size());
    for (size_t i = 0; i < x_coords.size(); ++i)
    {
        double numerator = std::abs(dx[i] * ddy[i] - dy[i] * ddx[i]);
        double denom = std::pow(dx[i] * dx[i] + dy[i] * dy[i], 1.5);
        if (denom < 1e-6)
            denom = 1e-6; // evitar divisão por zero
        curvature[i] = numerator / denom;
    }

    // Retorna a média da curvatura
    double sum = std::accumulate(curvature.begin(), curvature.end(), 0.0);
    return sum / curvature.size();
}

std::vector<double> MPCOptimizer::_predictStateWithLatency(double x0, double y0, double yaw0,
                                                           double v0, double throttle, double steer,
                                                           double latency) const
{
    // Prever estado futuro considerando latência
    double x = x0, y = y0, yaw = yaw0, v = v0;
    double cte = 0.0, epsi = 0.0; // Para compatibilidade com novo modelo
    double steps = latency / MPCConfig::dt;

    // Use coeficientes vazios para predição de latência (simplificação)
    std::vector<double> empty_coeffs;

    for (int i = 0; i < (int)steps; ++i)
    {
        _kinematicModel(x, y, yaw, v, cte, epsi, throttle, steer, empty_coeffs);
    }

    return {x, y, yaw, v};
}