#include "car_controls/includes/ControlsManager.hpp"
#include "car_controls/includes/MPCOptimizer.hpp"
#include "car_controls/includes/MPCPlanner.hpp"
#include "car_controls/includes/CommonTypes.hpp"
#include <QApplication>
#include <QTimer>
#include <iostream>
#include <cmath>
#include <chrono>

class MPCIntegratedApp : public QObject {
    Q_OBJECT

private:
    ControlsManager* controls_manager;
    MPCPlanner* mpc_planner;
    QTimer* mpc_timer;
    
    // Estado do MPC
    bool mpc_active = false;
    std::vector<Point2D> recorded_waypoints;
    VehicleState current_state{0.0, 0.0, 0.0, 0.5};
    int step_counter = 0;

public:
    MPCIntegratedApp(int argc, char** argv, QObject* parent = nullptr) 
        : QObject(parent) {
        
        // Inicializar o sistema de controles existente
        controls_manager = new ControlsManager(argc, argv, this);
        
        // Inicializar MPC
        mpc_planner = new MPCPlanner();
        
        // Timer para executar MPC periodicamente
        mpc_timer = new QTimer(this);
        connect(mpc_timer, &QTimer::timeout, this, &MPCIntegratedApp::runMPCStep);
        
        std::cout << "=== MPC Integrated System ===" << std::endl;
        std::cout << "Sistema iniciado com controle manual" << std::endl;
        std::cout << "Comandos disponíveis:" << std::endl;
        std::cout << "- Use joystick para mover e gravar trajetória" << std::endl;
        std::cout << "- Pressione ENTER para alternar Manual/MPC" << std::endl;
        std::cout << "- Pressione 'r' para iniciar/parar gravação" << std::endl;
        std::cout << "- Pressione 'q' para sair" << std::endl;
        
        // Conectar stdin para comandos
        setupKeyboardInput();
    }
    
    ~MPCIntegratedApp() {
        delete mpc_planner;
    }

private slots:
    void runMPCStep() {
        if (!mpc_active || recorded_waypoints.size() < 3) {
            return;
        }
        
        try {
            // Usar waypoints gravados como referência
            LaneInfo lane_info(0.0, 0.0);
            ControlCommand control = mpc_planner->plan(current_state, recorded_waypoints, &lane_info);
            
            // Converter controles MPC para comandos do hardware
            int mpc_steering = static_cast<int>(control.steer * 45.0 / 0.35);
            int mpc_speed = static_cast<int>(control.throttle * 100.0 / 0.6);
            
            // Limitar comandos
            mpc_steering = std::max(-45, std::min(45, mpc_steering));
            mpc_speed = std::max(0, std::min(100, mpc_speed));
            
            // Aplicar através do EngineController
            // Nota: Você precisará expor métodos públicos no ControlsManager
            // para acessar o EngineController
            // controls_manager->getEngineController()->set_steering(mpc_steering);
            // controls_manager->getEngineController()->set_speed(mpc_speed);
            
            // Por enquanto, apenas log
            step_counter++;
            if (step_counter % 20 == 0) {
                std::cout << "MPC Step " << step_counter 
                          << " | Throttle: " << std::fixed << std::setprecision(3) << control.throttle 
                          << " | Steering: " << control.steer 
                          << " | Target Speed: " << mpc_speed
                          << " | Target Steering: " << mpc_steering << std::endl;
            }
            
            // Simular atualização do estado (em um sistema real, 
            // isso viria de sensores/odometria)
            updateVehicleState(control.throttle, control.steer);
                      
        } catch (const std::exception& e) {
            std::cerr << "Erro no MPC: " << e.what() << std::endl;
        }
    }
    
    void handleKeyPress() {
        static bool recording = false;
        
        std::string input;
        std::getline(std::cin, input);
        
        if (input.empty() || input == "m") {
            // Alternar modo
            toggleMPCMode();
        } else if (input == "r") {
            // Alternar gravação
            recording = !recording;
            std::cout << (recording ? "Iniciando gravação de trajetória" 
                                   : "Parando gravação de trajetória") << std::endl;
            if (recording) {
                recorded_waypoints.clear();
                // Aqui você iniciaria a gravação baseada no movimento real do joystick
                startRecording();
            }
        } else if (input == "c") {
            // Limpar waypoints
            recorded_waypoints.clear();
            std::cout << "Trajetória limpa (" << recorded_waypoints.size() << " waypoints)" << std::endl;
        } else if (input == "s") {
            // Mostrar status
            showStatus();
        } else if (input == "q") {
            // Sair
            QApplication::quit();
        } else {
            std::cout << "Comando não reconhecido. Use: m(modo), r(gravar), c(limpar), s(status), q(sair)" << std::endl;
        }
    }

private:
    void setupKeyboardInput() {
        // Setup non-blocking keyboard input (simplified)
        std::cout << "Digite comandos (m: modo, r: gravar, c: limpar, s: status, q: sair):" << std::endl;
        
        // Timer para verificar input
        QTimer* input_timer = new QTimer(this);
        connect(input_timer, &QTimer::timeout, [this]() {
            if (std::cin.rdbuf()->in_avail()) {
                handleKeyPress();
            }
        });
        input_timer->start(100); // Check every 100ms
    }
    
    void toggleMPCMode() {
        mpc_active = !mpc_active;
        
        if (mpc_active) {
            if (recorded_waypoints.size() < 3) {
                std::cout << "Erro: Precisa de pelo menos 3 waypoints gravados!" << std::endl;
                mpc_active = false;
                return;
            }
            
            // Mudar para modo autônomo
            controls_manager->setMode(DrivingMode::Automatic);
            mpc_timer->start(50); // 20 Hz
            std::cout << "Modo MPC ATIVADO - Seguindo trajetória com " 
                      << recorded_waypoints.size() << " waypoints" << std::endl;
        } else {
            // Mudar para modo manual
            controls_manager->setMode(DrivingMode::Manual);
            mpc_timer->stop();
            std::cout << "Modo MANUAL ATIVADO - Use joystick" << std::endl;
        }
    }
    
    void startRecording() {
        // Em um sistema real, você conectaria aos sinais do joystick
        // para gravar as posições conforme o usuário move o carro
        
        // Simulação: gerar alguns waypoints de exemplo
        std::cout << "Simulando gravação... (em sistema real, use o joystick)" << std::endl;
        
        // Gerar trajetória curva simples
        for (int i = 0; i < 20; ++i) {
            double x = i * 0.5;
            double y = 1.5 * std::sin(x * 0.3);
            recorded_waypoints.emplace_back(x, y);
        }
        
        std::cout << "Trajetória simulada gravada com " << recorded_waypoints.size() << " pontos" << std::endl;
    }
    
    void updateVehicleState(double throttle, double steer) {
        // Modelo cinemático simples para simular movimento
        double dt = 0.05; // 20 Hz
        double wheelbase = 0.15; // 15cm
        
        current_state.velocity += throttle * dt;
        current_state.velocity = std::max(0.1, std::min(current_state.velocity, 1.5));
        
        current_state.x += current_state.velocity * std::cos(current_state.yaw) * dt;
        current_state.y += current_state.velocity * std::sin(current_state.yaw) * dt;
        current_state.yaw += (current_state.velocity / wheelbase) * std::tan(steer) * dt;
        
        // Normalizar ângulo
        while (current_state.yaw > M_PI) current_state.yaw -= 2.0 * M_PI;
        while (current_state.yaw < -M_PI) current_state.yaw += 2.0 * M_PI;
    }
    
    void showStatus() {
        std::cout << "\n=== STATUS ===" << std::endl;
        std::cout << "Modo: " << (mpc_active ? "MPC (Autônomo)" : "Manual") << std::endl;
        std::cout << "Waypoints gravados: " << recorded_waypoints.size() << std::endl;
        std::cout << "Posição atual: (" << std::fixed << std::setprecision(2) 
                  << current_state.x << ", " << current_state.y << ")" << std::endl;
        std::cout << "Velocidade: " << current_state.velocity << " m/s" << std::endl;
        std::cout << "Orientação: " << current_state.yaw * 180 / M_PI << " graus" << std::endl;
        std::cout << "MPC Steps: " << step_counter << std::endl;
        std::cout << "===============\n" << std::endl;
    }
};

// Include moc file for Qt
#include "main.moc"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    
    std::cout << "Iniciando sistema integrado MPC + Car Controls..." << std::endl;
    
    try {
        MPCIntegratedApp integrated_app(argc, argv);
        
        std::cout << "Sistema pronto! Use os comandos no terminal." << std::endl;
        
        return app.exec();
        
    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << std::endl;
        return 1;
    }
}
