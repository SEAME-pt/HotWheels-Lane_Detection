/*!
 * @file main.cpp
 * @brief Main function for the car controls service.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the main function for the car controls service.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "ControlsManager.hpp"
#include <QCoreApplication>
#include <atomic>
#include <csignal>
#include <iostream>
#include <atomic>

volatile bool keepRunning = true;
std::atomic<bool> g_running{true}; //! REMOVE THIS LINE IF YOU DO NOT NEED IT

ControlsManager *g_controlsManager = nullptr;

/*!
 * @brief SIGINT signal handler.
 * @details This function will be called when the SIGINT signal is received.
 * The function will quit the QCoreApplication.
 */
void handleSigint(int) {
	qDebug() << "SIGINT received. Quitting application...";

	if(g_controlsManager) {
		delete g_controlsManager;
		g_controlsManager = nullptr;
	}

	QCoreApplication::quit();
}

/*!
 * @brief Entry point for the car controls service.
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments.
 * @return An integer indicating the exit status of the application.
 * @details Initializes the QCoreApplication and sets up signal handling for
 * SIGINT. Instantiates the ControlsManager and starts the event loop. If an
 * exception is thrown during execution, it is caught and logged, returning a
 * non-zero exit status. The application runs until quit is invoked.
 */

int main(int argc, char *argv[]) {
    QCoreApplication a(argc, argv);
    std::signal(SIGINT, handleSigint);
    std::signal(SIGTERM, handleSigint);

    try {
        g_controlsManager = new ControlsManager(argc, argv);
        
        // === CONFIGURAÇÃO HÍBRIDA ===
        g_controlsManager->setDirectFlowEnabled(true);    // Fluxo direto para MPC
        g_controlsManager->setZeroMQMaintained(true);     // ZeroMQ para apps externas
        
        INFO_LOG("Main", "Sistema híbrido configurado:");
        INFO_LOG("Main", "- Fluxo direto para MPC (baixa latência)");
        INFO_LOG("Main", "- ZeroMQ mantido para aplicações externas");
        
        return a.exec();
    } catch(const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
