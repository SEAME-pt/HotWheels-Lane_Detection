/*!
 * @file ControlsManager.hpp
 * @brief File containing the ControlsManager class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the declaration of the ControlsManager class, which
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef CONTROLSMANAGER_HPP
#define CONTROLSMANAGER_HPP

#include "../../ZeroMQ/Publisher.hpp"
#include "../../ZeroMQ/Subscriber.hpp"
#include "AutonomousMode.hpp"
#include "EngineController.hpp"
#include "JoysticksController.hpp"
#include "inference/CameraStreamer.hpp"
#include <QObject>
#include <QProcess>
#include <QThread>

/*!
 * @brief The ControlsManager class.
 * @details This class is responsible for managing the controls of the car.
 */
class ControlsManager : public QObject {
		Q_OBJECT

	private:
		EngineController m_engineController;
		JoysticksController *m_manualController;
		DrivingMode m_currentMode;

		Subscriber *m_subscriberJoystickObject;
		CameraStreamer *m_cameraStreamerObject;
		AutonomousMode *m_autonomousModeObject;

		std::atomic<bool> m_running;

		QThread *m_manualControllerThread;
		QThread *m_joystickControlThread;
		QThread *m_subscriberJoystickThread;
		QThread *m_cameraStreamerThread;

	public:
		explicit ControlsManager (int argc, char **argv, QObject *parent = nullptr);
		~ControlsManager ();

		void setMode (DrivingMode mode);
		void readJoystickEnable ();
		bool isProcessRunning (const QString &processName);
		DrivingMode getCurrentMode () const {
			return m_currentMode;
		}

		// Direct access to AutonomousMode for advanced control
		AutonomousMode *getAutonomousMode () {
			return m_autonomousModeObject;
		}

	private:
		void initializeAutonomousMode ();
		void setupDirectMPCCallback ();
};

#endif // CONTROLSMANAGER_HPP
