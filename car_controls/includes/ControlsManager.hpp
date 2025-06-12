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

#include "EngineController.hpp"
#include "JoysticksController.hpp"
#include "inference/CameraStreamer.hpp"
#include "../../ZeroMQ/Subscriber.hpp"
#include "../../ZeroMQ/Publisher.hpp"
#include "MPCPlanner.hpp"
#include "CommonTypes.hpp"
#include <QObject>
#include <QThread>
#include <QProcess>

/*!
 * @brief The ControlsManager class.
 * @details This class is responsible for managing the controls of the car.
 */
class ControlsManager : public QObject {
	Q_OBJECT

private:
    MPCPlanner			*m_mpcPlanner;
	EngineController	m_engineController;
	JoysticksController	*m_manualController;
	DrivingMode 		m_currentMode;

	Subscriber 			*m_subscriberJoystickObject;
	//Subscriber *m_subscriberInferenceThreadObject;
	//Subscriber *m_subscriberODThreadObject;

	CameraStreamer 		*m_cameraStreamerObject;
	//YOLOv5TRT *m_yoloObject;

	//std::shared_ptr<IInferencer> inferencer;

	std::atomic<bool> 	m_running;
    std::atomic<bool> 	m_autonomousMode;

	QThread *m_autonomousControlThread;
	QThread *m_manualControllerThread;
	QThread *m_joystickControlThread;

	QThread *m_subscriberJoystickThread;
	QThread *m_cameraStreamerThread;

public:
	explicit ControlsManager(int argc, char **argv, QObject *parent = nullptr);
	~ControlsManager();

	void setMode(DrivingMode mode);
	void readJoystickEnable();
	bool isProcessRunning(const QString &processName);
	void startAutonomousControl();
    void stopAutonomousControl();
    void autonomousControlLoop();

	VehicleState getCurrentVehicleState();
	std::vector<Point2D> getWaypointsFromVision();
	bool checkEmergencyObstacles();
};

#endif // CONTROLSMANAGER_HPP
