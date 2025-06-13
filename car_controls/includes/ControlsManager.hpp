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

#include "../ZeroMQ/Publisher.hpp"
#include "../ZeroMQ/Subscriber.hpp"
#include "CommonTypes.hpp"
#include "EngineController.hpp"
#include "JoysticksController.hpp"
#include "MPCPlanner.hpp"
#include "Polyfitter.hpp"
#include "inference/CameraStreamer.hpp"
#include <QObject>
#include <QProcess>
#include <QThread>
#include <vector>
#include <string>
#include <sstream>
#include <atomic>

/*!
 * @brief The ControlsManager class.
 * @details This class is responsible for managing the controls of the car.
 */
class ControlsManager : public QObject {
	Q_OBJECT

private:
	EngineController	m_engineController;
	JoysticksController	*m_manualController;
	DrivingMode 		m_currentMode;

	Subscriber 			*m_subscriberJoystickObject;
	
	QThread *m_manualControllerThread;
	QThread *m_joystickControlThread;
	QThread *m_subscriberJoystickThread;
	QThread *m_cameraStreamerThread;
	
	std::atomic<bool> 	m_running;
	//Subscriber *m_subscriberInferenceThreadObject;
	//Subscriber *m_subscriberODThreadObject;

	//YOLOv5TRT *m_yoloObject;
	
	//std::shared_ptr<IInferencer> inferencer;
	MPCPlanner			*m_mpcPlanner;
	Polyfitter* 		m_polyfitter;
    std::atomic<bool> 	m_autonomousMode;
	QThread *m_autonomousControlThread;
	
	CameraStreamer 		*m_cameraStreamerObject;


public:
	explicit ControlsManager(int argc, char **argv, QObject *parent = nullptr);
	~ControlsManager();

	void setMode(DrivingMode mode);
	void readJoystickEnable();
	bool isProcessRunning(const QString &processName);
	void startAutonomousControl();
    void stopAutonomousControl();
    void autonomousControlLoop();
    // Exibe a imagem da câmera com as lanes e centerline desenhadas
    void showVisionDebug();

private:
	VehicleState getCurrentVehicleState();
    std::vector<Point2D> getWaypointsFromVision();
    LaneInfo getLaneInfoFromVision();
    bool checkEmergencyObstacles();
    std::string serializeMask(const cv::Mat& mask);
    cv::Mat deserializeMask(const std::string& data);
};

#endif // CONTROLSMANAGER_HPP
