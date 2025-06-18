/*!
 * @file EngineController.hpp
 * @brief File containing the EngineController class.
 * @version 0.1
 * @date 2025-02-12
 * @details This class is responsible for controlling the engine of the car.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef ENGINECONTROLLER_HPP
#define ENGINECONTROLLER_HPP

#include "IPeripheralController.hpp"
#include "enums.hpp"
#include <QObject>
#include <atomic>

/*!
 * @brief The EngineController class
 * @details This class is responsible for controlling the engine of the car.
 */
class EngineController : public QObject {
  Q_OBJECT

private:
  const int MAX_ANGLE = 180;
  const int SERVO_CENTER_PWM = 340;
  const int SERVO_LEFT_PWM = 340 - 100;
  const int SERVO_RIGHT_PWM = 340 + 130;
  const int STEERING_CHANNEL = 0;

  std::atomic<bool> m_running;
  std::atomic<int> m_current_speed;
  std::atomic<int> m_current_angle;
  CarDirection m_currentDirection = CarDirection::Stop;

  void setDirection(CarDirection newDirection);

  IPeripheralController *pcontrol;

public:
  EngineController();
  EngineController(int servo_addr, int motor_addr, QObject *parent = nullptr);
  ~EngineController();

  void start();
  void stop();
  void set_speed(int speed);
  void set_steering(int angle);

signals:
  void directionUpdated(CarDirection newDirection);
  void steeringUpdated(int newAngle);
};

#endif // ENGINECONTROLLER_HPP
