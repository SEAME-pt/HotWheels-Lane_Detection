/*!
 * @file JoysticksController.hpp
 * @brief File containing the JoysticksController class.
 * @version 0.1
 * @date 2025-02-12
 * @details This class is responsible for controlling the joysticks of the car.
 * @note This class is a subclass of QObject.
 * 
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef JOYSTICKS_CONTROLLER_HPP
#define JOYSTICKS_CONTROLLER_HPP

#include <QObject>
#include <SDL2/SDL.h>
#include <functional>

/*!
 * @brief The JoysticksController class
 * @details This class is responsible for controlling the joysticks of the car.
 */
class JoysticksController : public QObject
{
	Q_OBJECT

private:
	SDL_Joystick *m_joystick;
	std::function<void(int)> m_updateSteering;
	std::function<void(int)> m_updateSpeed;
	bool m_running;

public:
	JoysticksController(std::function<void(int)> steeringCallback,
											std::function<void(int)> speedCallback,
											QObject *parent = nullptr);
	~JoysticksController();
	bool init();
	void requestStop();

public slots:
	void processInput();

signals:
	void finished();
};

#endif // JOYSTICKS_CONTROLLER_HPP
