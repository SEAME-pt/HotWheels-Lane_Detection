/*!
 * @file MockPeripheralController.hpp
 * @brief File containing Mock classes to test the peripheral controller.
 * @version 0.1
 * @date 2025-01-30
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Ricardo Melo (@reomelo)
 * @author Tiago Pereira (@t-pereira06)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef MOCKPERIPHERALCONTROLLER_HPP
#define MOCKPERIPHERALCONTROLLER_HPP

#include "IPeripheralController.hpp"
#include <gmock/gmock.h>

/*!
 * @class MockPeripheralController
 * @brief Class to emulate the behavior of the peripheral controller.
 *
 */
class MockPeripheralController : public IPeripheralController {
public:
  /*! @brief Mocked method to write a byte of data to the I2C bus. */
  MOCK_METHOD(int, i2c_smbus_write_byte_data,
              (int file, uint8_t command, uint8_t value), (override));
  /*! @brief Mocked method to read a byte of data from the I2C bus. */
  MOCK_METHOD(int, i2c_smbus_read_byte_data, (int file, uint8_t command),
              (override));

  /*! @brief Mocked method to write a byte of data to a specific register. */
  MOCK_METHOD(void, write_byte_data, (int fd, int reg, int value), (override));
  /*! @brief Mocked method to read a byte of data from a specific register. */
  MOCK_METHOD(int, read_byte_data, (int fd, int reg), (override));

  /*! @brief Mocked method to set the PWM of a servo motor. */
  MOCK_METHOD(void, set_servo_pwm, (int channel, int on_value, int off_value),
              (override));
  /*! @brief Mocked method to set the PWM of a motor. */
  MOCK_METHOD(void, set_motor_pwm, (int channel, int value), (override));

  /*! @brief Mocked method to initialize the servo motors. */
  MOCK_METHOD(void, init_servo, (), (override));
  /*! @brief Mocked method to initialize the motors. */
  MOCK_METHOD(void, init_motors, (), (override));
};

#endif // MOCKPERIPHERALCONTROLLER_HPP
