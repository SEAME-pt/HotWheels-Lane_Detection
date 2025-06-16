/*!
 * @file test_PeripheralController.cpp
 * @brief Unit tests for the PeripheralController class.
 * @version 0.1
 * @date 2025-01-30
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Ricardo Melo (@reomelo)
 * @author Tiago Pereira (@t-pereira06)
 * @author Michel Batista (@MicchelFAB)
 *
 * @details This file contains unit tests for the PeripheralController class,
 * using Google Test and Google Mock frameworks.
 */

#include "../mocks/MockPeripheralController.hpp"
#include "tests/mocks/MockPeripheralController.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::Return;
using ::testing::Throw;

/*!
 * @test Tests if the servo PWM is set correctly.
 * @brief Ensures that set_servo_pwm() is called with the correct parameters.
 *
 * @details This test verifies that set_servo_pwm() is called with the correct
 * parameters.
 *
 * @see PeripheralController::set_servo_pwm
 */
TEST(PeripheralControllerTest, TestServoPWM) {
  MockPeripheralController mockController;

  EXPECT_CALL(mockController, set_servo_pwm(0, 1024, 2048)).Times(1);
  EXPECT_CALL(mockController, set_servo_pwm(1, 0, 4096)).Times(1);

  mockController.set_servo_pwm(0, 1024, 2048);
  mockController.set_servo_pwm(1, 0, 4096);
}

/*!
 * @test Tests if the motor PWM is set correctly.
 * @brief Ensures that set_motor_pwm() is called with the correct parameters.
 *
 * @details This test verifies that set_motor_pwm() is called with the correct
 * parameters.
 *
 * @see PeripheralController::set_motor_pwm
 */
TEST(PeripheralControllerTest, TestMotorPWM) {
  MockPeripheralController mockController;

  EXPECT_CALL(mockController, set_motor_pwm(0, 1500)).Times(1);
  EXPECT_CALL(mockController, set_motor_pwm(1, 3000)).Times(1);

  mockController.set_motor_pwm(0, 1500);
  mockController.set_motor_pwm(1, 3000);
}

/*!
 * @test Tests if the servo is initialized correctly.
 * @brief Ensures that init_servo() is called.
 *
 * @details This test verifies that init_servo() is called.
 *
 * @see PeripheralController::init_servo
 */
TEST(PeripheralControllerTest, TestInitServo) {
  MockPeripheralController mockController;

  EXPECT_CALL(mockController, init_servo()).Times(1);

  mockController.init_servo();
}

/*!
 * @test Tests if the motors are initialized correctly.
 * @brief Ensures that init_motors() is called.
 *
 * @details This test verifies that init_motors() is called.
 *
 * @see PeripheralController::init_motors
 */
TEST(PeripheralControllerTest, TestInitMotors) {
  MockPeripheralController mockController;

  EXPECT_CALL(mockController, init_motors()).Times(1);

  mockController.init_motors();
}

/*!
 * @test Tests if I2C write byte data is called correctly.
 * @brief Ensures that i2c_smbus_write_byte_data() is called with the correct
 * parameters.
 *
 * @details This test verifies that i2c_smbus_write_byte_data() is called with
 * the correct parameters and returns the expected result.
 *
 * @see PeripheralController::i2c_smbus_write_byte_data
 */
TEST(PeripheralControllerTest, TestI2CWriteByteData) {
  MockPeripheralController mockController;

  EXPECT_CALL(mockController, i2c_smbus_write_byte_data(1, 0x10, 0x20))
      .WillOnce(Return(0));

  int result = mockController.i2c_smbus_write_byte_data(1, 0x10, 0x20);

  EXPECT_EQ(result, 0);
}

/*!
 * @test Tests if I2C read byte data is called correctly.
 * @brief Ensures that i2c_smbus_read_byte_data() is called with the correct
 * parameters.
 *
 * @details This test verifies that i2c_smbus_read_byte_data() is called with
 * the correct parameters and returns the expected result.
 *
 * @see PeripheralController::i2c_smbus_read_byte_data
 */
TEST(PeripheralControllerTest, TestI2CReadByteData) {
  MockPeripheralController mockController;

  EXPECT_CALL(mockController, i2c_smbus_read_byte_data(1, 0x10))
      .WillOnce(Return(0x30));

  int result = mockController.i2c_smbus_read_byte_data(1, 0x10);

  EXPECT_EQ(result, 0x30);
}

/*!
 * @test Tests if write_byte_data() throws an exception on failure.
 * @brief Ensures that write_byte_data() throws a runtime_error exception.
 *
 * @details This test verifies that write_byte_data() throws a runtime_error
 * exception when the I2C write fails.
 *
 * @see PeripheralController::write_byte_data
 */
TEST(PeripheralControllerTest, TestWriteByteDataException) {
  MockPeripheralController mockController;

  EXPECT_CALL(mockController, write_byte_data(_, _, _))
      .WillOnce(Throw(std::runtime_error("I2C write failed")));

  EXPECT_THROW(mockController.write_byte_data(1, 0x10, 0x20),
               std::runtime_error);
}

/*!
 * @test Tests if read_byte_data() throws an exception on failure.
 * @brief Ensures that read_byte_data() throws a runtime_error exception.
 *
 * @details This test verifies that read_byte_data() throws a runtime_error
 * exception when the I2C read fails.
 *
 * @see PeripheralController::read_byte_data
 */
TEST(PeripheralControllerTest, TestReadByteDataException) {
  MockPeripheralController mockController;

  EXPECT_CALL(mockController, read_byte_data(_, _))
      .WillOnce(Throw(std::runtime_error("I2C read failed")));

  EXPECT_THROW(mockController.read_byte_data(1, 0x10), std::runtime_error);
}
