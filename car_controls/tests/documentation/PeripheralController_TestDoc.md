# PeripheralController Unit Tests

This test suite verifies the correct behavior of the `PeripheralController` class using **Google Test** and **Google Mock**. The tests focus on method invocations and expected interactions with hardware interfaces like PWM and I2C.

---

## ✅ What’s Covered

| Area                          | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| PWM control                   | Validates `set_servo_pwm` and `set_motor_pwm` are called with correct params |
| Initialization                | Verifies that `init_servo` and `init_motors` are called                     |
| I2C communication             | Tests `i2c_smbus_write_byte_data` and `i2c_smbus_read_byte_data` behavior  |
| Error handling                | Ensures `write_byte_data` and `read_byte_data` throw exceptions on failure |

---

## Test Descriptions

### PWM Functionality

- `TestServoPWM`
  Confirms `set_servo_pwm` is called with expected channel, on, and off parameters.

- `TestMotorPWM`
  Confirms `set_motor_pwm` is called correctly for each motor.

---

### Peripheral Initialization

- `TestInitServo`
  Verifies that `init_servo()` is invoked once.

- `TestInitMotors`
  Ensures `init_motors()` is invoked once.

---

### I2C Communication

- `TestI2CWriteByteData`
  Checks if `i2c_smbus_write_byte_data()` is called with expected parameters and returns `0`.

- `TestI2CReadByteData`
  Confirms that `i2c_smbus_read_byte_data()` is called correctly and returns the expected byte.

---

### Exception Handling

- `TestWriteByteDataException`
  Simulates a failure and ensures that `write_byte_data()` throws a `std::runtime_error`.

- `TestReadByteDataException`
  Ensures that `read_byte_data()` throws an exception when the mock simulates failure.

---

## 🔧 How It Works

These tests use `MockPeripheralController`, a mock implementation of the `PeripheralController` interface. With **Google Mock**, the tests define expectations and simulate return values or exceptions.

This allows complete validation of:
- API contract (method calls and arguments)
- Error propagation
- Return value correctness

---

## Requirements

- Google Test and Google Mock
- A `MockPeripheralController.hpp` header that mocks all tested methods

