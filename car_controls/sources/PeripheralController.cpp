/*!
 * @file PeripheralController.cpp
 * @brief Implementation of the PeripheralController class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the implementation of the PeripheralController class,
 * which is responsible for controlling the peripherals of the car.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "PeripheralController.hpp"

/*!
 * @union i2c_smbus_data
 * @brief Represents data formats for I2C SMBus communication.
 * - `byte`: Single byte of data.
 * - `word`: 16-bit (2 byte) data.
 * - `block`: Array for up to 34-byte block transfers.
 */
union i2c_smbus_data {
	uint8_t byte;
	uint16_t word;
	uint8_t block[34]; // Block size for SMBus
};

/* ------------------------------------ */

#define I2C_SMBUS_WRITE 0
#define I2C_SMBUS_READ 1
#define I2C_SMBUS_BYTE_DATA 2

/*!
 * @brief Clamps a value to a given range.
 *
 * @param value Value to be clamped.
 * @param min_val Minimum value of the range.
 * @param max_val Maximum value of the range.
 * @return The clamped value, or the original value if it is within the range.
 */
template <typename T> T clamp(T value, T min_val, T max_val) {
	return (value < min_val) ? min_val : ((value > max_val) ? max_val : value);
}

/*!
 * @brief Constructor for the PeripheralController class.
 *
 * @param servo_addr The address of the servo controller.
 * @param motor_addr The address of the motor controller.
 *
 * @details Initializes the I2C buses and sets the device addresses.
 * Throws an exception if either the servo or motor controller cannot be opened
 * or if the address cannot be set.
 */
PeripheralController::PeripheralController(int servo_addr, int motor_addr)
		: servo_addr_(servo_addr), motor_addr_(motor_addr) {
	// Initialize I2C buses
	servo_bus_fd_ = open("/dev/i2c-1", O_RDWR);
	motor_bus_fd_ = open("/dev/i2c-1", O_RDWR);

	if (servo_bus_fd_ < 0 || motor_bus_fd_ < 0) {
		throw std::runtime_error("Failed to open I2C device");
		return;
	}

	// Set device addresses
	if (ioctl(servo_bus_fd_, I2C_SLAVE, servo_addr_) < 0 ||
			ioctl(motor_bus_fd_, I2C_SLAVE, motor_addr_) < 0) {
		throw std::runtime_error("Failed to set I2C address");
		return;
	}
}

/*!
 * @brief Destructor for the PeripheralController class.
 *
 * @details Closes the file descriptors for the servo and motor I2C buses,
 * ensuring that resources are properly released when the object is
 * destroyed.
 */
PeripheralController::~PeripheralController() {
	close(servo_bus_fd_);
	close(motor_bus_fd_);
}

/*!
 * @brief Writes a byte of data to a specific register.
 *
 * @param file The file descriptor for the I2C bus.
 * @param command The register address to write to.
 * @param value The byte of data to write to the register.
 * @return The result of the write operation (0 on success, -1 on failure).
 * @throws std::runtime_error if the I2C write operation fails.
 */
int PeripheralController::i2c_smbus_write_byte_data(int file, uint8_t command,
																										uint8_t value) {
	union i2c_smbus_data data;
	data.byte = value;

	struct i2c_smbus_ioctl_data args;
	args.read_write = I2C_SMBUS_WRITE;
	args.command = command;
	args.size = I2C_SMBUS_BYTE_DATA;
	args.data = &data;

	return ioctl(file, I2C_SMBUS, &args);
}

/*!
 * @brief Reads a byte of data from a specific register.
 *
 * @param file The file descriptor of the I2C bus.
 * @param command The register address to read from.
 * @return The byte of data read from the register, or -1 if the operation fails.
 */
int PeripheralController::i2c_smbus_read_byte_data(int file, uint8_t command) {
	union i2c_smbus_data data;

	struct i2c_smbus_ioctl_data args;
	args.read_write = I2C_SMBUS_READ;
	args.command = command;
	args.size = I2C_SMBUS_BYTE_DATA;
	args.data = &data;

	if (ioctl(file, I2C_SMBUS, &args) < 0) {
		return -1;
	}
	return data.byte;
}

/*!
 * @brief Writes a byte of data to a specific register.
 *
 * @param fd The file descriptor for the I2C bus.
 * @param reg The register address to write to.
 * @param value The byte of data to write to the register.
 * @throws std::runtime_error if the I2C write operation fails.
 */
void PeripheralController::write_byte_data(int fd, int reg, int value) {
	if (i2c_smbus_write_byte_data(fd, reg, value) < 0) {
		throw std::runtime_error("I2C write failed");
	}
}

/*!
 * @brief Reads a byte of data from a specific register.
 *
 * @param fd The file descriptor for the I2C bus.
 * @param reg The register address to read from.
 * @return The byte of data read from the register.
 * @throws std::runtime_error if the I2C read operation fails.
 */

int PeripheralController::read_byte_data(int fd, int reg) {
	int result = i2c_smbus_read_byte_data(fd, reg);
	if (result < 0) {
		throw std::runtime_error("I2C read failed");
	}
	return result;
}

/*!
 * @brief Sets the PWM of a servo motor.
 *
 * @param channel The channel number of the servo motor to control.
 * @param on_value The on-time value of the PWM signal (in range 0-4095).
 * @param off_value The off-time value of the PWM signal (in range 0-4095).
 *
 * @details The on-time value is stored in the first two bytes of the register,
 * and the off-time value is stored in the second two bytes. The actual PWM
 * frequency is 50 Hz.
 */
void PeripheralController::set_servo_pwm(int channel, int on_value,
																				 int off_value) {
	int base_reg = 0x06 + (channel * 4);
	write_byte_data(servo_bus_fd_, base_reg, on_value & 0xFF);
	write_byte_data(servo_bus_fd_, base_reg + 1, on_value >> 8);
	write_byte_data(servo_bus_fd_, base_reg + 2, off_value & 0xFF);
	write_byte_data(servo_bus_fd_, base_reg + 3, off_value >> 8);
}

/*!
 * @brief Sets the PWM value for a motor.
 *
 * @param channel The motor channel to set.
 * @param value The desired PWM value.
 *
 * @details The value is clamped to [0, 4095] and then written to the motor
 * controller.
 */
void PeripheralController::set_motor_pwm(int channel, int value) {
	value = clamp(value, 0, 4095);
	write_byte_data(motor_bus_fd_, 0x06 + (4 * channel), 0);
	write_byte_data(motor_bus_fd_, 0x07 + (4 * channel), 0);
	write_byte_data(motor_bus_fd_, 0x08 + (4 * channel), value & 0xFF);
	write_byte_data(motor_bus_fd_, 0x09 + (4 * channel), value >> 8);
}

/*!
 * @brief Initializes the servo controller.
 *
 * @details Configures the servo controller with specific register settings.
 * The method writes a sequence of commands to the servo bus to prepare the 
 * controller for operation. It includes setting the mode, setting the prescale 
 * value, and enabling the output. Each command is followed by a delay to 
 * ensure proper initialization.
 */

void PeripheralController::init_servo() {
	write_byte_data(servo_bus_fd_, 0x00, 0x06);
	usleep(100000);

	write_byte_data(servo_bus_fd_, 0x00, 0x10);
	usleep(100000);

	write_byte_data(servo_bus_fd_, 0xFE, 0x79);
	usleep(100000);

	write_byte_data(servo_bus_fd_, 0x01, 0x04);
	usleep(100000);

	write_byte_data(servo_bus_fd_, 0x00, 0x20);
	usleep(100000);
}

/*!
 * @brief Initializes the motor controllers.
 *
 * @details Sets up the motor controllers, configuring them for 100 Hz PWM
 * operation. The motors are then enabled.
 */
void PeripheralController::init_motors() {
	write_byte_data(motor_bus_fd_, 0x00, 0x20);

	int prescale = static_cast<int>(std::floor(25000000.0 / 4096.0 / 100 - 1));
	int oldmode = read_byte_data(motor_bus_fd_, 0x00);
	int newmode = (oldmode & 0x7F) | 0x10;

	write_byte_data(motor_bus_fd_, 0x00, newmode);
	write_byte_data(motor_bus_fd_, 0xFE, prescale);
	write_byte_data(motor_bus_fd_, 0x00, oldmode);
	usleep(5000);
	write_byte_data(motor_bus_fd_, 0x00, oldmode | 0xa1);
}
