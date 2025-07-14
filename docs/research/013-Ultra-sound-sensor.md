## SRF08 Ultrasonic Sensor
<img src="https://static.rapidonline.com/catalogueimages/product/78/10/s78-1086p01wj.jpg" alt="SRF08 Ultrasonic Sensor" style="width:40%;">

#### Hardware Overview
- High-accuracy ultrasonic rangefinder module with detection range from ~3 cm to ~6 m
- Communicates over the I²C protocol with configurable 7-bit addresses (default: `0x70`)
- Measures not only the nearest object, but can return up to 16 echo distances per ranging cycle
- Integrated ambient light sensor (photocell) provides additional environmental data
- Powered via 5 V input and draws approximately 15 mA during active ranging
- On-board PIC microcontroller handles pulse timing and echo processing autonomously
- Available gain and range register configuration to fine-tune sensitivity and detection range

#### How it works
- The SRF08 emits a 40 kHz ultrasonic pulse and listens for echoes to return
- Time-of-flight is calculated internally and output as distance in microseconds, inches, or centimeters
- Host controller (e.g., Jetson, Raspberry Pi, Arduino) sends an I²C command to initiate ranging
- Light level and up to 16 echo ranges can then be read from dedicated I²C registers
- Advanced mode: returns a 32-byte sonar “profile” buffer showing reflected signal strengths by bin

#### What kind of signals it uses
- I²C communication (SCL and SDA) for all commands and data reads
- Operates at 5 V logic levels (use logic level shifter when interfacing with 3.3 V devices like Jetson)
- No analog or PWM signals required; fully digital operation over I²C
- Optional broadcast ping available to trigger multiple SRF08s simultaneously

#### What you need to interact with it
- A microcontroller or embedded system with I²C support (e.g., NVIDIA Jetson, Raspberry Pi, Arduino)
- Pull-up resistors (typically 4.7 kΩ) between SCL/SDA and 5 V if not already onboard
- Optional logic level shifter (for 3.3 V devices)
- 5 V power supply
- Python or C code to control I²C communication and interpret distance data
- Optional: mount bracket or housing if integrating in a robot or fixed-position application

#### Datasheet PDF

https://www.robot-electronics.co.uk/htm/srf08tech.html

#### Documentation
https://zeldor.biz/2012/10/arduino-srf08/
