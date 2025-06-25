## Micro::bit V2

![Micro:bit V2](https://kitronik.co.uk/cdn/shop/products/56100_large-micro-bit-v2-board-only_800x.jpg?v=1607514556)

#### Hardware Overview
- Features a 5×5 red LED matrix display for visual output (can show text, icons, animations)
- Includes two onboard programmable buttons (A and B) for user interaction
- Equipped with a built-in microphone and speaker for sound input/output
- Contains motion sensors: accelerometer and magnetometer (compass)
- Edge connector with 25 exposed GPIO pads, including 3 main rings for easy clip-on connections
- Built-in 2.4 GHz radio and Bluetooth Low Energy (BLE) for wireless communication
- Touch-sensitive logo and power/reset button
- Powered via micro-USB or 2×AAA battery pack (with JST connector)

#### How it works

The micro:bit V2 is a compact, all-in-one microcontroller board designed for easy physical computing, coding education, and prototyping.
It works by running user-written programs (in Python, MakeCode, or C++) that interact with its onboard sensors, actuators, and I/O pins.
The board executes instructions stored in its flash memory, allowing it to:
- Control external components (LEDs, servos, sensors),
- Respond to physical inputs (buttons, accelerometer, microphone),
- Communicate wirelessly using Bluetooth or its built-in radio.
Programs are uploaded via USB or Bluetooth, and it starts running them immediately after power-up.

#### Kind of signals used

| Signal Type          | Purpose                                                |
| -------------------- | ------------------------------------------------------ |
| **Digital I/O**      | To turn devices ON/OFF (e.g. LEDs, buzzers)            |
| **PWM**              | To control motors and servos (smooth movement)         |
| **Analog Input**     | To read variable sensors (light, temperature)          |
| **I2C / SPI / UART** | To communicate with other boards or sensors            |
| **Radio (2.4 GHz)**  | To wirelessly send short messages to other micro\:bits |
| **Bluetooth LE**     | To connect to phones, tablets, or PCs wirelessly       |

#### What you need to interact with it

| Component / Tool                                          | Purpose                                                 |
| --------------------------------------------------------- | ------------------------------------------------------- |
| **USB cable (micro-USB)**                                 | To power and program the micro\:bit                     |
| **Computer or mobile device**                             | To write and upload code (via MakeCode, Python, or app) |
| **Battery pack (optional)**                               | To run the micro\:bit without USB (e.g. in the field)   |
| **External components** (e.g. STOP\:bit, sensors, motors) | To expand its functionality                             |
| **Alligator clips / breakout board**                      | To connect to the GPIO edge pins                        |
| **MakeCode or MicroPython editor**                        | To write, test, and deploy code                         |

----------------------------------------------------------

## Kitronik STOP:bit

![Kitronik STOP:bit](https://kitronik.co.uk/cdn/shop/products/5642_large-stop-bit-bbc-microbit-pedestrian-crossing-traffic-light_f8cf5103-c02d-42f3-86df-059e4afe8764_800x.jpg?v=1582131763)

#### Hardware Overview
- Features three large 10 mm LEDs in red, yellow, and green—perfectly sized and spaced like a pedestrian crossing light
- It plugs into the micro:bit via edge pads or can be clipped/screwed on for stability .
- LEDs are wired to three GPIO pins; power (3V and 0V) comes directly from the micro:bit

#### How it works
- The board connects directly to the micro:bit using screws or crocodile clips, aligning the micro:bit’s P0, P1, P2 pins to the large 10 mm LEDS representing red/yellow/green lights
- It's pre-assembled, with resistors included, so there's no soldering required
- Mountable with laser-cut stands or integrated in larger setups (e.g., paired STOP:bits coordinating via radio)

#### Kind of signals used
- Digital I/O outputs from micro:bit drive each LED. Writing 1 to P0 turns on the red LED, 0 turns it off; same for P1 (yellow) and P2 (green)
- Optionally, radio communication between multiple micro:bits allows synchronizing multiple STOP:bit units, enabling traffic coordination

#### What you need to interact with it
- A BBC micro:bit (v1 or v2) — it provides both control signals and power to the LEDs
- A micro USB cable to program the micro:bit.
- The MakeCode extension (kitronik-stopbit) or MicroPython support to control the lights
- Screws or crocodile clips to physically secure the micro:bit to the board .
- Optionally, two micro:bits and STOP:bits for radio-linked traffic light setups

----------------------------------------------------------

## Kitronik ACCESS:bit

![Kitronik ACCESS:bit](https://kitronik.co.uk/cdn/shop/products/5646_large-access-bit-microbit-transportation-pedestrian-crossing-projects_53f05acd-fcfe-486e-b843-cd909d027dbf_800x.jpg?v=1582131993)

#### Hardware Overview
- Features a servo-controlled barrier arm that swings up and down like a real access gate
- Includes a built-in buzzer for sound alerts
- Connects to the BBC micro:bit via edge pads (can be clipped or screwed on securely)
- The servo is controlled using a PWM signal from one GPIO pin; the buzzer uses a standard digital output
- Powered by 3× AAA batteries via an onboard battery holder with on/off switch
- All components are pre-assembled—no soldering required

#### How it works
- The ACCESS:bit clips or screws onto your micro:bit, providing stable physical mounting
- A servo motor physically moves the barrier arm through ~180° based on commands from the micro:bit
- It also includes:
	- A buzzer for sound alerts,
	- A simple on/off switch and battery holder (3×AAA),
	- Electrical connections (edge pins or crocodile clips) to the micro:bit

#### What kind of signals it uses
- Digital PWM signals from the micro:bit to control the servo angle (using PWM pins).
- Digital I/O signals to trigger the buzzer.
- Power rails: 3V (or external battery) and ground, supplied via clip-on or screws

#### What you need to interact with it
- A BBC micro:bit V1 or V2 to give commands and power
- 3 AAA batteries for the servo and buzzer (unless powered via micro:bit)
- A micro USB cable to program the micro:bit.
- Kitronik MakeCode extension for ACCESS:bit, giving you child-friendly drag‑and‑drop blocks to control it
- Optionally: screws or crocodile clips to attach the micro:bit.

#### Mounting PDF

https://resources.kitronik.co.uk/pdf/5646-access-bit-microbit-pedestrian-crossing-datsheet.pdf

----------------------------------------------------------

## Set up Jetson Nano to interact with both controllers

#### System Architecture

Jetson Nano
- Acts as the central controller.
- Sends text-based commands to the micro:bit via USB serial communication.

BBC micro:bit
- Acts as a bridge between Jetson and the STOP:bit / ACCESS:bit hardware.
- Receives serial commands from Jetson.
- Converts those commands into GPIO or PWM signals to control STOP:bit and ACCESS:bit.

STOP:bit
- A traffic light board with 3 large LEDs (red, yellow, green).
- Controlled via digital output pins from the micro:bit.

ACCESS:bit
- A servo-operated barrier with an optional buzzer.
- The servo uses PWM signals to move the barrier.
- The buzzer uses a digital on/off signal.

#### Communication Flow
1. Jetson sends a string command (like light-red or open-barrier) over USB.
2. The micro:bit receives this via its UART (serial) interface.
3. The micro:bit decodes the command and triggers the corresponding hardware:
4. Sets digital pins to turn STOP:bit LEDs on/off.
5. Sends PWM to the servo motor to move the ACCESS:bit barrier.
6. Activates the buzzer using a digital signal.

The micro:bit acts as a multi-protocol hardware adapter: it receives simple text commands from the Jetson and uses its pins to control physical components.

This avoids the need for the Jetson to directly interface with complex IO like PWM, making the system both modular and safe.
