## Kitronik STOP:bit

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
