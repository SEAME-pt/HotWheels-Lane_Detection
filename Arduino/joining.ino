#include <SPI.h>
#include "mcp2515_can.h"
#include <Wire.h>

#define SRF08_ADDR 0x70

// Pin definitions
const int SENSOR_PIN = 3;    // Digital pin for sensor input
const float WHEEL_DIAMETER = 0.067;  // Wheel diameter in meters
const int SLOTS_IN_DISK = 20;        // Number of slots in sensor disk
const float WHEEL_CIRCUMFERENCE = PI * WHEEL_DIAMETER;  // Wheel circumference in meters

// Debounce settings
const unsigned long DEBOUNCE_TIME = 1;  // Minimum time between valid readings (milliseconds)

// Variables for calculations
volatile unsigned long lastPulseTime = 0;
volatile unsigned long lastValidPulseTime = 0;
volatile unsigned long currentPulseTime = 0;
volatile boolean newPulse = false;
int speed = 0;
int rpm = 0;

// For averaging
const int AVERAGE_POINTS = 20;
int speedReadings[AVERAGE_POINTS];
int rpmReadings[AVERAGE_POINTS];
int readIndex = 0;

// CAN Bus setup
const int SPI_CS_PIN = 9;   // Chip select pin for CAN module

mcp2515_can CAN(SPI_CS_PIN);  // Create an instance of the MCP_CAN class

// SELECT UNITS
const bool METRIC = true;  // Set to false for mph
const float CONVERT_TO_KMH = 3.6;  // Conversion factor for km/h or mph
const float CONVERT_TO_MPH = 2.237;

void handlePulse() {
  lastPulseTime = currentPulseTime;
  currentPulseTime = micros();
  newPulse = true;
}

void setupSpeedSensor() {
  // Start serial communication for debugging
  Serial.begin(9600);

  // Initialize CAN Bus
  if (!(CAN.begin(CAN_500KBPS) == CAN_OK)) {
    while (1);  // Stop the program if CAN initialization fails
  }

  // Set CAN mode to normal
  CAN.setMode(MODE_NORMAL);

  pinMode(SENSOR_PIN, INPUT);

  // Initialize averaging array
  for(int i = 0; i < AVERAGE_POINTS; i++) {
    speedReadings[i] = 0;
  }

  attachInterrupt(digitalPinToInterrupt(SENSOR_PIN), handlePulse, RISING);
}

void loopSpeedSensor() {
  static unsigned long lastPrintTime = 0; // Tracks the last time a valid pulse was processed
  const unsigned long NO_PULSE_TIMEOUT = 50; // Timeout in ms to consider speed as 0

  if (newPulse) {
    // Convert to milliseconds for easier debugging
    float timeDiff_ms = (float)(currentPulseTime - lastValidPulseTime) / 1000.0;

    // Only process if enough time has passed (debounce)
    if (timeDiff_ms >= DEBOUNCE_TIME) {
      // Convert time difference to seconds for calculations
      float timeDiff_sec = timeDiff_ms / 1000.0;

      if (timeDiff_sec > 0) { // Prevent division by zero
        // One complete revolution takes (SLOTS_IN_DISK * timeDiff_sec) seconds
        float revTime = timeDiff_sec * SLOTS_IN_DISK;

        // Calculate raw RPM and round to integer
        int rawRpm = round(60.0 / revTime);

        // Add RPM to averaging array
        rpmReadings[readIndex] = rawRpm;

        // Calculate average RPM
        int avgRpm = 0;
        for (int i = 0; i < AVERAGE_POINTS; i++) {
            avgRpm += rpmReadings[i];
        }
        avgRpm /= AVERAGE_POINTS;
        rpm = avgRpm; // Store averaged RPM

        float units = METRIC ? CONVERT_TO_KMH : CONVERT_TO_MPH;
        // Calculate speed using averaged RPM
        int rawSpeed = round((WHEEL_CIRCUMFERENCE * (rpm / 60.0)) * units);

        // Add speed to averaging array
        speedReadings[readIndex] = rawSpeed;

        // Calculate average speed
        int avgSpeed = 0;
        for (int i = 0; i < AVERAGE_POINTS; i++) {
            avgSpeed += speedReadings[i];
        }
        avgSpeed /= AVERAGE_POINTS;
        avgSpeed /= 2; // Divide by 2 to get a more realistic value

        // Increment read index for next averaging cycle
        readIndex = (readIndex + 1) % AVERAGE_POINTS;

        // Send speed value over CAN Bus
        byte speedMessage[1];
        speedMessage[0] = avgSpeed & 0xFF;

        // Send the CAN message with ID 0x100, standard frame, 1 byte of data
        if (CAN.sendMsgBuf(0x100, 0, 1, speedMessage) == CAN_OK) {
            Serial.println("SENT speed: " + String(avgSpeed));
        } else {
            Serial.println("SPEED CAN MESSAGE ERROR!");
        }

        // Send averaged rpm value over CAN Bus
        //byte rpmMessage[2];

        // Split 16-bit rpm into 2 bytes
        //rpmMessage[0] = (rpm >> 8) & 0xFF; // High byte
        //rpmMessage[1] = rpm & 0xFF; // Low byte

        // Send the CAN message with ID 0x200, standard frame, 2 bytes of data
        // if (CAN.sendMsgBuf(0x200, 0, 2, rpmMessage) == CAN_OK) {
        //     Serial.println("SENT rpm: " + String(rpm));
        // } else {
        //     Serial.println("RPM CAN MESSAGE ERROR!");
        // }

        // Update the last valid pulse time
        lastValidPulseTime = currentPulseTime;
        lastPrintTime = millis();
      }
    }
    newPulse = false;
  }

  // Check if no pulse has been detected for the timeout period
  if ((millis() - lastValidPulseTime > NO_PULSE_TIMEOUT) && (millis() - lastPrintTime > NO_PULSE_TIMEOUT)) {
    // Print 0 speed only when pulses stop arriving
    byte speedMessage[1];  // Message data payload
    speedMessage[0] = 0 & 0xFF;  // Example: speed value in 1 byte

    // Send the CAN message with ID 0x100, standard frame, 1 byte of data
    if (CAN.sendMsgBuf(0x100, 0, 1, speedMessage) == CAN_OK) {
      Serial.println("SENT speed: 0");
    } else {
      Serial.println("SPEED CAN MESSAGE ERROR!");
    }

    // Send rpm value over CAN Bus
    //byte rpmMessage[2];  // Message data payload

    // Split 16-bit rpm into 2 bytes
    //rpmMessage[0] = (0 >> 8) & 0xFF;    // High byte
    //rpmMessage[1] = 0 & 0xFF;           // Low byte

    // Send the CAN message with ID 0x200, standard frame, 1 byte of data
    // if (CAN.sendMsgBuf(0x200, 0, 2, rpmMessage) == CAN_OK) {
    //   Serial.println("RPM rpm: 0");
    // } else {
    //   Serial.println("RPM CAN MESSAGE ERROR!");
    // }

    lastPrintTime = millis(); // Avoid repetitive printing
  }
}

// ---- SRF08 SETUP ----
void setupSRF08() {
  Wire.begin();
  delay(100);  // Let the sensor stabilize
}

// ---- SRF08 LOOP ----
void loopSRF08() {
  static unsigned long lastRead = 0;
  if (millis() - lastRead >= 300) {
    lastRead = millis();
    readSRF08();
  }
}

// ---- READ DISTANCE ----
void readSRF08() {
  // Send ranging command (0x51 = range in cm)
  Wire.beginTransmission(SRF08_ADDR);
  Wire.write(0x00);
  Wire.write(0x51);
  Wire.endTransmission();

  delay(70);  // Wait for measurement (~65 ms)

  // Request distance result (high and low byte)
  Wire.beginTransmission(SRF08_ADDR);
  Wire.write(0x02);
  Wire.endTransmission();

  Wire.requestFrom(SRF08_ADDR, 2);
  if (Wire.available() == 2) {
    byte high = Wire.read();
    byte low = Wire.read();
    int distance = (high << 8) | low;

    Serial.print("Ultra-sound distance: ");
    Serial.print(distance);
    Serial.println(" cm");
  } else {
    Serial.println("Failed to read distance");
  }
}

void setup() {
  setupSpeedSensor();
  setupSRF08();
}

void loop() {
  loopSpeedSensor();
  loopSRF08();
}
