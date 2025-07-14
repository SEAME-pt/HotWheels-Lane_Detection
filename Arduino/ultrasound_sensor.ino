#include <Wire.h>

#define SRF08_ADDR 0x70  // 7-bit I2C address of SRF08

void setup() {
  Wire.begin();
  Serial.begin(9600);
  delay(100);
}

void loop() {
  // 1. Send ranging command (0x51 = range in cm)
  Wire.beginTransmission(SRF08_ADDR);
  Wire.write(0x00);     // Command register
  Wire.write(0x51);     // Range in cm
  Wire.endTransmission();

  delay(70); // Wait for measurement (SRF08 needs ~65 ms)

  // 2. Read the result from register 0x02 (MSB) and 0x03 (LSB)
  Wire.beginTransmission(SRF08_ADDR);
  Wire.write(0x02);     // Starting register to read from
  Wire.endTransmission();

  Wire.requestFrom(SRF08_ADDR, 2);
  if (Wire.available() == 2) {
    byte high = Wire.read();
    byte low = Wire.read();
    int distance = (high << 8) | low;

    Serial.print("Distance: ");
    Serial.print(distance);
    Serial.println(" cm");
  } else {
    Serial.println("Failed to read distance");
  }

  delay(300); // Repeat every 300 ms
}
