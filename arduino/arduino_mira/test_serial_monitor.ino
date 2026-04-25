/*
 * test_serial_monitor.ino
 * =======================
 * Standalone sensor test — run this in Arduino IDE Serial Monitor
 * to verify all sensors are wired and working BEFORE connecting to the Pi.
 *
 * This sketch does NOT send motor commands or Pi CSV format.
 * It prints a clean human-readable table of all sensor readings.
 *
 * ── HOW TO USE ────────────────────────────────────────────────────────────────
 *   1. Flash this sketch to the Arduino Mega
 *   2. Open Arduino IDE → Tools → Serial Monitor
 *   3. Set baud rate to 115200
 *   4. Watch live readings from all sensors
 *   5. Wave hand in front of each TOF sensor to confirm it responds
 *   6. Tilt the board to confirm IMU pitch/roll change
 *   7. Cover/uncover LDR to confirm it changes
 *   8. Once everything checks out → flash arduino_mira.ino for the real run
 *
 * ── EXPECTED OUTPUT EXAMPLE ──────────────────────────────────────────────────
 *   ========================================
 *   TOF0 (C-FRONT) :   1.24m   ok
 *   TOF1 (L-FRONT) :   2.01m   ok
 *   TOF2 (R-FRONT) :   1.87m   ok
 *   TOF3 (L-SIDE ) :   0.95m   ok
 *   TOF4 (R-SIDE ) : 999.00m   timeout
 *   ----------------------------------------
 *   Pitch  :  +2.30 deg
 *   Roll   :  -0.80 deg
 *   Yaw    :  +0.10 deg/s
 *   ----------------------------------------
 *   LDR    :  0.82  (bright)
 *   Ultra  :  1.50m
 *   Speed  :  0.00 m/s
 *   ========================================
 *
 * ── LIBRARIES NEEDED ─────────────────────────────────────────────────────────
 *   - SparkFun VL53L1X (by SparkFun Electronics)
 *   - MPU6050 (by Electronic Cats or Jeff Rowberg)
 *   - Wire (built-in)
 */

#include <Wire.h>
#include "SparkFun_VL53L1X.h"
#include "MPU6050.h"

// ── Pins — must match arduino_mira.ino ───────────────────────
#define LDR_PIN       A0
#define ULTRA_FRONT_TRIG  22
#define ULTRA_FRONT_ECHO  23
#define ULTRA_REAR_TRIG   24
#define ULTRA_REAR_ECHO   25
// ── I2C ──────────────────────────────────────────────────────
#define TCA_ADDR       0x70
#define MPU_FRONT_ADDR 0x68

// ── Constants ────────────────────────────────────────────────
#define N_TOF           5
#define TOF_TIMEOUT_MM  8000
// ── Globals ──────────────────────────────────────────────────
SFEVL53L1X tof_sensor;
MPU6050    mpu(MPU_FRONT_ADDR);

float speed_mps = 0.0;   // always 0 — no encoder fitted

float tof_m[N_TOF]     = {999.0, 999.0, 999.0, 999.0, 999.0};
bool  tof_valid[N_TOF] = {false};

const char* TOF_NAMES[N_TOF] = {
  "TOF0 (C-FRONT)",
  "TOF1 (L-FRONT)",
  "TOF2 (R-FRONT)",
  "TOF3 (L-SIDE )",
  "TOF4 (R-SIDE )"
};

// ── TCA9548A ─────────────────────────────────────────────────
void tcaSelect(uint8_t ch) {
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(1 << ch);
  Wire.endTransmission();
  delay(1);
}

void tcaClose() {
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(0x00);
  Wire.endTransmission();
}

// ── Encoder ──────────────────────────────────────────────────
// ── Ultrasonic ───────────────────────────────────────────────
float readUltrasonic(uint8_t trig, uint8_t echo) {
  digitalWrite(trig, LOW);
  delayMicroseconds(2);
  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW);
  long dur = pulseIn(echo, HIGH, 30000);
  if (dur == 0) return 9.99;
  return (dur * 0.000343f) / 2.0f;
}

// ── Setup ────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000);

  Serial.println("\n[MIRA] Sensor Test Starting...");

  // TOF sensors
  Serial.println("[Init] TOF sensors...");
  for (uint8_t ch = 0; ch < N_TOF; ch++) {
    tcaSelect(ch);
    if (tof_sensor.begin() == 0) {
      tof_sensor.setDistanceModeLong();
      tof_sensor.setTimingBudgetInMs(20);
      tof_sensor.setIntermeasurementPeriod(33);
      tof_sensor.startRanging();
      tof_valid[ch] = true;
      Serial.print("  Channel "); Serial.print(ch); Serial.println(" : OK");
    } else {
      Serial.print("  Channel "); Serial.print(ch); Serial.println(" : FAILED — check wiring");
    }
  }
  tcaClose();

  // IMU
  Serial.println("[Init] MPU-6050...");
  mpu.initialize();
  if (mpu.testConnection()) {
    mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);
    mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);
    Serial.println("  MPU-6050 : OK");
  } else {
    Serial.println("  MPU-6050 : FAILED — check SDA/SCL wiring and I2C address (0x68)");
  }

  // Ultrasonic
  pinMode(ULTRA_FRONT_TRIG, OUTPUT); pinMode(ULTRA_FRONT_ECHO, INPUT);
  pinMode(ULTRA_REAR_TRIG,  OUTPUT); pinMode(ULTRA_REAR_ECHO,  INPUT);
  Serial.println("[Init] Ultrasonic x2 (front + rear) : ready");

  Serial.println("\n[Running] Reading sensors every 500ms...\n");
  delay(500);
}

// ── Loop ─────────────────────────────────────────────────────
void loop() {
  unsigned long now = millis();

  // TOF
  bool any_tof_ok = false;
  for (uint8_t ch = 0; ch < N_TOF; ch++) {
    if (!tof_valid[ch]) { tof_m[ch] = 999.0; continue; }
    tcaSelect(ch);
    if (tof_sensor.checkForDataReady()) {
      uint16_t mm = tof_sensor.getDistance();
      tof_sensor.clearInterrupt();
      if (mm > 0 && mm < TOF_TIMEOUT_MM) {
        tof_m[ch] = mm / 1000.0f;
        any_tof_ok = true;
      } else {
        tof_m[ch] = 999.0;
      }
    }
  }
  tcaClose();

  // IMU
  int16_t ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw;
  mpu.getMotion6(&ax_raw, &ay_raw, &az_raw, &gx_raw, &gy_raw, &gz_raw);
  float ax = ax_raw / 16384.0f, ay = ay_raw / 16384.0f, az = az_raw / 16384.0f;
  float pitch = atan2(ay, sqrt(ax*ax + az*az)) * 180.0f / PI;
  float roll  = atan2(-ax, az) * 180.0f / PI;
  float yaw   = gz_raw / 131.0f;

  // LDR
  float ldr = analogRead(LDR_PIN) / 1023.0f;
  const char* ldr_label = (ldr >= 0.6) ? "bright" : (ldr >= 0.25) ? "dim" : "DARK";

  // Ultrasonic
  float ultra_front = readUltrasonic(ULTRA_FRONT_TRIG, ULTRA_FRONT_ECHO);
  float ultra_rear  = readUltrasonic(ULTRA_REAR_TRIG,  ULTRA_REAR_ECHO);

  // Speed — no encoder fitted
  // speed_mps stays 0.0

  // ── Print ───────────────────────────────────────────────
  Serial.println("========================================");

  for (uint8_t i = 0; i < N_TOF; i++) {
    Serial.print(TOF_NAMES[i]);
    Serial.print(" : ");
    if (tof_m[i] >= 999.0) {
      Serial.println("  ---    timeout");
    } else {
      Serial.print(tof_m[i], 2);
      Serial.print("m");
      if (tof_m[i] < 0.5)       Serial.println("  !! STOP range");
      else if (tof_m[i] < 1.5)  Serial.println("  !  SLOW range");
      else                       Serial.println("  ok");
    }
  }

  Serial.print("TOF status   : ");
  Serial.println(any_tof_ok ? "OK (at least one valid)" : "ALL FAILED — ultrasonic fallback active");

  Serial.println("----------------------------------------");

  Serial.print("Pitch  : ");
  Serial.print(pitch, 2);
  Serial.println(" deg");

  Serial.print("Roll   : ");
  Serial.print(roll, 2);
  Serial.println(" deg");

  Serial.print("Yaw    : ");
  Serial.print(yaw, 2);
  Serial.println(" deg/s (raw gyro Z)");

  Serial.println("----------------------------------------");

  Serial.print("LDR    : ");
  Serial.print(ldr, 2);
  Serial.print("  (");
  Serial.print(ldr_label);
  Serial.println(")");

  Serial.print("Ultra F: ");
  Serial.print(ultra_front, 2);
  Serial.println("m");

  Serial.print("Ultra R: ");
  Serial.print(ultra_rear, 2);
  Serial.println("m");

  Serial.print("Speed  : ");
  Serial.print(speed_mps, 2);
  Serial.println(" m/s");

  Serial.println("========================================\n");

  delay(500);  // 2 readings per second — easy to read in Serial Monitor
}
