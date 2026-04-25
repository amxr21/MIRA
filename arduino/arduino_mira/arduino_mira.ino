// ============================================================
// MIRA ROVER - SENSOR FUSION DATA PIPELINE
// Output format matches mira_pipeline.py expectations
// ============================================================

#include <Wire.h>
#include <VL53L1X.h>
#include <MPU6050_light.h>

// ===== TOF SENSORS =====
#define N_TOF 5
const int     XSHUT_PINS[N_TOF] = {38, 39, 40, 41, 42};
const uint8_t TOF_ADDRS[N_TOF]  = {0x30, 0x31, 0x32, 0x33, 0x34};
VL53L1X tof_sensors[N_TOF];
bool tof_valid[N_TOF] = {false, false, false, false, false};

// ===== ULTRASONIC SENSORS =====
#define ULTRA_FRONT_TRIG 24
#define ULTRA_FRONT_ECHO 25
#define ULTRA_REAR_TRIG  29
#define ULTRA_REAR_ECHO  28

// ===== IMU =====
MPU6050 imu(Wire);
bool imu_valid = false;

// ===== MOTORS =====
#define ENA 3
#define ENB 9
#define IN1 5
#define IN2 6
#define IN3 7
#define IN4 8

#define SPEED_NORMAL 60
#define SPEED_SLOW   40
#define SPEED_TURN   50

// ===== LDR =====
#define LDR_PIN A0

// ===== COMMUNICATION =====
#define SENSOR_UPDATE_RATE 30  // Hz (~33ms interval)
unsigned long last_sensor_read = 0;

// Command from Raspberry Pi
char current_command = 'X';  // X=stop, F=forward, B=back, L=left, R=right, S=slow
unsigned long last_command_time = 0;
#define WATCHDOG_TIMEOUT 500  // Stop if no command for 500ms

// ===== SENSOR DATA =====
float tof_distances[N_TOF] = {999.0, 999.0, 999.0, 999.0, 999.0};
float pitch = 0.0;
float roll = 0.0;
float yaw = 0.0;
float ldr_value = 1.0;
float ultra_front = 999.0;  // in METERS
float ultra_rear = 999.0;   // in METERS
float speed = 0.0;
bool tof_ok = false;

// ===== SETUP =====
void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000);
  
  setupMotors();
  initializeTOF();
  initializeUltrasonic();
  initializeIMU();
  pinMode(LDR_PIN, INPUT);
  
  delay(1000);
}

// ===== INITIALIZATION =====

void setupMotors() {
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  motorsStop();
}

void initializeTOF() {
  // Shut down all sensors
  for (int i = 0; i < N_TOF; i++) {
    pinMode(XSHUT_PINS[i], OUTPUT);
    digitalWrite(XSHUT_PINS[i], LOW);
  }
  delay(10);
  
  // Boot one at a time, assign unique address
  for (int i = 0; i < N_TOF; i++) {
    digitalWrite(XSHUT_PINS[i], HIGH);
    delay(50);
    
    tof_sensors[i].setTimeout(500);
    if (tof_sensors[i].init()) {
      tof_sensors[i].setAddress(TOF_ADDRS[i]);
      tof_sensors[i].setDistanceMode(VL53L1X::Short);
      tof_sensors[i].setMeasurementTimingBudget(33000);
      tof_sensors[i].startContinuous(33);
      tof_valid[i] = true;
    }
  }
}

void initializeUltrasonic() {
  pinMode(ULTRA_FRONT_TRIG, OUTPUT);
  pinMode(ULTRA_FRONT_ECHO, INPUT);
  pinMode(ULTRA_REAR_TRIG, OUTPUT);
  pinMode(ULTRA_REAR_ECHO, INPUT);
}

void initializeIMU() {
  byte status = imu.begin();
  if (status == 0) {
    imu.calcOffsets(true, true);
    imu_valid = true;
  }
}

// ===== MOTOR FUNCTIONS =====

void motorsStop() {
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

void motorsForward(int speed) {
  analogWrite(ENA, speed);
  analogWrite(ENB, speed);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, HIGH);
}

void motorsBackward(int speed) {
  analogWrite(ENA, speed);
  analogWrite(ENB, speed);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

void motorsTurnLeft(int speed) {
  analogWrite(ENA, speed);
  analogWrite(ENB, speed);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void motorsTurnRight(int speed) {
  analogWrite(ENA, speed);
  analogWrite(ENB, speed);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

void executeCommand(char cmd) {
  switch(cmd) {
    case 'F':
      motorsForward(SPEED_NORMAL);
      break;
    case 'S':
      motorsForward(SPEED_SLOW);
      break;
    case 'B':
      motorsBackward(SPEED_NORMAL);
      break;
    case 'L':
      motorsTurnLeft(SPEED_TURN);
      break;
    case 'R':
      motorsTurnRight(SPEED_TURN);
      break;
    case 'X':
    default:
      motorsStop();
      break;
  }
}

// ===== SENSOR READING =====

void readAllSensors() {
  // Read TOF sensors
  bool any_tof_ok = false;
  for (int i = 0; i < N_TOF; i++) {
    if (!tof_valid[i]) {
      tof_distances[i] = 999.0;
      continue;
    }
    
    int mm = tof_sensors[i].read(false);
    
    if (mm == 0) {
      // mm=0 means target TOO CLOSE (< 4cm) - sensor can't measure
      tof_distances[i] = 0.04;  // Report minimum detectable distance
      any_tof_ok = true;
    } else if (mm > 0 && mm < 4000) {
      // Valid reading: convert mm to meters
      tof_distances[i] = mm / 1000.0;
      any_tof_ok = true;
    } else {
      // mm >= 4000 or timeout: object is FAR AWAY or no detection
      tof_distances[i] = 999.0;
    }
  }
  tof_ok = any_tof_ok;
  
  // Read ultrasonic sensors - returns CM, convert to METERS
  float front_cm = readUltrasonic(ULTRA_FRONT_TRIG, ULTRA_FRONT_ECHO);
  if (front_cm > 0 && front_cm < 400) {
    ultra_front = front_cm / 100.0;  // Convert cm to meters
  } else {
    ultra_front = 999.0;
  }
  
  float rear_cm = readUltrasonic(ULTRA_REAR_TRIG, ULTRA_REAR_ECHO);
  if (rear_cm > 0 && rear_cm < 400) {
    ultra_rear = rear_cm / 100.0;  // Convert cm to meters
  } else {
    ultra_rear = 999.0;
  }
  
  // Read IMU
  if (imu_valid) {
    imu.update();
    pitch = imu.getAngleX();
    roll = imu.getAngleY();
    yaw = imu.getGyroZ();
  }
  
  // Read LDR (0-1 range)
  ldr_value = analogRead(LDR_PIN) / 1023.0;
  
  // Speed (always 0 - no encoder)
  speed = 0.0;
}

float readUltrasonic(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  long duration = pulseIn(echoPin, HIGH, 30000);
  if (duration == 0) return -1;
  
  // Returns distance in CENTIMETERS
  return duration * 0.034 / 2;
}

// ===== DATA OUTPUT - MATCHES PYTHON PARSER =====

void sendSensorData() {
  // Format: "tof1:X.XXX   tof2:X.XXX   tof3:X.XXX   tof4:X.XXX   tof5:X.XXX   
  //          pitch:X.XX   roll:X.XX   yaw:X.XX   ldr:X.XXX   
  //          ultra_front:X.XXX   ultra_rear:X.XXX   speed:X.XXX   STATUS"
  
  Serial.print("tof1:");
  Serial.print(tof_distances[0], 3);
  Serial.print("   tof2:");
  Serial.print(tof_distances[1], 3);
  Serial.print("   tof3:");
  Serial.print(tof_distances[2], 3);
  Serial.print("   tof4:");
  Serial.print(tof_distances[3], 3);
  Serial.print("   tof5:");
  Serial.print(tof_distances[4], 3);
  
  Serial.print("   pitch:");
  Serial.print(pitch, 2);
  Serial.print("   roll:");
  Serial.print(roll, 2);
  Serial.print("   yaw:");
  Serial.print(yaw, 2);
  
  Serial.print("   ldr:");
  Serial.print(ldr_value, 3);
  
  Serial.print("   ultra_front:");
  Serial.print(ultra_front, 3);
  Serial.print("   ultra_rear:");
  Serial.print(ultra_rear, 3);
  
  Serial.print("   speed:");
  Serial.print(speed, 3);
  
  // Status: 1 if TOF OK, 0 if not
  Serial.print("   ");
  Serial.println(tof_ok ? 1 : 0);
}

// ===== COMMAND PROCESSING =====

void processIncomingCommand() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'F' || cmd == 'S' || cmd == 'B' || cmd == 'L' || cmd == 'R' || cmd == 'X') {
      current_command = cmd;
      last_command_time = millis();
    }
  }
  
  // Watchdog: stop if no command received
  if (millis() - last_command_time > WATCHDOG_TIMEOUT) {
    current_command = 'X';
  }
}

// ===== MAIN LOOP =====

void loop() {
  unsigned long now = millis();
  
  // Read sensors at fixed rate (~30Hz)
  if (now - last_sensor_read >= (1000 / SENSOR_UPDATE_RATE)) {
    readAllSensors();
    sendSensorData();
    last_sensor_read = now;
  }
  
  // Process incoming commands from Raspberry Pi
  processIncomingCommand();
  
  // Execute motor command
  executeCommand(current_command);
}