/*
 * test_motors.ino
 * ===============
 * MIRA Rover — Motor Test
 * Tests all movement directions with PWM speed control.
 * Flash this to the Arduino Mega and watch the rover move.
 *
 * ── PIN MAPPING ──────────────────────────────────────────────────────────────
 *   E1 = 3   Left motor enable  (PWM — speed control)
 *   M1 = 5   Left motor direction
 *   E2 = 6   Right motor enable (PWM — speed control)
 *   M2 = 7   Right motor direction
 *
 * ── WIRING (L298N or equivalent) ─────────────────────────────────────────────
 *   Arduino pin 3  → ENA  (left motor enable)
 *   Arduino pin 5  → IN1  (left motor direction)
 *   Arduino pin 6  → ENB  (right motor enable)
 *   Arduino pin 7  → IN2  (right motor direction)
 *   Motor driver GND → Arduino GND (common ground — mandatory)
 *
 * ── SPEED TUNING ─────────────────────────────────────────────────────────────
 *   All speeds are PWM values 0–255.
 *   Start low and increase if motors don't move.
 *   If rover drifts left/right during FORWARD, adjust LEFT_TRIM or RIGHT_TRIM.
 *
 * ── HOW TO USE ───────────────────────────────────────────────────────────────
 *   1. Flash this sketch
 *   2. Open Serial Monitor at 115200 baud to see what's happening
 *   3. The rover cycles through: STOP → FORWARD → SLOW → REVERSE → RIGHT → LEFT
 *   4. Tune SPEED_* values until movement looks correct
 *   5. Copy final speed values to arduino_mira.ino
 */

// ── Pins ─────────────────────────────────────────────────────
#define E1  3    // Left motor  — PWM speed (must be PWM pin)
#define M1  5    // Left motor  — direction
#define E2  6    // Right motor — PWM speed (must be PWM pin)
#define M2  7    // Right motor — direction

// ── Speed values (0–255) — tune these ────────────────────────
#define SPEED_NORMAL   120   // standard forward speed
#define SPEED_SLOW      70   // slow/cautious forward
#define SPEED_REVERSE   90   // reverse speed
#define SPEED_TURN      90   // turning speed

// ── Trim correction (if rover drifts during FORWARD) ─────────
// Increase the weaker motor's trim value until it goes straight
// Example: if rover drifts right → increase RIGHT_TRIM slightly
#define LEFT_TRIM        0   // added to left motor PWM  (range: 0–30)
#define RIGHT_TRIM       0   // added to right motor PWM (range: 0–30)

// ── Test step duration ───────────────────────────────────────
#define STEP_MS       2000   // milliseconds per test step

// =============================================================================
// MOTOR FUNCTIONS
// =============================================================================

void stopMotors() {
  analogWrite(E1, 0);
  analogWrite(E2, 0);
  digitalWrite(M1, LOW);
  digitalWrite(M2, LOW);
}

void forwardMotors(int spd) {
  // Both motors forward
  digitalWrite(M1, HIGH);
  digitalWrite(M2, HIGH);
  analogWrite(E1, constrain(spd + LEFT_TRIM,  0, 255));
  analogWrite(E2, constrain(spd + RIGHT_TRIM, 0, 255));
}

void reverseMotors(int spd) {
  // Both motors reverse
  digitalWrite(M1, LOW);
  digitalWrite(M2, LOW);
  analogWrite(E1, constrain(spd + LEFT_TRIM,  0, 255));
  analogWrite(E2, constrain(spd + RIGHT_TRIM, 0, 255));
}

void turnRight(int spd) {
  // Left motor forward, right motor reverse → pivot right
  digitalWrite(M1, HIGH);
  digitalWrite(M2, LOW);
  analogWrite(E1, spd);
  analogWrite(E2, spd);
}

void turnLeft(int spd) {
  // Left motor reverse, right motor forward → pivot left
  digitalWrite(M1, LOW);
  digitalWrite(M2, HIGH);
  analogWrite(E1, spd);
  analogWrite(E2, spd);
}

// =============================================================================
// SETUP
// =============================================================================

void setup() {
  Serial.begin(115200);

  pinMode(E1, OUTPUT);
  pinMode(M1, OUTPUT);
  pinMode(E2, OUTPUT);
  pinMode(M2, OUTPUT);

  stopMotors();

  Serial.println("========================================");
  Serial.println("  MIRA Motor Test");
  Serial.println("  E1=3  M1=5  E2=6  M2=7");
  Serial.print  ("  SPEED_NORMAL=");  Serial.println(SPEED_NORMAL);
  Serial.print  ("  SPEED_SLOW=");    Serial.println(SPEED_SLOW);
  Serial.print  ("  SPEED_TURN=");    Serial.println(SPEED_TURN);
  Serial.println("========================================\n");
  Serial.println("Starting in 2 seconds...");
  delay(2000);
}

// =============================================================================
// LOOP — cycles through all movements
// =============================================================================

void loop() {

  // ── STOP ────────────────────────────────────────────────────
  Serial.println("[1] STOP");
  stopMotors();
  delay(STEP_MS);

  // ── FORWARD (normal speed) ───────────────────────────────────
  Serial.print("[2] FORWARD  spd="); Serial.println(SPEED_NORMAL);
  forwardMotors(SPEED_NORMAL);
  delay(STEP_MS);
  stopMotors();
  delay(500);

  // ── FORWARD (slow speed) ────────────────────────────────────
  Serial.print("[3] SLOW     spd="); Serial.println(SPEED_SLOW);
  forwardMotors(SPEED_SLOW);
  delay(STEP_MS);
  stopMotors();
  delay(500);

  // ── REVERSE ─────────────────────────────────────────────────
  Serial.print("[4] REVERSE  spd="); Serial.println(SPEED_REVERSE);
  reverseMotors(SPEED_REVERSE);
  delay(STEP_MS);
  stopMotors();
  delay(500);

  // ── TURN RIGHT ──────────────────────────────────────────────
  Serial.print("[5] TURN RIGHT  spd="); Serial.println(SPEED_TURN);
  turnRight(SPEED_TURN);
  delay(STEP_MS);
  stopMotors();
  delay(500);

  // ── TURN LEFT ───────────────────────────────────────────────
  Serial.print("[6] TURN LEFT   spd="); Serial.println(SPEED_TURN);
  turnLeft(SPEED_TURN);
  delay(STEP_MS);
  stopMotors();
  delay(500);

  Serial.println("\n--- Cycle complete. Repeating...\n");
  delay(1000);
}
