/*
  Lane-following car controller for DC-motor steering.

  Matching Python protocol (115200 baud):
    C,<steering>,<drive_pwm>\n
      steering : -1000 .. +1000
                 negative = left correction
                 positive = right correction
      drive_pwm: 0 .. 255, forward only

    X\n         immediate drive + steering stop

  IMPORTANT:
  - This is proportional pulse steering, not true angle control.
  - Pins 12 and 13 are not hardware-PWM pins on an Uno/Nano, so steering
    strength is produced by changing pulse ON-time within a repeating cycle.
  - For exact steering angles, add a potentiometer/encoder or use a servo.
*/

#include <Arduino.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// -----------------------------------------------------------------------------
// Motor pins: retained from the original sketch
// -----------------------------------------------------------------------------

const uint8_t LEFT_MOTOR_1 = 11;
const uint8_t LEFT_MOTOR_2 = 10;

const uint8_t RIGHT_MOTOR_1 = 8;
const uint8_t RIGHT_MOTOR_2 = 9;

const uint8_t STEERING_1 = 12;
const uint8_t STEERING_2 = 13;

// Change to true if positive Python steering turns the real wheels left.
const bool INVERT_STEERING = false;

// -----------------------------------------------------------------------------
// Control tuning
// -----------------------------------------------------------------------------

const int STEERING_COMMAND_MAX = 1000;

// Commands smaller than this do not energize the steering motor.
const int STEERING_DEADBAND = 70;

// Software pulse control for steering pins 12/13.
// One cycle repeats every 110 ms. Small corrections are ON briefly; strong
// corrections remain ON for most of the cycle.
const unsigned long STEERING_CYCLE_MS = 110;
const unsigned long STEERING_MIN_ON_MS = 12;
const unsigned long STEERING_MAX_ON_MS = 88;

// Stop briefly before reversing steering motor direction.
const unsigned long STEERING_REVERSE_BRAKE_MS = 25;

// Python sends at 20 Hz. If commands stop arriving, halt everything.
const unsigned long COMMAND_TIMEOUT_MS = 400;

// -----------------------------------------------------------------------------
// Runtime state
// -----------------------------------------------------------------------------

int targetSteering = 0;  // -1000 .. +1000
int targetDrivePwm = 0;  // 0 .. 255
bool commandActive = false;

unsigned long lastCommandTime = 0;
unsigned long steeringCycleStart = 0;
unsigned long steeringDirectionChangedAt = 0;
int previousRequestedDirection = 0;  // -1 left, 0 stop, +1 right

const size_t RX_BUFFER_SIZE = 48;
char rxBuffer[RX_BUFFER_SIZE];
size_t rxLength = 0;

// -----------------------------------------------------------------------------
// Drive motor helpers
// -----------------------------------------------------------------------------

void motorForward(uint8_t in1, uint8_t in2, int pwm)
{
  pwm = constrain(pwm, 0, 255);
  digitalWrite(in1, LOW);
  analogWrite(in2, pwm);
}

void motorStop(uint8_t in1, uint8_t in2)
{
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
}

void driveForward(int pwm)
{
  if (pwm <= 0) {
    motorStop(LEFT_MOTOR_1, LEFT_MOTOR_2);
    motorStop(RIGHT_MOTOR_1, RIGHT_MOTOR_2);
    return;
  }

  // LEFT_MOTOR_2=10 and RIGHT_MOTOR_2=9 are PWM pins on Uno/Nano.
  motorForward(LEFT_MOTOR_1, LEFT_MOTOR_2, pwm);
  motorForward(RIGHT_MOTOR_1, RIGHT_MOTOR_2, pwm);
}

void driveStop()
{
  motorStop(LEFT_MOTOR_1, LEFT_MOTOR_2);
  motorStop(RIGHT_MOTOR_1, RIGHT_MOTOR_2);
}

// -----------------------------------------------------------------------------
// Steering motor helpers
// -----------------------------------------------------------------------------

void steeringStop()
{
  digitalWrite(STEERING_1, LOW);
  digitalWrite(STEERING_2, LOW);
}

void steeringLeftOn()
{
  // Same electrical direction as the original steering_A().
  digitalWrite(STEERING_1, LOW);
  digitalWrite(STEERING_2, HIGH);
}

void steeringRightOn()
{
  // Same electrical direction as the original steering_D().
  digitalWrite(STEERING_1, HIGH);
  digitalWrite(STEERING_2, LOW);
}

void applySteeringDirection(int direction)
{
  if (INVERT_STEERING) {
    direction = -direction;
  }

  if (direction < 0) {
    steeringLeftOn();
  }
  else if (direction > 0) {
    steeringRightOn();
  }
  else {
    steeringStop();
  }
}

// -----------------------------------------------------------------------------
// Safety and command handling
// -----------------------------------------------------------------------------

void carStop()
{
  targetSteering = 0;
  targetDrivePwm = 0;
  commandActive = false;
  previousRequestedDirection = 0;

  driveStop();
  steeringStop();
}

void acceptControlCommand(int steering, int drivePwm)
{
  targetSteering = constrain(
    steering,
    -STEERING_COMMAND_MAX,
    STEERING_COMMAND_MAX
  );
  targetDrivePwm = constrain(drivePwm, 0, 255);
  commandActive = true;
  lastCommandTime = millis();
}

void processSerialLine(char *line)
{
  // Ignore leading spaces.
  while (*line == ' ' || *line == '\t') {
    ++line;
  }

  if (strcmp(line, "X") == 0 || strcmp(line, "STOP") == 0) {
    carStop();
    Serial.println(F("ACK,STOP"));
    return;
  }

  int steering = 0;
  int drivePwm = 0;

  if (sscanf(line, "C,%d,%d", &steering, &drivePwm) == 2) {
    acceptControlCommand(steering, drivePwm);
    return;
  }

  // Invalid messages never move the car.
  Serial.print(F("ERR,"));
  Serial.println(line);
}

void readSerialCommands()
{
  while (Serial.available() > 0) {
    const char c = (char)Serial.read();

    if (c == '\n') {
      rxBuffer[rxLength] = '\0';
      if (rxLength > 0) {
        processSerialLine(rxBuffer);
      }
      rxLength = 0;
    }
    else if (c != '\r') {
      if (rxLength < RX_BUFFER_SIZE - 1) {
        rxBuffer[rxLength++] = c;
      }
      else {
        // Overflow: discard the malformed line safely.
        rxLength = 0;
        carStop();
        Serial.println(F("ERR,OVERFLOW"));
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Periodic output updates
// -----------------------------------------------------------------------------

void updateDrive()
{
  if (!commandActive || targetDrivePwm <= 0) {
    driveStop();
    return;
  }

  driveForward(targetDrivePwm);
}

void updateSteering()
{
  const unsigned long now = millis();
  const int magnitude = abs(targetSteering);

  int requestedDirection = 0;
  if (commandActive && magnitude > STEERING_DEADBAND) {
    requestedDirection = (targetSteering < 0) ? -1 : 1;
  }

  if (requestedDirection != previousRequestedDirection) {
    steeringStop();
    previousRequestedDirection = requestedDirection;
    steeringDirectionChangedAt = now;
    steeringCycleStart = now;
  }

  if (requestedDirection == 0) {
    steeringStop();
    return;
  }

  // Protect the H-bridge and motor from instantaneous direction reversal.
  if (now - steeringDirectionChangedAt < STEERING_REVERSE_BRAKE_MS) {
    steeringStop();
    return;
  }

  unsigned long elapsed = now - steeringCycleStart;
  if (elapsed >= STEERING_CYCLE_MS) {
    const unsigned long completedCycles = elapsed / STEERING_CYCLE_MS;
    steeringCycleStart += completedCycles * STEERING_CYCLE_MS;
    elapsed = now - steeringCycleStart;
  }

  const unsigned long onTime = map(
    magnitude,
    STEERING_DEADBAND + 1,
    STEERING_COMMAND_MAX,
    STEERING_MIN_ON_MS,
    STEERING_MAX_ON_MS
  );

  if (elapsed < onTime) {
    applySteeringDirection(requestedDirection);
  }
  else {
    steeringStop();
  }
}

void applyCommandWatchdog()
{
  if (!commandActive) {
    return;
  }

  if (millis() - lastCommandTime > COMMAND_TIMEOUT_MS) {
    carStop();
    Serial.println(F("WATCHDOG,STOP"));
  }
}

// -----------------------------------------------------------------------------
// Arduino lifecycle
// -----------------------------------------------------------------------------

void setup()
{
  Serial.begin(115200);

  pinMode(LEFT_MOTOR_1, OUTPUT);
  pinMode(LEFT_MOTOR_2, OUTPUT);
  pinMode(RIGHT_MOTOR_1, OUTPUT);
  pinMode(RIGHT_MOTOR_2, OUTPUT);
  pinMode(STEERING_1, OUTPUT);
  pinMode(STEERING_2, OUTPUT);

  carStop();
  steeringCycleStart = millis();
  steeringDirectionChangedAt = millis();
  lastCommandTime = millis();

  Serial.println(F("READY,LANE_FOLLOW_DC_V1"));
  Serial.println(F("PROTOCOL,C,<steering -1000..1000>,<drive 0..255>"));
  Serial.println(F("STOP,X"));
}

void loop()
{
  readSerialCommands();
  applyCommandWatchdog();
  updateDrive();
  updateSteering();
}
