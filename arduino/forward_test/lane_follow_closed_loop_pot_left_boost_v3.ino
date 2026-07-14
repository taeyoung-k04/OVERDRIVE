/*
  Lane-following vehicle controller with potentiometer steering feedback

  Board: Arduino Uno / Nano (ATmega328P)

  Pin assignment
    Left drive motor  : D10, D11
    Right drive motor : D5,  D6
    Steering motor    : D3,  D9
    Steering pot      : A0

  Serial protocol: 115200 baud

    C,<steering>,<drive_pwm>\n
      steering : -1000 .. +1000
                 -1000 = left, 0 = center, +1000 = right
      drive_pwm: 0 .. 255, forward only

    X\n       : stop immediately
    Q\n       : print status once
    DBG,1\n   : enable 10 Hz telemetry
    DBG,0\n   : disable telemetry

  Measured steering calibration
    Full left : 455
    Center    : 396
    Full right: 313

  Left-turn compensation
    Incoming negative steering commands are multiplied by 1.25.
    The left target range uses up to -980 instead of -900.

  Drive-speed compensation
    Incoming drive PWM is multiplied by 1.25, with a minimum effective
    PWM of 90 and a maximum safety cap of 220.

  IMPORTANT
    1. Lift the wheels off the floor for the first steering test.
    2. Send C,-500,0 and C,500,0 with the drive PWM set to zero.
    3. If the steering moves opposite to the command, change
       INVERT_STEERING_MOTOR from false to true and upload again.
*/

#include <Arduino.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// -----------------------------------------------------------------------------
// Pin configuration
// -----------------------------------------------------------------------------

const uint8_t LEFT_MOTOR_1 = 10;   // red
const uint8_t LEFT_MOTOR_2 = 11;   // orange

const uint8_t RIGHT_MOTOR_1 = 5;   // yellow
const uint8_t RIGHT_MOTOR_2 = 6;   // green

const uint8_t STEERING_1 = 3;      // black
const uint8_t STEERING_2 = 9;      // brown

const uint8_t STEERING_POT_PIN = A0;

// Change only when the physical motor direction is reversed.
const bool INVERT_STEERING_MOTOR = false;

// Change these only when a drive motor rotates backward.
const bool INVERT_LEFT_DRIVE_MOTOR = false;
const bool INVERT_RIGHT_DRIVE_MOTOR = false;

// false: normal Python operation, watchdog = 400 ms
// true : easier Serial Monitor testing, watchdog = 3000 ms
const bool SERIAL_MONITOR_TEST_MODE = false;

// -----------------------------------------------------------------------------
// Potentiometer calibration
// -----------------------------------------------------------------------------

const bool STEERING_CALIBRATED = true;

const int POT_LEFT_RAW = 462; //462 455
const int POT_CENTER_RAW = 400; //400 396
const int POT_RIGHT_RAW = 317; //317 313

// Separate steering limits. The left side is allowed to use more of its
// calibrated range because the measured left travel is relatively small.
// Keep these below 1000 to leave a small safety margin at the hard stops.
const int LEFT_STEERING_TARGET_LIMIT = 980;
const int RIGHT_STEERING_TARGET_LIMIT = 900;

// Scale incoming steering commands independently.
// Example: -500 from Python becomes approximately -625 on the left.
// Increase LEFT_STEERING_GAIN if the vehicle still turns left too weakly.
const float LEFT_STEERING_GAIN = 1.15f;
const float RIGHT_STEERING_GAIN = 1.00f;

// Fine straight-ahead trim in normalized units.
// Positive = slightly right, negative = slightly left.
const int STEERING_CENTER_TRIM = 120;

// Sensor readings slightly outside calibration are allowed, but large
// deviations are treated as a wiring/sensor fault.
const int POT_CALIBRATION_MARGIN_RAW = 80;
const int POT_VALID_MIN_RAW = 3;
const int POT_VALID_MAX_RAW = 1020;

// -----------------------------------------------------------------------------
// Closed-loop steering tuning
// -----------------------------------------------------------------------------

const unsigned long STEERING_CONTROL_PERIOD_MS = 10;  // 100 Hz
const unsigned long STEERING_REVERSE_BRAKE_MS = 25;
const unsigned long STEERING_NO_PROGRESS_TIMEOUT_MS = 1200;
const unsigned long STEERING_DIRECTION_CHECK_MS = 180;

// Error is represented in normalized units: -1000 .. +1000.
// With this potentiometer range, 65 is roughly 4-5 ADC counts.
const int STEERING_POSITION_DEADBAND = 65;
const int STEERING_SLOW_ZONE = 220;

// Conservative initial PWM values for first vehicle testing.
const int STEERING_MIN_PWM = 72;
const int STEERING_MAX_PWM = 150;
const int STEERING_SLOW_MAX_PWM = 105;

const float STEERING_KP_PWM = 0.12f;
const float STEERING_KD_PWM = 1.50f;

// Lower values produce smoother but slower sensor response.
const float POT_FILTER_ALPHA = 0.20f;

// Motion-safety thresholds in normalized units.
const int STEERING_PROGRESS_DELTA = 40;
const int STEERING_WRONG_DIRECTION_DELTA = 70;
const int STEERING_HARD_GUARD = 975;

// Python normally sends commands at about 20 Hz.
const unsigned long COMMAND_TIMEOUT_MS =
  SERIAL_MONITOR_TEST_MODE ? 3000UL : 400UL;

// -----------------------------------------------------------------------------
// Drive-speed tuning
// -----------------------------------------------------------------------------

// All non-zero drive PWM commands received from Python are multiplied by this.
// 1.00 = unchanged, 1.20 = 20% faster, 1.30 = 30% faster.
const float DRIVE_PWM_GAIN = 1.20f;

// Helps the motors overcome static friction at low commands.
// Set to 0 if low-speed control becomes too abrupt.
const int DRIVE_MIN_EFFECTIVE_PWM = 200;

// Overall safety cap for drive speed. Increase gradually after testing.
const int DRIVE_MAX_PWM = 255;

// -----------------------------------------------------------------------------
// Runtime state
// -----------------------------------------------------------------------------

int targetSteeringCommand = 0;
int targetDrivePwm = 0;
int appliedDrivePwm = 0;

bool commandActive = false;
bool telemetryEnabled = false;
bool steeringFault = false;

unsigned long lastCommandTime = 0;
unsigned long lastSteeringUpdate = 0;
unsigned long lastTelemetryTime = 0;
unsigned long steeringDirectionChangedAt = 0;
unsigned long steeringProgressTime = 0;
unsigned long steeringDirectionCheckTime = 0;

int activeSteeringDirection = 0;     // -1 left, 0 stop, +1 right
int requestedSteeringDirection = 0;
int lastSteeringPwm = 0;

float filteredPotRaw = 0.0f;
int currentSteeringCommand = 0;
int previousSteeringCommand = 0;
int currentSteeringError = 0;

int steeringProgressReference = 0;
int steeringDirectionReference = 0;
int directionCheckRequestedDirection = 0;

const size_t RX_BUFFER_SIZE = 64;
char rxBuffer[RX_BUFFER_SIZE];
size_t rxLength = 0;

// -----------------------------------------------------------------------------
// Mapping and calibration helpers
// -----------------------------------------------------------------------------

long mapLong(long value, long inA, long inB, long outA, long outB)
{
  if (inA == inB) {
    return outA;
  }

  return outA + (value - inA) * (outB - outA) / (inB - inA);
}

bool calibrationLooksValid()
{
  const bool increasing =
    POT_LEFT_RAW < POT_CENTER_RAW && POT_CENTER_RAW < POT_RIGHT_RAW;

  const bool decreasing =
    POT_LEFT_RAW > POT_CENTER_RAW && POT_CENTER_RAW > POT_RIGHT_RAW;

  const int leftSpan = abs(POT_CENTER_RAW - POT_LEFT_RAW);
  const int rightSpan = abs(POT_RIGHT_RAW - POT_CENTER_RAW);
  const int totalSpan = abs(POT_RIGHT_RAW - POT_LEFT_RAW);

  return
    (increasing || decreasing) &&
    leftSpan >= 40 &&
    rightSpan >= 40 &&
    totalSpan >= 100;
}

bool rawPotLooksValid(int raw)
{
  if (raw < POT_VALID_MIN_RAW || raw > POT_VALID_MAX_RAW) {
    return false;
  }

  const int lowCalibration = min(POT_LEFT_RAW, POT_RIGHT_RAW);
  const int highCalibration = max(POT_LEFT_RAW, POT_RIGHT_RAW);

  return
    raw >= lowCalibration - POT_CALIBRATION_MARGIN_RAW &&
    raw <= highCalibration + POT_CALIBRATION_MARGIN_RAW;
}

int rawPotToSteeringCommand(int raw)
{
  const int lowRaw = min(POT_LEFT_RAW, POT_RIGHT_RAW);
  const int highRaw = max(POT_LEFT_RAW, POT_RIGHT_RAW);
  raw = constrain(raw, lowRaw, highRaw);

  const bool rawIsOnLeftSide =
    (POT_LEFT_RAW < POT_RIGHT_RAW)
      ? (raw <= POT_CENTER_RAW)
      : (raw >= POT_CENTER_RAW);

  long normalized = 0;

  if (rawIsOnLeftSide) {
    normalized = mapLong(raw, POT_LEFT_RAW, POT_CENTER_RAW, -1000, 0);
  }
  else {
    normalized = mapLong(raw, POT_CENTER_RAW, POT_RIGHT_RAW, 0, 1000);
  }

  return constrain((int)normalized, -1000, 1000);
}

int steeringCommandToRawPot(int command)
{
  command = constrain(command, -1000, 1000);

  if (command < 0) {
    return (int)mapLong(
      command,
      -1000,
      0,
      POT_LEFT_RAW,
      POT_CENTER_RAW
    );
  }

  return (int)mapLong(
    command,
    0,
    1000,
    POT_CENTER_RAW,
    POT_RIGHT_RAW
  );
}

int applySteeringCommandCompensation(int command)
{
  command = constrain(command, -1000, 1000);

  if (command < 0) {
    const int boosted = (int)round((float)command * LEFT_STEERING_GAIN);
    return constrain(boosted, -LEFT_STEERING_TARGET_LIMIT, 0);
  }

  const int boosted = (int)round((float)command * RIGHT_STEERING_GAIN);
  return constrain(boosted, 0, RIGHT_STEERING_TARGET_LIMIT);
}

int calculateAppliedDrivePwm(int requestedPwm)
{
  requestedPwm = constrain(requestedPwm, 0, 255);

  if (requestedPwm == 0) {
    return 0;
  }

  int boosted = (int)round((float)requestedPwm * DRIVE_PWM_GAIN);

  if (DRIVE_MIN_EFFECTIVE_PWM > 0) {
    boosted = max(boosted, DRIVE_MIN_EFFECTIVE_PWM);
  }

  return constrain(boosted, 0, DRIVE_MAX_PWM);
}

// -----------------------------------------------------------------------------
// Drive motor helpers
// -----------------------------------------------------------------------------

void driveMotorForward(uint8_t in1, uint8_t in2, int pwm, bool inverted)
{
  pwm = constrain(pwm, 0, 255);

  if (pwm == 0) {
    analogWrite(in1, 0);
    analogWrite(in2, 0);
    return;
  }

  if (inverted) {
    analogWrite(in1, pwm);
    analogWrite(in2, 0);
  }
  else {
    analogWrite(in1, 0);
    analogWrite(in2, pwm);
  }
}

void driveStop()
{
  analogWrite(LEFT_MOTOR_1, 0);
  analogWrite(LEFT_MOTOR_2, 0);
  analogWrite(RIGHT_MOTOR_1, 0);
  analogWrite(RIGHT_MOTOR_2, 0);
}

void driveForward(int pwm)
{
  pwm = constrain(pwm, 0, 255);

  driveMotorForward(
    LEFT_MOTOR_1,
    LEFT_MOTOR_2,
    pwm,
    INVERT_LEFT_DRIVE_MOTOR
  );

  driveMotorForward(
    RIGHT_MOTOR_1,
    RIGHT_MOTOR_2,
    pwm,
    INVERT_RIGHT_DRIVE_MOTOR
  );
}

// -----------------------------------------------------------------------------
// Steering motor helpers
// -----------------------------------------------------------------------------

void steeringStop()
{
  analogWrite(STEERING_1, 0);
  analogWrite(STEERING_2, 0);

  activeSteeringDirection = 0;
  lastSteeringPwm = 0;
}

void steeringLeftPwm(int pwm)
{
  pwm = constrain(pwm, 0, 255);

  if (INVERT_STEERING_MOTOR) {
    analogWrite(STEERING_1, pwm);
    analogWrite(STEERING_2, 0);
  }
  else {
    analogWrite(STEERING_1, 0);
    analogWrite(STEERING_2, pwm);
  }

  activeSteeringDirection = -1;
  lastSteeringPwm = pwm;
}

void steeringRightPwm(int pwm)
{
  pwm = constrain(pwm, 0, 255);

  if (INVERT_STEERING_MOTOR) {
    analogWrite(STEERING_1, 0);
    analogWrite(STEERING_2, pwm);
  }
  else {
    analogWrite(STEERING_1, pwm);
    analogWrite(STEERING_2, 0);
  }

  activeSteeringDirection = 1;
  lastSteeringPwm = pwm;
}

// -----------------------------------------------------------------------------
// Steering sensor and controller
// -----------------------------------------------------------------------------

void initializePotFilter()
{
  long sum = 0;

  for (int i = 0; i < 20; ++i) {
    sum += analogRead(STEERING_POT_PIN);
    delay(2);
  }

  filteredPotRaw = (float)sum / 20.0f;
  currentSteeringCommand =
    rawPotToSteeringCommand((int)round(filteredPotRaw));
  previousSteeringCommand = currentSteeringCommand;

  steeringProgressReference = currentSteeringCommand;
  steeringDirectionReference = currentSteeringCommand;
}

bool updatePotReading()
{
  const int raw = analogRead(STEERING_POT_PIN);

  if (!rawPotLooksValid(raw)) {
    return false;
  }

  filteredPotRaw += POT_FILTER_ALPHA * ((float)raw - filteredPotRaw);

  previousSteeringCommand = currentSteeringCommand;
  currentSteeringCommand =
    rawPotToSteeringCommand((int)round(filteredPotRaw));

  return true;
}

int calculateSteeringPwm(int error, int measuredVelocity)
{
  const int absError = abs(error);

  float pwm =
    (float)STEERING_MIN_PWM + STEERING_KP_PWM * (float)absError;

  const bool movingTowardTarget =
    (error > 0 && measuredVelocity > 0) ||
    (error < 0 && measuredVelocity < 0);

  if (movingTowardTarget) {
    pwm -= STEERING_KD_PWM * (float)abs(measuredVelocity);
  }

  if (absError < STEERING_SLOW_ZONE) {
    const int nearTargetCap = (int)mapLong(
      absError,
      STEERING_POSITION_DEADBAND,
      STEERING_SLOW_ZONE,
      STEERING_MIN_PWM,
      STEERING_SLOW_MAX_PWM
    );

    pwm = min(pwm, (float)nearTargetCap);
  }

  return constrain(
    (int)round(pwm),
    STEERING_MIN_PWM,
    STEERING_MAX_PWM
  );
}

void setSteeringFault(const __FlashStringHelper *message)
{
  const bool isNewFault = !steeringFault;

  steeringFault = true;
  steeringStop();
  driveStop();

  if (isNewFault) {
    Serial.println(message);
  }
}

void resetSteeringMotionSafety(int requestedDirection)
{
  const unsigned long now = millis();

  steeringProgressReference = currentSteeringCommand;
  steeringProgressTime = now;

  steeringDirectionReference = currentSteeringCommand;
  steeringDirectionCheckTime = now;
  directionCheckRequestedDirection = requestedDirection;
}

bool steeringMotionIsSafe(int requestedDirection)
{
  const unsigned long now = millis();

  // Do not drive farther outward at the measured hard limits.
  if (
    (currentSteeringCommand <= -STEERING_HARD_GUARD && requestedDirection < 0) ||
    (currentSteeringCommand >= STEERING_HARD_GUARD && requestedDirection > 0)
  ) {
    setSteeringFault(F("FAULT,STEERING_LIMIT"));
    return false;
  }

  // Reset safety references whenever the requested direction changes.
  if (requestedDirection != directionCheckRequestedDirection) {
    resetSteeringMotionSafety(requestedDirection);
    return true;
  }

  // Confirm that the sensor is changing in the expected direction.
  if (now - steeringDirectionCheckTime >= STEERING_DIRECTION_CHECK_MS) {
    const int delta =
      currentSteeringCommand - steeringDirectionReference;

    const bool movingWrongWay =
      (requestedDirection < 0 && delta > STEERING_WRONG_DIRECTION_DELTA) ||
      (requestedDirection > 0 && delta < -STEERING_WRONG_DIRECTION_DELTA);

    if (movingWrongWay) {
      setSteeringFault(F("FAULT,STEERING_DIRECTION"));
      return false;
    }

    steeringDirectionReference = currentSteeringCommand;
    steeringDirectionCheckTime = now;
  }

  // Detect a jammed steering mechanism or disconnected motor.
  if (
    abs(currentSteeringCommand - steeringProgressReference) >=
    STEERING_PROGRESS_DELTA
  ) {
    steeringProgressReference = currentSteeringCommand;
    steeringProgressTime = now;
  }

  if (now - steeringProgressTime > STEERING_NO_PROGRESS_TIMEOUT_MS) {
    setSteeringFault(F("FAULT,STEERING_NO_PROGRESS"));
    return false;
  }

  return true;
}

void updateSteeringClosedLoop()
{
  const unsigned long now = millis();

  if (now - lastSteeringUpdate < STEERING_CONTROL_PERIOD_MS) {
    return;
  }
  lastSteeringUpdate = now;

  if (!updatePotReading()) {
    setSteeringFault(F("FAULT,POT_SENSOR"));
    return;
  }

  if (!STEERING_CALIBRATED || !calibrationLooksValid()) {
    steeringStop();
    driveStop();
    return;
  }

  if (!commandActive || steeringFault) {
    steeringStop();
    return;
  }

  const int target = applySteeringCommandCompensation(
    targetSteeringCommand + STEERING_CENTER_TRIM
  );

  currentSteeringError = target - currentSteeringCommand;
  const int absError = abs(currentSteeringError);

  if (absError <= STEERING_POSITION_DEADBAND) {
    steeringStop();
    requestedSteeringDirection = 0;
    resetSteeringMotionSafety(0);
    return;
  }

  requestedSteeringDirection =
    currentSteeringError < 0 ? -1 : 1;

  if (!steeringMotionIsSafe(requestedSteeringDirection)) {
    return;
  }

  // Briefly stop before reversing H-bridge direction.
  if (
    activeSteeringDirection != 0 &&
    requestedSteeringDirection != activeSteeringDirection
  ) {
    steeringStop();
    steeringDirectionChangedAt = now;
    resetSteeringMotionSafety(requestedSteeringDirection);
    return;
  }

  if (
    activeSteeringDirection == 0 &&
    now - steeringDirectionChangedAt < STEERING_REVERSE_BRAKE_MS
  ) {
    steeringStop();
    return;
  }

  const int measuredVelocity =
    currentSteeringCommand - previousSteeringCommand;

  const int pwm =
    calculateSteeringPwm(currentSteeringError, measuredVelocity);

  if (requestedSteeringDirection < 0) {
    steeringLeftPwm(pwm);
  }
  else {
    steeringRightPwm(pwm);
  }
}

// -----------------------------------------------------------------------------
// Commands, watchdog and status
// -----------------------------------------------------------------------------

void carStop()
{
  targetSteeringCommand = 0;
  targetDrivePwm = 0;
  appliedDrivePwm = 0;

  commandActive = false;
  steeringFault = false;

  requestedSteeringDirection = 0;
  currentSteeringError = 0;

  driveStop();
  steeringStop();
  resetSteeringMotionSafety(0);
}

void acceptControlCommand(int steeringTarget, int drivePwm)
{
  const bool startingNewRun = !commandActive;

  targetSteeringCommand = constrain(steeringTarget, -1000, 1000);
  targetDrivePwm = constrain(drivePwm, 0, 255);

  commandActive = true;
  lastCommandTime = millis();

  if (startingNewRun) {
    steeringFault = false;
    resetSteeringMotionSafety(0);
  }
}

void printStatus()
{
  const int raw = analogRead(STEERING_POT_PIN);

  const int appliedTarget = applySteeringCommandCompensation(
    targetSteeringCommand + STEERING_CENTER_TRIM
  );

  const int targetRaw = steeringCommandToRawPot(appliedTarget);

  Serial.print(F("POS,raw="));
  Serial.print(raw);

  Serial.print(F(",filtered="));
  Serial.print(filteredPotRaw, 1);

  Serial.print(F(",current="));
  Serial.print(currentSteeringCommand);

  Serial.print(F(",command="));
  Serial.print(targetSteeringCommand);

  Serial.print(F(",target="));
  Serial.print(appliedTarget);

  Serial.print(F(",target_raw="));
  Serial.print(targetRaw);

  Serial.print(F(",error="));
  Serial.print(currentSteeringError);

  Serial.print(F(",pwm="));
  Serial.print(lastSteeringPwm);

  Serial.print(F(",drive_cmd="));
  Serial.print(targetDrivePwm);

  Serial.print(F(",drive_applied="));
  Serial.print(appliedDrivePwm);

  Serial.print(F(",active="));
  Serial.print(commandActive ? 1 : 0);

  Serial.print(F(",fault="));
  Serial.println(steeringFault ? 1 : 0);
}

void processSerialLine(char *line)
{
  while (*line == ' ' || *line == '\t') {
    ++line;
  }

  if (
    strcmp(line, "X") == 0 ||
    strcmp(line, "STOP") == 0
  ) {
    carStop();
    Serial.println(F("ACK,STOP"));
    return;
  }

  if (
    strcmp(line, "Q") == 0 ||
    strcmp(line, "STATUS") == 0
  ) {
    printStatus();
    return;
  }

  int debugValue = 0;
  if (sscanf(line, "DBG,%d", &debugValue) == 1) {
    telemetryEnabled = debugValue != 0;

    Serial.print(F("ACK,DBG,"));
    Serial.println(telemetryEnabled ? 1 : 0);
    return;
  }

  int steeringTarget = 0;
  int drivePwm = 0;

  if (
    sscanf(line, "C,%d,%d", &steeringTarget, &drivePwm) == 2
  ) {
    acceptControlCommand(steeringTarget, drivePwm);
    return;
  }

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
        rxLength = 0;
        carStop();
        Serial.println(F("ERR,RX_OVERFLOW"));
      }
    }
  }
}

void applyCommandWatchdog()
{
  if (
    commandActive &&
    millis() - lastCommandTime > COMMAND_TIMEOUT_MS
  ) {
    carStop();
    Serial.println(F("WATCHDOG,STOP"));
  }
}

void updateDrive()
{
  if (
    !commandActive ||
    targetDrivePwm <= 0 ||
    steeringFault ||
    !STEERING_CALIBRATED ||
    !calibrationLooksValid()
  ) {
    appliedDrivePwm = 0;
    driveStop();
    return;
  }

  appliedDrivePwm = calculateAppliedDrivePwm(targetDrivePwm);
  driveForward(appliedDrivePwm);
}

void updateTelemetry()
{
  if (!telemetryEnabled) {
    return;
  }

  const unsigned long now = millis();

  if (now - lastTelemetryTime >= 100) {
    lastTelemetryTime = now;
    printStatus();
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
  pinMode(STEERING_POT_PIN, INPUT);

  driveStop();
  steeringStop();

  initializePotFilter();

  const unsigned long now = millis();
  lastCommandTime = now;
  lastSteeringUpdate = now;
  lastTelemetryTime = now;
  steeringDirectionChangedAt = now;
  steeringProgressTime = now;
  steeringDirectionCheckTime = now;

  Serial.println(F("READY,LANE_FOLLOW_POT_CLOSED_LOOP_V3_LEFT_BOOST"));
  Serial.println(F("PROTOCOL,C,<steering -1000..1000>,<drive 0..255>"));
  Serial.println(F("STOP,X"));
  Serial.println(F("STATUS,Q"));

  if (!STEERING_CALIBRATED) {
    Serial.println(F("FAULT,CALIBRATION_DISABLED"));
  }
  else if (!calibrationLooksValid()) {
    Serial.println(F("FAULT,INVALID_POT_CALIBRATION"));
  }
  else if (!rawPotLooksValid(analogRead(STEERING_POT_PIN))) {
    Serial.println(F("FAULT,POT_SENSOR_AT_STARTUP"));
  }
  else {
    Serial.println(F("CALIBRATION,OK"));
  }

  printStatus();
}

void loop()
{
  readSerialCommands();
  applyCommandWatchdog();
  updateSteeringClosedLoop();
  updateDrive();
  updateTelemetry();
}
