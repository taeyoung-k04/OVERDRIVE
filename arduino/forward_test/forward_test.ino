int leftMotor1 = 11;
int leftMotor2 = 10;

int rightMotor1 = 8;
int rightMotor2 = 9;

int steering1 = 12;
int steering2 = 13;

// 주행 속도 조절: 0~255
const int DRIVE_SPEED = 100;

// 조향 속도
const int STEERING_SPEED = 100;

const unsigned long DRIVE_TIMEOUT = 200;
const unsigned long STEERING_TIMEOUT = 150;

int driveState = 0;     // 0: 정지, 1: 전진, -1: 후진
int steeringState = 0;  // 0: 정지, -1: A방향, 1: D방향

unsigned long lastDriveCmdTime = 0;
unsigned long lastSteeringCmdTime = 0;

void motor_backward(int IN1, int IN2, int speed)
{
  analogWrite(IN1, speed);
  digitalWrite(IN2, LOW);
}

void motor_forward(int IN1, int IN2, int speed)
{
  digitalWrite(IN1, LOW);
  analogWrite(IN2, speed);
}

void motor_stop(int IN1, int IN2)
{
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
}

void drive_forward()
{
  motor_forward(leftMotor1, leftMotor2, DRIVE_SPEED);
  motor_forward(rightMotor1, rightMotor2, DRIVE_SPEED);
}

void drive_backward()
{
  motor_backward(leftMotor1, leftMotor2, DRIVE_SPEED);
  motor_backward(rightMotor1, rightMotor2, DRIVE_SPEED);
}

void drive_stop()
{
  motor_stop(leftMotor1, leftMotor2);
  motor_stop(rightMotor1, rightMotor2);
}

void steering_A()
{
  motor_forward(steering1, steering2, STEERING_SPEED);
}

void steering_D()
{
  motor_backward(steering1, steering2, STEERING_SPEED);
}

void steering_stop()
{
  motor_stop(steering1, steering2);
}

void car_stop()
{
  drive_stop();
  steering_stop();

  driveState = 0;
  steeringState = 0;
}

void read_serial_command()
{
  while (Serial.available() > 0) {
    char cmd = Serial.read();

    if (cmd == 'w' || cmd == 'W') {
      driveState = 1;
      lastDriveCmdTime = millis();
    }
    else if (cmd == 'x' || cmd == 'X') {
      driveState = -1;
      lastDriveCmdTime = millis();
    }
    else if (cmd == 'a' || cmd == 'A') {
      steeringState = -1;
      lastSteeringCmdTime = millis();
    }
    else if (cmd == 'd' || cmd == 'D') {
      steeringState = 1;
      lastSteeringCmdTime = millis();
    }
    else if (cmd == 's' || cmd == 'S') {
      Serial.println("Stop");
      car_stop();
    }
  }
}

void update_drive()
{
  unsigned long now = millis();

  if (driveState != 0 && now - lastDriveCmdTime > DRIVE_TIMEOUT) {
    driveState = 0;
  }

  if (driveState == 1) {
    drive_forward();
  }
  else if (driveState == -1) {
    drive_backward();
  }
  else {
    drive_stop();
  }
}

void update_steering()
{
  unsigned long now = millis();

  if (steeringState != 0 && now - lastSteeringCmdTime > STEERING_TIMEOUT) {
    steeringState = 0;
  }

  if (steeringState == -1) {
    steering_A();
  }
  else if (steeringState == 1) {
    steering_D();
  }
  else {
    steering_stop();
  }
}

void setup() {
  Serial.begin(9600);

  pinMode(leftMotor1, OUTPUT);
  pinMode(leftMotor2, OUTPUT);

  pinMode(rightMotor1, OUTPUT);
  pinMode(rightMotor2, OUTPUT);

  pinMode(steering1, OUTPUT);
  pinMode(steering2, OUTPUT);

  car_stop();

  lastDriveCmdTime = millis();
  lastSteeringCmdTime = millis();

  Serial.println("Ready");
  Serial.println("w: forward");
  Serial.println("x: backward");
  Serial.println("a: steering A");
  Serial.println("d: steering D");
  Serial.println("s: stop");
}

void loop() {
  read_serial_command();

  update_drive();
  update_steering();
}