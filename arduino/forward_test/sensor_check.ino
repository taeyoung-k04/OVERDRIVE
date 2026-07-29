// 왼쪽 초음파 센서
const int LEFT_TRIG = 12;
const int LEFT_ECHO = 13;

// 중앙 초음파 센서
const int CENTER_TRIG = 2;
const int CENTER_ECHO = 4;

// 오른쪽 초음파 센서
const int RIGHT_TRIG = 7;
const int RIGHT_ECHO = 8;

float readDistanceCm(int trigPin, int echoPin) {
  // 이전 신호 정리
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  // 10us 동안 초음파 발사
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // 최대 약 4m까지만 기다림
  unsigned long duration = pulseIn(echoPin, HIGH, 25000UL);

  // 반사 신호가 없으면 -1 반환
  if (duration == 0) {
    return -1.0;
  }

  float distance = duration * 0.0343 / 2.0;

  // 비정상 범위 제거
  if (distance < 2.0 || distance > 400.0) {
    return -1.0;
  }

  return distance;
}

void printDistance(float distance) {
  if (distance < 0) {
    Serial.print("TIMEOUT");
  } else {
    Serial.print(distance, 1);
    Serial.print("cm");
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(LEFT_TRIG, OUTPUT);
  pinMode(LEFT_ECHO, INPUT);

  pinMode(CENTER_TRIG, OUTPUT);
  pinMode(CENTER_ECHO, INPUT);

  pinMode(RIGHT_TRIG, OUTPUT);
  pinMode(RIGHT_ECHO, INPUT);

  digitalWrite(LEFT_TRIG, LOW);
  digitalWrite(CENTER_TRIG, LOW);
  digitalWrite(RIGHT_TRIG, LOW);

  Serial.println("3 ultrasonic sensors test start");
}

void loop() {
  float leftDistance = readDistanceCm(LEFT_TRIG, LEFT_ECHO);

  // 센서끼리 초음파가 섞이지 않도록 간격 확보
  delay(30);

  float centerDistance = readDistanceCm(CENTER_TRIG, CENTER_ECHO);

  delay(30);

  float rightDistance = readDistanceCm(RIGHT_TRIG, RIGHT_ECHO);

  Serial.print("LEFT: ");
  printDistance(leftDistance);

  Serial.print(" | CENTER: ");
  printDistance(centerDistance);

  Serial.print(" | RIGHT: ");
  printDistance(rightDistance);

  Serial.println();

  delay(50);
}