// 가변저항 연결 핀
const int POT_PIN = A0;

void setup() {
  Serial.begin(115200);
  pinMode(POT_PIN, INPUT);

  Serial.println("Potentiometer test start");
}

void loop() {
  int potValue = analogRead(POT_PIN);

  Serial.print("Potentiometer: ");
  Serial.println(potValue);

  delay(100);
}