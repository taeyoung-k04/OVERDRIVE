const int leftMotor1 = 10; // red
const int leftMotor2 = 11; // orange

const int rightMotor1 = 5; // yellow
const int rightMotor2 = 6; // green

const int steering1 = 3; // black
const int steering2 = 9; // brown

const int DEFAULT_DRIVE_SPEED = 100;
const int DEFAULT_STEERING_SPEED = 100;

const unsigned long DRIVE_TIMEOUT = 200;
const unsigned long STEERING_TIMEOUT = 150;
const unsigned long DIRECTION_CHANGE_DELAY = 50;

// 상태: -1 = 역방향/좌, 0 = 정지, 1 = 정방향/우
int driveState = 0;
int steeringState = 0;
int driveSpeed = DEFAULT_DRIVE_SPEED;
int steeringSpeed = DEFAULT_STEERING_SPEED;

// 모터에 적용된 상태
int appliedDriveState = 0;
int appliedSteeringState = 0;

unsigned long lastDriveCmdTime = 0;
unsigned long lastSteeringCmdTime = 0;
unsigned long driveDirectionChangeTime = 0;
unsigned long steeringDirectionChangeTime = 0;

bool driveDirectionChanging = false;
bool steeringDirectionChanging = false;

const int SERIAL_BUFFER_SIZE = 16;
char serialBuffer[SERIAL_BUFFER_SIZE];
int serialBufferIndex = 0;

// 모터 제어 함수
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

// 주행 제어 함수
void drive_forward(int speed)
{
    motor_forward(leftMotor1, leftMotor2, speed);
    motor_forward(rightMotor1, rightMotor2, speed);
}

void drive_backward(int speed)
{
    motor_backward(leftMotor1, leftMotor2, speed);
    motor_backward(rightMotor1, rightMotor2, speed);
}

void drive_stop()
{
    motor_stop(leftMotor1, leftMotor2);
    motor_stop(rightMotor1, rightMotor2);
}

// 조향 제어 함수
void steering_left(int speed)
{
    motor_forward(steering1, steering2, speed);
}

void steering_right(int speed)
{
    motor_backward(steering1, steering2, speed);
}

void steering_stop()
{
    motor_stop(steering1, steering2);
}

// 차량 정지 함수
void car_stop()
{
    driveState = 0;
    steeringState = 0;
    appliedDriveState = 0;
    appliedSteeringState = 0;
    driveDirectionChanging = false;
    steeringDirectionChanging = false;

    drive_stop();
    steering_stop();
}

bool parse_command(const char *message, char &cmd, int &speed, bool &stop)
{
    cmd = '\0';
    speed = -1;
    stop = false;

    if (message[0] == '\0')
        return false;

    if (strcmp(message, "stop") == 0)
    {
        stop = true;
        return true;
    }

    cmd = tolower(message[0]);
    if (cmd != 'w' && cmd != 's' && cmd != 'a' && cmd != 'd')
        return false;

    if (message[1] == '\0')
        return true;

    int value = 0;
    for (int i = 1; message[i] != '\0'; i++)
    {
        if (!isDigit(message[i]))
            return false;

        value = value * 10 + (message[i] - '0');
        if (value > 255)
            return false;
    }

    speed = value;
    return true;
}

void execute_command(const char *message)
{
    char cmd;
    int speed;
    bool stop;

    if (!parse_command(message, cmd, speed, stop))
    {
        Serial.println("Invalid command");
        return;
    }

    if (stop)
    {
        car_stop();

        Serial.println("Stop");
        return;
    }

    unsigned long now = millis();

    if (cmd == 'w' || cmd == 's')
    {
        driveState = (cmd == 'w') ? 1 : -1;
        if (speed >= 0)
            driveSpeed = constrain(speed, 0, 255);
        else
            driveSpeed = DEFAULT_DRIVE_SPEED;
        lastDriveCmdTime = now;
    }
    else
    {
        steeringState = (cmd == 'a') ? -1 : 1;
        if (speed >= 0)
            steeringSpeed = constrain(speed, 0, 255);
        else
            driveSpeed = DEFAULT_DRIVE_SPEED;
        lastSteeringCmdTime = now;
    }
}

void read_serial_command()
{
    while (Serial.available() > 0)
    {
        char c = Serial.read();

        if (c == '\r')
            continue;

        if (c == '\n')
        {
            serialBuffer[serialBufferIndex] = '\0';
            if (serialBufferIndex > 0)
                execute_command(serialBuffer);
            serialBufferIndex = 0;
        }
        else if (serialBufferIndex < SERIAL_BUFFER_SIZE - 1)
        {
            serialBuffer[serialBufferIndex++] = c;
        }
        else
        {
            serialBufferIndex = 0;
        }
    }
}

void update_drive()
{
    unsigned long now = millis();

    if (driveState != 0 && now - lastDriveCmdTime > DRIVE_TIMEOUT)
        driveState = 0;

    if (driveState == 0 || driveSpeed == 0)
    {
        appliedDriveState = 0;
        driveDirectionChanging = false;

        drive_stop();
        return;
    }

    // 주행 중 반대 방향 요청
    if (!driveDirectionChanging && appliedDriveState != 0 && driveState != appliedDriveState)
    {
        appliedDriveState = 0;
        driveDirectionChanging = true;
        driveDirectionChangeTime = now;

        drive_stop();
        return;
    }

    if (driveDirectionChanging)
    {
        drive_stop();

        if (now - driveDirectionChangeTime < DIRECTION_CHANGE_DELAY)
            return;

        driveDirectionChanging = false;
    }

    if (driveState == 1)
        drive_forward(driveSpeed);
    else
        drive_backward(driveSpeed);

    appliedDriveState = driveState;
}

void update_steering()
{
    unsigned long now = millis();

    if (steeringState != 0 && now - lastSteeringCmdTime > STEERING_TIMEOUT)
        steeringState = 0;

    if (steeringState == 0 || steeringSpeed == 0)
    {
        appliedSteeringState = 0;
        steeringDirectionChanging = false;

        steering_stop();
        return;
    }

    // 조향 중 반대 방향 요청
    if (!steeringDirectionChanging && appliedSteeringState != 0 && steeringState != appliedSteeringState)
    {
        appliedSteeringState = 0;
        steeringDirectionChanging = true;
        steeringDirectionChangeTime = now;

        steering_stop();
        return;
    }

    if (steeringDirectionChanging)
    {
        steering_stop();

        if (now - steeringDirectionChangeTime < DIRECTION_CHANGE_DELAY)
            return;
            
        steeringDirectionChanging = false;
    }

    if (steeringState == -1)
        steering_left(steeringSpeed);
    else
        steering_right(steeringSpeed);

    appliedSteeringState = steeringState;
}

void setup()
{
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
}

void loop()
{
    read_serial_command();

    update_drive();
    update_steering();
}
