/*
  ParkiScan IMU node — ESP32 + MPU-6050 (GY-521), USB serial JSON lines at 100 Hz.

  Wiring (ESP32 DevKit):  MPU-6050 VCC -> 3V3, GND -> GND, SDA -> GPIO21, SCL -> GPIO22, AD0 -> GND (addr 0x68)
  Arduino IDE: Board "ESP32 Dev Module" (esp32 core >= 2.0), Serial Monitor at 921600 baud. No extra libraries required.
  Output (one line per sample, matches parkiscan/sensors/imu.py):
    {"v":1,"dev":"esp32-wrist-1a2b3c","seq":42,"t_ms":12345,"ax":0.012,"ay":-0.981,"az":0.105,
     "gx":1.25,"gy":-0.40,"gz":0.31,"placement":"wrist","sim":false}
  Units: g and deg/s. Ranges: ±4 g, ±500 deg/s, DLPF 44 Hz. Forward to the service with
    python -m parkiscan.sensors.serial_bridge --port /dev/cu.usbserial-XXXX --session <session_id>
  Serial commands: 'p' pause/resume, 'c' re-run gyro bias calibration (keep the sensor still ~2 s).
  This is a research data logger, not a medical device.
*/
#include <Wire.h>

#define PLACEMENT "wrist"          // wrist | ankle | waist — set before flashing
#define SAMPLE_HZ 100
#define SDA_PIN 21
#define SCL_PIN 22
static const uint8_t MPU = 0x68;
static const float ACC_LSB = 8192.0f;   // ±4 g
static const float GYR_LSB = 65.5f;     // ±500 dps

uint32_t seq = 0;
bool paused = false;
float gbx = 0, gby = 0, gbz = 0;
char devId[32];

bool writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU); Wire.write(reg); Wire.write(val);
  return Wire.endTransmission() == 0;
}

bool readRaw(int16_t* a, int16_t* g) {
  Wire.beginTransmission(MPU); Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)MPU, 14) != 14) return false;
  for (int i = 0; i < 3; i++) a[i] = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read();  // temperature (unused)
  for (int i = 0; i < 3; i++) g[i] = (Wire.read() << 8) | Wire.read();
  return true;
}

bool initMpu() {
  if (!writeReg(0x6B, 0x00)) return false;   // wake, internal clock
  delay(50);
  writeReg(0x6B, 0x01);                      // PLL with X gyro reference (more stable)
  writeReg(0x1A, 0x03);                      // DLPF ~44 Hz
  writeReg(0x19, 0x09);                      // sample rate divider -> 100 Hz internal
  writeReg(0x1B, 0x08);                      // gyro ±500 dps
  writeReg(0x1C, 0x08);                      // accel ±4 g
  return true;
}

void calibrateGyro() {
  const int N = 200;
  double sx = 0, sy = 0, sz = 0; int n = 0;
  int16_t a[3], g[3];
  for (int i = 0; i < N; i++) {
    if (readRaw(a, g)) { sx += g[0]; sy += g[1]; sz += g[2]; n++; }
    delay(10);
  }
  if (n > 0) { gbx = sx / n / GYR_LSB; gby = sy / n / GYR_LSB; gbz = sz / n / GYR_LSB; }
  Serial.printf("# gyro bias dps: %.3f %.3f %.3f (n=%d)\n", gbx, gby, gbz, n);
}

void setup() {
  Serial.begin(921600);   // ~170 B x 100 Hz = 17 kB/s needs more than 115200 baud
  delay(300);
  Wire.begin(SDA_PIN, SCL_PIN, 400000);
  uint64_t mac = ESP.getEfuseMac();
  snprintf(devId, sizeof(devId), "esp32-%s-%06x", PLACEMENT, (uint32_t)(mac >> 24) & 0xFFFFFF);
  while (!initMpu()) {
    Serial.println("# MPU-6050 not found on I2C 0x68 — check wiring; retrying in 1 s");
    delay(1000);
  }
  Serial.printf("# ParkiScan IMU %s ready, %d Hz, placement=%s\n", devId, SAMPLE_HZ, PLACEMENT);
  calibrateGyro();
}

void loop() {
  static uint32_t next = micros();
  const uint32_t period = 1000000UL / SAMPLE_HZ;

  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'p') { paused = !paused; Serial.printf("# %s\n", paused ? "paused" : "streaming"); }
    if (c == 'c') calibrateGyro();
  }
  if ((int32_t)(micros() - next) < 0) return;
  next += period;
  if (paused) return;

  int16_t a[3], g[3];
  if (!readRaw(a, g)) {                       // bus glitch: re-init, keep seq so gaps are visible
    initMpu();
    seq++;
    return;
  }
  Serial.printf("{\"v\":1,\"dev\":\"%s\",\"seq\":%lu,\"t_ms\":%lu,\"ax\":%.4f,\"ay\":%.4f,\"az\":%.4f,"
                "\"gx\":%.3f,\"gy\":%.3f,\"gz\":%.3f,\"placement\":\"%s\",\"sim\":false}\n",
                devId, (unsigned long)seq++, (unsigned long)millis(),
                a[0] / ACC_LSB, a[1] / ACC_LSB, a[2] / ACC_LSB,
                g[0] / GYR_LSB - gbx, g[1] / GYR_LSB - gby, g[2] / GYR_LSB - gbz, PLACEMENT);
}
