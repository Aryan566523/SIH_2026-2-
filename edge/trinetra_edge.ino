// ESP32-S3 + BME280 (temperature, pressure, humidity) example. Flags suspicious minutes before sending.
// Libraries: Adafruit BME280, PubSubClient (MQTT over GPRS modem or Wi-Fi).
#include <Wire.h>
#include <Adafruit_BME280.h>
#include "edge_qc.h"

Adafruit_BME280 bme;
trinetra::EdgeQC qc;
const char* STATION_ID = "AWS-DEL-014";

void setup() {
  Serial.begin(115200);
  bme.begin(0x76);
}

void loop() {
  trinetra::Reading r{bme.readTemperature(), bme.readPressure() / 100.0f, bme.readHumidity()};
  uint16_t flags = qc.update(r);
  // Publish the reading always, with the flag mask, so the server layer can run the physics, AI and neighbour checks.
  // Optional power saving: when flags == 0 and the value barely changed, send every 5th minute only.
  char msg[128];
  snprintf(msg, sizeof(msg), "{\"id\":\"%s\",\"t\":%.2f,\"p\":%.2f,\"rh\":%.1f,\"flags\":%u}", STATION_ID, r.t, r.p, r.rh, flags);
  Serial.println(msg);                 // replace with mqtt.publish("aws/obs", msg)
  esp_sleep_enable_timer_wakeup(60ULL * 1000000ULL);
  esp_light_sleep_start();             // wake once a minute; light sleep keeps the baseline in RAM
}
