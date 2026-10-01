// TRINETRA edge layer for ESP32 / any C++11 MCU.  No heap, no STL, about 200 bytes of state.
// Runs the station-side checks (range, step, stuck, dew-point physics, adaptive z-score) on every reading and
// tells the caller whether to flag the packet.  Heavy AI and the neighbour check run on the server.
#pragma once
#include <math.h>
#include <stdint.h>

namespace trinetra {

enum Flag : uint16_t {
  F_NONE = 0, F_MISSING = 1, F_RANGE = 2, F_STEP = 4, F_STUCK = 8, F_DEWPOINT = 16, F_ZSCORE = 32,
};

struct Reading { float t, p, rh; };          // degC, hPa, %

inline float satVp(float t) { return 6.112f * expf(17.67f * t / (t + 243.5f)); }
inline float dewPoint(float t, float rh) {
  float e = fmaxf(rh, 0.5f) / 100.0f * satVp(t), g = logf(e / 6.112f);
  return 243.5f * g / (17.67f - g);
}

class EdgeQC {
 public:
  // Limits follow WMO No. 8 / Zahumensky (2004). Tune for your climate.
  float lo[3] = {-15.f, 500.f, 0.f}, hi[3] = {50.f, 1080.f, 100.f};
  float stepMax[3] = {3.f, 1.5f, 20.f};        // largest believable 1-minute change
  float noise[3] = {0.08f, 0.05f, 0.5f};       // sensor noise floor (1 sigma)
  uint8_t stuckMin[3] = {15, 15, 20};          // identical readings in a row
  float zLimit = 8.f;                          // adaptive z-score threshold
  float dewLimit = 6.f;                        // degC dew-point change from its running mean

  // returns a bit mask of Flag values (0 = clean). Bad readings never update the learned baseline.
  uint16_t update(const Reading& r) {
    const float x[3] = {r.t, r.p, r.rh};
    uint16_t f = F_NONE;
    for (int i = 0; i < 3; i++) {
      if (!isfinite(x[i]) || x[i] <= -90.f) { f |= F_MISSING; continue; }
      if (x[i] < lo[i] || x[i] > hi[i]) f |= F_RANGE;
      if (n_ > 0 && fabsf(x[i] - last_[i]) > stepMax[i]) f |= F_STEP;
      if (n_ > 0 && x[i] == last_[i] && !(i == 2 && x[i] >= 98.f)) { if (run_[i] < 255) run_[i]++; } else run_[i] = 0;
      if (run_[i] + 1 >= stuckMin[i]) f |= F_STUCK;
      if (n_ >= warmup_) {                      // z-score of the innovation against an EWMA baseline
        float z = fabsf(x[i] - mu_[i]) / sqrtf(var_[i] + noise[i] * noise[i]);
        if (z > zLimit) f |= F_ZSCORE;
      }
    }
    if (!(f & (F_MISSING | F_RANGE))) {
      float td = dewPoint(r.t, r.rh);
      if (n_ >= warmup_ && fabsf(td - tdMu_) > dewLimit) f |= F_DEWPOINT;
      if (!(f & ~F_NONE)) {                     // learn only from clean readings
        for (int i = 0; i < 3; i++) {
          if (n_ == 0) { mu_[i] = x[i]; var_[i] = 0.f; }
          float d = x[i] - mu_[i];
          mu_[i] += alpha_ * d;
          var_[i] = (1 - alpha_) * (var_[i] + alpha_ * d * d);
        }
        tdMu_ = (n_ == 0) ? td : tdMu_ + alpha_ * (td - tdMu_);
        if (n_ < 65535) n_++;
      }
    }
    if (!(f & F_MISSING)) { last_[0] = r.t; last_[1] = r.p; last_[2] = r.rh; if (n_ == 0) n_ = 0; }
    return f;
  }
  static bool suspicious(uint16_t f) { return (f & (F_MISSING | F_RANGE | F_STEP | F_STUCK | F_DEWPOINT | F_ZSCORE)) != 0; }

 private:
  float mu_[3] = {0, 0, 0}, var_[3] = {0, 0, 0}, last_[3] = {0, 0, 0}, tdMu_ = 0;
  uint8_t run_[3] = {0, 0, 0};
  uint16_t n_ = 0, warmup_ = 30;
  float alpha_ = 0.03f;                         // about a 30-minute memory at 1-minute sampling
};

}  // namespace trinetra
