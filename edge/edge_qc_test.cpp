// Build and run:  g++ -O2 -std=c++11 edge_qc_test.cpp -o edge_qc_test && ./edge_qc_test
#include <cstdio>
#include <cstdlib>
#include "edge_qc.h"
using namespace trinetra;

static float rnd() { return (float)rand() / RAND_MAX - 0.5f; }

int main() {
  EdgeQC qc;
  srand(4);
  int flagged = 0, clean = 0, falseAlarm = 0;
  printf("sizeof(EdgeQC) = %u bytes\n", (unsigned)sizeof(EdgeQC));
  for (int m = 0; m < 400; m++) {
    float t = 30 + 3 * sinf(m / 60.f) + 0.06f * rnd(), td = 20.f;
    float p = 1000 + 0.6f * sinf(m / 90.f) + 0.04f * rnd();
    float rh = 100 * satVp(td) / satVp(t) + 0.4f * rnd();
    if (m >= 150) { float k = m < 162 ? (m - 150) / 12.f : expf(-(m - 162) / 25.f); t -= 6.f * k; rh = 100 * satVp(td + 1) / satVp(t); }   // real storm: T falls, RH rises
    Reading r{t, p, rh};
    const char* tag = "";
    if (m == 250) { r.t += 17.f; tag = "SPIKE (+17 C)"; }
    if (m == 251) tag = "SPIKE-RETURN";
    if (m >= 300 && m < 330) { static Reading frozen; if (m == 300) frozen = r; r = frozen; tag = "STUCK"; }
    if (m >= 350 && m < 355) { r.t = -99.9f; r.p = 0.f; r.rh = -99.9f; tag = "DROPOUT"; }
    uint16_t f = qc.update(r);
    bool truth = tag[0] != 0;
    if (EdgeQC::suspicious(f)) { flagged++; if (!truth) { falseAlarm++; printf("  false alarm at minute %d flags=0x%02x\n", m, f); } if (truth && (m == 250 || m == 315 || m == 352)) printf("minute %d  %-14s flags=0x%02x\n", m, tag, f); }
    else clean++;
  }
  printf("flagged %d minutes, false alarms %d (the real storm from minute 150 must stay clean)\n", flagged, falseAlarm);
  return falseAlarm ? 1 : 0;
}
