"""Sweep the camera lever arm (= scope magnification in screen px per mouse count) to find where
the loop actually becomes unstable with the reported settings, and check that the governor
suppresses exactly that case while leaving every stable case untouched."""
import numpy as np
from closed_loop_validation import governor_gain_series, MIN_DT_SCALE, MAX_DT_SCALE, DEADZONE, REF_DT

SENS = 0.79
FPS = 200
DT = 1000.0 / FPS
DTS = min(max(DT / REF_DT, MIN_DT_SCALE), MAX_DT_SCALE)
TICKS = 400


def run(lever, use_governor, ticks=TICKS, target=250.0):
    k = 1.0 - SENS
    cam = 0.0
    errors = []
    for n in range(ticks):
        e = target - cam
        errors.append(e)
        u = k * e * DTS if abs(e) > DEADZONE else 0.0
        cam += u * lever * DTS
    errors = np.array(errors)
    gains = governor_gain_series(errors, DT) if use_governor else np.ones(ticks)

    cam = 0.0
    out = []
    for n in range(ticks):
        e = target - cam
        out.append(e)
        u = k * gains[n] * e * DTS if abs(e) > DEADZONE else 0.0
        cam += u * lever * DTS
    return np.array(out), gains


print("Loop gain per tick = (1 - sens) * lever = %.2f * lever" % (1 - SENS))
print()
print(" lever | perTickGain | gain 1.0: p2p(px) finalErr | governor: p2p(px) finalErr meanGain")
for lever in (1, 2, 4, 6, 8, 9.3, 10, 12, 16):
    e_off, _ = run(lever, False)
    e_on, g_on = run(lever, True)
    p2p_off = e_off[200:].max() - e_off[200:].min()
    p2p_on = e_on[200:].max() - e_on[200:].min()
    print("%6.1f | %11.2f | %18.1f %9.1f | %17.1f %9.1f %8.3f" % (
        lever, (1 - SENS) * lever, p2p_off, abs(e_off[-1]), p2p_on, abs(e_on[-1]), g_on[200:].mean()))

print()
print("Worst case: lever 12 (heavily magnified), 200 fps - error trajectory sample")
e_off, _ = run(12, False, ticks=60)
e_on, g_on = run(12, True, ticks=60)
print("  gain 1.0  errors:", " ".join("%8.1f" % v for v in e_off[:24]))
print("  governor  errors:", " ".join("%8.1f" % v for v in e_on[:24]))
print("  governor  gain  :", " ".join("%8.3f" % v for v in g_on[:24]))
