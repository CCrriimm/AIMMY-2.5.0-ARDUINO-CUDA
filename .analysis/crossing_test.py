"""As the target crosses the crosshair, the measurement keeps its sign while the camera sweeps
through - exactly the condition that produced the 24 Hz oscillation in the recorded clip."""
import numpy as np
from closed_loop_validation import governor_gain_series, MIN_DT_SCALE, MAX_DT_SCALE, DEADZONE, REF_DT

K = 1.0 - 0.79          # 0.21
DTS = min(max(80.0 / REF_DT, MIN_DT_SCALE), MAX_DT_SCALE)


def make_targets(ticks, amp, period, noise=0.0, seed=3):
    rng = np.random.default_rng(seed)
    t = amp * np.sin(2 * np.pi * np.arange(ticks) / period)
    if noise:
        t = t + rng.normal(0, noise, ticks)
    return t


def play(targets, lever, gains, fps_dt=80.0):
    cam = 0.0
    errs = []
    for n in range(len(targets)):
        e = targets[n] - cam
        errs.append(e)
        u = K * gains[n] * e * DTS if abs(e) > DEADZONE else 0.0
        cam += u * lever * DTS
    return np.array(errs)


print("Sinusoidal target (90 px amplitude, 30-tick period), dt=5ms (200 fps)")
print()
print(" lever | noise | gain 1.0: meanAbs  p2p  | governor: meanAbs  p2p  meanGain")
for lever in (4, 9.3, 12, 16, 24):
    for noise in (0.0, 3.0):
        targets = make_targets(800, 90.0, 30.0, noise=noise)
        base = play(targets, lever, np.ones(800))
        gains = governor_gain_series(base, 5.0)
        on = play(targets, lever, gains)
        sl = slice(200, None)
        print("%6.1f | %5.1f | %19.1f %5.1f | %19.1f %5.1f %8.3f" % (
            lever, noise, np.abs(base[sl]).mean(), base[sl].max() - base[sl].min(),
            np.abs(on[sl]).mean(), on[sl].max() - on[sl].min(), gains[sl].mean()))

print()
print("Deliberate slow strafe (target moves steadily through the crosshair), 4x scope, lever 9.3")
ticks = 800
targets = np.array([-300.0 + 1.2 * n for n in range(ticks)])   # constant-speed crossing
for label, use in (("gain 1.0", False), ("governor", True)):
    base = play(targets, 9.3, np.ones(ticks))
    gains = governor_gain_series(base, 5.0) if use else np.ones(ticks)
    on = play(targets, 9.3, gains)
    sl = slice(200, None)
    print("  %-9s meanErr=%6.2f  p2p=%6.2f  meanGain=%.3f" % (
        label, np.abs(on[sl]).mean(), on[sl].max() - on[sl].min(), gains[sl].mean()))
