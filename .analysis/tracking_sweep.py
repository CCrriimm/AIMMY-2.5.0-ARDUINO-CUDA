"""Tracking validation: sweep camera lever arm while the target moves continuously.
This is the case that actually oscillates, because the error never reaches the lock-on deadzone."""
import numpy as np
from closed_loop_validation import governor_gain_series, MIN_DT_SCALE, MAX_DT_SCALE, DEADZONE, REF_DT

SENS = 0.79
FPS = 200
DT = 1000.0 / FPS
DTS = min(max(DT / REF_DT, MIN_DT_SCALE), MAX_DT_SCALE)
K = 1.0 - SENS


def run(lever, use_governor, ticks=800, vel=1.6, jitter=0.0, seed=1):
    rng = np.random.default_rng(seed)
    targets = [vel * n for n in range(ticks)]
    if jitter:
        targets = [t + rng.normal(0, jitter) for t in targets]

    def play(gains):
        cam = 0.0
        errs = []
        for n in range(ticks):
            e = targets[n] - cam
            errs.append(e)
            u = K * gains[n] * e * DTS if abs(e) > DEADZONE else 0.0
            cam += u * lever * DTS
        return np.array(errs)

    base = play(np.ones(ticks))
    gains = governor_gain_series(base, DT) if use_governor else np.ones(ticks)
    errs = play(gains)
    return errs, gains, np.array(targets)


print("Moving target (1.6 px/frame = ~96 px/s sweep), 200 fps. Lag error and oscillation amplitude:")
print()
print(" lever | gain 1.0: meanErr  p2p   | governor: meanErr  p2p  meanGain")
for lever in (1, 2, 4, 6, 8, 9.3, 12, 16, 20):
    e_off, _, _ = run(lever, False)
    e_on, g_on, _ = run(lever, True)
    sl = slice(300, None)
    print("%6.1f | %15.1f %6.1f | %16.1f %5.1f %8.3f" % (
        lever, np.abs(e_off[sl]).mean(), e_off[sl].max() - e_off[sl].min(),
        np.abs(e_on[sl]).mean(), e_on[sl].max() - e_on[sl].min(), g_on[sl].mean()))

print()
print("Sensitivity sweep at lever 9.3 (4x scope), 200 fps, moving target:")
print()
print(" sens | gain 1.0: meanErr p2p | governor: meanErr p2p meanGain")
for sens in (0.5, 0.7, 0.79, 0.9, 0.95, 0.99):
    K2 = 1.0 - sens
    ticks = 800
    targets = [1.6 * n for n in range(ticks)]
    lever = 9.3

    def play(gains):
        cam = 0.0
        errs = []
        for n in range(ticks):
            e = targets[n] - cam
            errs.append(e)
            u = K2 * gains[n] * e * DTS if abs(e) > DEADZONE else 0.0
            cam += u * lever * DTS
        return np.array(errs)

    base = play(np.ones(ticks))
    gains = governor_gain_series(base, DT)
    e_on = play(gains)
    sl = slice(300, None)
    print("%5.2f | %15.1f %6.1f | %16.1f %5.1f %8.3f" % (
        sens, np.abs(base[sl]).mean(), base[sl].max() - base[sl].min(),
        np.abs(e_on[sl]).mean(), e_on[sl].max() - e_on[sl].min(), gains[sl].mean()))
