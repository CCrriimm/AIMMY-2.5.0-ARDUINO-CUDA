"""End-to-end validation of the shipped fix:

  movement = (1 - sens) * governorGain * error * dtScale,  then capped to <= 0.85 * |error|
  camera  += movement * lever * dtScale

Compared against the original code (governorGain = 1, no cap) across lever arms and update
rates. 'lever' stands in for the game's per-optic sensitivity: ~4 at 2x, ~9-12 at 4x in Apex,
higher still for 6x/8x.
"""
import numpy as np

REF_DT = 16.0
MIN_GAIN, MAX_GAIN = 0.16, 1.0
GAIN_CUT, GAIN_RECOVER_PER_S, STABLE_TICKS = 0.12, 3.0, 6
EFFICIENT, NOISY, DEADZONE = 0.50, 0.15, 3.0
REV_MIN_STEP, REV_GAIN, REV_CAP = 2.5, 0.35, 4.0
MIN_DT_SCALE, MAX_DT_SCALE = 0.70, 2.00
MAX_STEP_RATIO, MIN_STEP_CAP = 0.85, 6.0
SENS = 0.79


class Governor:
    def __init__(self):
        self.gain = MAX_GAIN
        self.prev = None
        self.total_path = self.net = 0.0
        self.ps = 0.0
        self.has_prev = False
        self.rev = 0.0
        self.priming = True
        self.prime = 0
        self.stable = 0

    def evaluate(self, e, dt):
        mag = abs(e)

        def recover():
            if self.stable >= STABLE_TICKS:
                self.gain = min(MAX_GAIN, self.gain + GAIN_RECOVER_PER_S * dt / 1000.0)

        if self.priming or self.prev is None:
            self.prime += 1
            if self.prime >= 3:
                self.priming = False
            self.prev = e
            self.total_path = self.net = 0.0
            self.stable = STABLE_TICKS
            recover()
            return
        if mag < DEADZONE:
            self.total_path = self.net = 0.0
            self.prev = e
            self.has_prev = False
            self.rev = max(0.0, self.rev - 0.5)
            self.stable = STABLE_TICKS
            recover()
            return

        s = e - self.prev
        if self.has_prev:
            if s * self.ps < 0 and abs(s) > REV_MIN_STEP:
                self.rev = min(REV_CAP, self.rev + 1.0)
            else:
                self.rev = max(0.0, self.rev - 0.25)
        self.ps, self.has_prev = s, True

        if self.rev >= 2.0:
            rate = min(max(dt / REF_DT, 0.25), 4.0)
            self.gain = max(MIN_GAIN, self.gain - REV_GAIN * rate)

        self.total_path += abs(s)
        self.net += s
        decay = np.exp(-dt / 250.0)
        self.total_path *= decay
        self.net *= decay
        self.prev = e

        if (self.total_path < mag * 2.0 or self.total_path < 12.0) and self.rev < 2.0:
            self.stable = STABLE_TICKS
            recover()
            return

        eff = min(max(abs(self.net) / self.total_path, 0.0), 1.0)
        if eff >= EFFICIENT and self.rev < 1.0:
            self.stable += 1
            recover()
        elif eff <= NOISY:
            self.stable = 0
            rate = min(max(dt / REF_DT, 0.25), 4.0)
            self.gain = max(MIN_GAIN, self.gain - GAIN_CUT * rate)
        else:
            self.stable = 0
            t = min(max((eff - NOISY) / (EFFICIENT - NOISY), 0.0), 1.0)
            target = MIN_GAIN + t * (MAX_GAIN - MIN_GAIN)
            self.gain += (target - self.gain) * 0.25
            self.gain = min(max(self.gain, MIN_GAIN), MAX_GAIN)


def run(lever, fps, mode, ticks=1200, speed=300.0, noise=0.0, seed=7):
    """mode: 'original' (gain 1, no cap) or 'fixed' (governor + cap)."""
    dt = 1000.0 / fps
    dts = min(max(dt / REF_DT, MIN_DT_SCALE), MAX_DT_SCALE)
    k = 1.0 - SENS
    rng = np.random.default_rng(seed)

    # player strafing target: a smooth triangle wave (constant speed with rounded turns),
    # amplitude chosen so the peak speed equals 'speed' px/s
    period = 240                                    # updates per half sweep
    peak_speed = speed
    amp = peak_speed * period * (dt / 1000.0) / 2.0
    t = np.zeros(ticks)
    v = peak_speed
    for n in range(1, ticks):
        t[n] = t[n - 1] + v * (dt / 1000.0)
        if n % period == 0:
            v = -v
    # smooth the corners so there is no instantaneous velocity change
    kern = np.hanning(11)
    kern /= kern.sum()
    t = np.convolve(t, kern, mode="same")
    t = t - t.mean()
    if noise:
        t = t + rng.normal(0, noise, ticks)

    gov = Governor()
    cam = 0.0
    errs = []
    gains = []
    for n in range(ticks):
        e = t[n] - cam
        errs.append(e)
        if mode == "fixed":
            gov.evaluate(e, dt)
            g = gov.gain
            gains.append(g)
        else:
            g = 1.0
            gains.append(1.0)
        if abs(e) <= DEADZONE:
            u = 0.0
        else:
            u = k * g * e * dts
            if mode == "fixed":
                cap = max(MIN_STEP_CAP, abs(e) * MAX_STEP_RATIO)
                u = np.clip(u, -cap, cap)
        cam += u * lever * dts
    errs = np.array(errs)
    sl = slice(300, None)
    return (np.abs(errs[sl]).mean(), errs[sl].max() - errs[sl].min(), np.array(gains)[sl].mean())


print("Tracking a strafing target (300 px/s), sensitivity 0.79, comparing the original code")
print("against the fix. 'p2p' is peak-to-peak crosshair displacement per update: a large value")
print("at a low mean error IS the shake (crosshair running back and forth over the target).")
print()
print(" lever | fps | original: meanErr   p2p    | fixed: meanErr   p2p  meanGain")
for lever in (2.0, 4.0, 6.0, 9.3, 12.0, 16.0, 24.0, 40.0):
    for fps in (60, 200):
        a, b, _ = run(lever, fps, "original")
        c, d, g = run(lever, fps, "fixed")
        flag = "  <-- diverged" if b > 400 else ""
        print("%6.1f | %3d | %14.1f %7.1f | %13.1f %6.1f %8.3f%s" % (lever, fps, a, b, c, d, g, flag))

print()
print("Detection noise 3 px added (models YOLO box jitter), lever 9.3 (4x-class):")
print(" lever | fps | original: meanErr   p2p    | fixed: meanErr   p2p  meanGain")
for lever in (4.0, 9.3, 16.0):
    for fps in (60, 200):
        a, b, _ = run(lever, fps, "original", noise=3.0)
        c, d, g = run(lever, fps, "fixed", noise=3.0)
        print("%6.1f | %3d | %14.1f %7.1f | %13.1f %6.1f %8.3f" % (lever, fps, a, b, c, d, g))

print()
print("Responsiveness: time to close a 200 px flick, lever 4 (2x), 200 updates/s")
for label, mode in (("original", "original"), ("fixed", "fixed")):
    dt = 5.0
    dts = min(max(dt / REF_DT, MIN_DT_SCALE), MAX_DT_SCALE)
    gov = Governor()
    cam = 0.0
    ticks = 0
    while ticks < 2000:
        e = 200.0 - cam
        if mode == "fixed":
            gov.evaluate(e, dt)
            g = gov.gain
        else:
            g = 1.0
        if abs(e) <= DEADZONE:
            break
        u = (1 - SENS) * g * e * dts
        if mode == "fixed":
            cap = max(MIN_STEP_CAP, abs(e) * MAX_STEP_RATIO)
            u = np.clip(u, -cap, cap)
        cam += u * 4.0 * dts
        ticks += 1
    print("  %-9s closed 200 px in %4d updates (%.0f ms)" % (label, ticks, ticks * dt))
