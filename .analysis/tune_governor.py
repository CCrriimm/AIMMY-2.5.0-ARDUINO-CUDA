"""Tune the reversal-detector thresholds so it does not fire on detection-box noise, while still
catching a genuinely runaway loop. Two scenarios:

  A) 'noisy but stable'  - the loop converges, error is dominated by a few px of box jitter
  B) 'runaway'           - lever*magnification makes the loop gain > 2, error explodes

The governor must stay near gain 1.0 in A and cut hard in B.
"""
import numpy as np

REF_DT = 16.0
MIN_GAIN, MAX_GAIN = 0.16, 1.0
GAIN_CUT, GAIN_RECOVER_PER_S, STABLE_TICKS = 0.12, 3.0, 6
EFFICIENT, NOISY, DEADZONE = 0.50, 0.15, 3.0
MIN_DT_SCALE, MAX_DT_SCALE = 0.70, 2.00
MAX_STEP_RATIO, MIN_STEP_CAP = 0.85, 6.0
SENS = 0.79


class Governor:
    def __init__(self, rev_min, rev_min_ratio, rev_gain, min_path, trigger=2.0):
        self.gain = MAX_GAIN
        self.prev = None
        self.total_path = self.net = 0.0
        self.ps = 0.0
        self.has_prev = False
        self.rev = 0.0
        self.priming, self.prime, self.stable = True, 0, 0
        self.rev_min = rev_min
        self.rev_min_ratio = rev_min_ratio
        self.rev_gain = rev_gain
        self.min_path = min_path
        self.trigger = trigger

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
            threshold = max(self.rev_min, self.rev_min_ratio * mag)
            if s * self.ps < 0 and abs(s) > threshold:
                self.rev = min(4.0, self.rev + 1.0)
            else:
                self.rev = max(0.0, self.rev - 0.25)
        self.ps, self.has_prev = s, True

        if self.rev >= self.trigger:
            rate = min(max(dt / REF_DT, 0.25), 4.0)
            self.gain = max(MIN_GAIN, self.gain - self.rev_gain * rate)

        self.total_path += abs(s)
        self.net += s
        decay = np.exp(-dt / 250.0)
        self.total_path *= decay
        self.net *= decay
        self.prev = e

        if (self.total_path < mag * 2.0 or self.total_path < self.min_path) and self.rev < 2.0:
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


def run(lever, fps, params, noise=0.0, speed=300.0, ticks=1200, seed=11, quiet=False):
    dt = 1000.0 / fps
    dts = min(max(dt / REF_DT, MIN_DT_SCALE), MAX_DT_SCALE)
    k = 1.0 - SENS
    rng = np.random.default_rng(seed)
    period = 240
    amp = speed * period * (dt / 1000.0) / 2.0
    t = np.zeros(ticks)
    v = speed
    for n in range(1, ticks):
        t[n] = t[n - 1] + v * (dt / 1000.0)
        if n % period == 0:
            v = -v
    kern = np.hanning(11)
    kern /= kern.sum()
    t = np.convolve(t, kern, mode="same")
    t -= t.mean()
    if noise:
        t = t + rng.normal(0, noise, ticks)

    gov = Governor(*params) if params else None
    cam = 0.0
    errs, gains = [], []
    diverged = False
    for n in range(ticks):
        e = t[n] - cam
        errs.append(e)
        g = 1.0
        if gov:
            gov.evaluate(e, dt)
            g = gov.gain
        gains.append(g)
        if abs(e) <= DEADZONE:
            u = 0.0
        else:
            u = k * g * e * dts
            if gov:
                cap = max(MIN_STEP_CAP, abs(e) * MAX_STEP_RATIO)
                u = np.clip(u, -cap, cap)
        cam += u * lever * dts
        if not np.isfinite(cam) or abs(cam) > 1e6:
            diverged = True
            errs.extend([float("nan")] * (ticks - len(errs)))
            gains.extend([g] * (ticks - len(gains)))
            break
    errs = np.array(errs, dtype=float)
    gains = np.array(gains, dtype=float)
    sl = slice(300, None)
    return np.nanmean(np.abs(errs[sl])), np.nanmean(gains[sl]), diverged


CANDIDATES = {
    "current:   2.5px / 0.00 ratio / trigger 2.0": (2.5, 0.00, 0.35, 12.0, 2.0),
    "tuned A:   3.0px / 0.25 ratio / trigger 2.0": (3.0, 0.25, 0.35, 12.0, 2.0),
    "tuned B:   3.0px / 0.35 ratio / trigger 3.0": (3.0, 0.35, 0.35, 20.0, 3.0),
    "tuned C:   4.0px / 0.35 ratio / trigger 3.0": (4.0, 0.35, 0.35, 20.0, 3.0),
}

print("Scenario A - noisy but STABLE loop (lever 4, noise 3 px): governor must stay near 1.0")
print()
print(" params                                        | fps | meanErr  meanGain")
for name, p in CANDIDATES.items():
    for fps in (60, 200):
        e, g, div = run(4.0, fps, p, noise=3.0)
        print(" %-45s | %3d | %7.1f %9.3f" % (name, fps, e, g))

print()
print("Scenario B - RUNAWAY loop (lever 24 @60 fps, lever 16 @200 fps, noise 2 px):")
print(" governor must cut the gain hard; original code diverges to NaN")
print()
print(" params                                        | lever | fps | meanErr  meanGain  diverged")
for name, p in CANDIDATES.items():
    for lever, fps in ((24.0, 60), (16.0, 200)):
        e, g, div = run(lever, fps, p, noise=2.0)
        print(" %-45s | %5.1f | %3d | %7.1f %9.3f  %s" % (name, lever, fps, e, g, div))

print()
print("Baseline - original code (gain 1.0, no cap):")
for lever, fps in ((4.0, 60), (4.0, 200), (24.0, 60), (16.0, 200)):
    e, g, div = run(lever, fps, None, noise=2.0)
    print("   lever %5.1f fps %3d -> meanErr %8.1f  diverged=%s" % (lever, fps, e, div))
