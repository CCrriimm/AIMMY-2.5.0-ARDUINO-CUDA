"""Realistic tracking scenarios: target speeds a player actually produces, with and without
measurement noise, at a high-magnification lever arm."""
import numpy as np

REF_DT = 16.0
MIN_GAIN, MAX_GAIN = 0.16, 1.0
GAIN_CUT = 0.12
GAIN_RECOVER_PER_S = 3.0
STABLE_TICKS = 6
EFFICIENT, NOISY = 0.50, 0.15
DEADZONE = 3.0
REV_MIN_STEP, REV_GAIN, REV_CAP = 2.5, 0.35, 4.0
MIN_DT_SCALE, MAX_DT_SCALE = 0.70, 2.00

DT = 5.0                                     # ms per update (200 updates/s)
DTS = min(max(DT / REF_DT, MIN_DT_SCALE), MAX_DT_SCALE)
K = 1.0 - 0.79                               # user sensitivity -> per-tick proportional gain


def governor_gains(errors, dt=DT):
    gain, prev = MAX_GAIN, None
    total_path = net = 0.0
    ps, has_prev, reversal = 0.0, False, 0.0
    priming, prime_ticks, stable = True, 0, 0
    out = []

    def recover():
        nonlocal gain
        if stable >= STABLE_TICKS:
            gain = min(MAX_GAIN, gain + GAIN_RECOVER_PER_S * dt / 1000.0)

    for e in errors:
        mag = abs(e)
        if priming or prev is None:
            prime_ticks += 1
            if prime_ticks >= 3:
                priming = False
            prev = e
            total_path = net = 0.0
            stable = STABLE_TICKS
            recover()
            out.append(gain)
            continue
        if mag < DEADZONE:
            total_path = net = 0.0
            prev = e
            has_prev = False
            reversal = max(0.0, reversal - 0.5)
            stable = STABLE_TICKS
            recover()
            out.append(gain)
            continue

        s = e - prev
        if has_prev:
            if s * ps < 0 and abs(s) > REV_MIN_STEP:
                reversal = min(REV_CAP, reversal + 1.0)
            else:
                reversal = max(0.0, reversal - 0.25)
        ps, has_prev = s, True

        if reversal >= 2.0:
            rate = min(max(dt / REF_DT, 0.25), 4.0)
            gain = max(MIN_GAIN, gain - REV_GAIN * rate)

        total_path += abs(s)
        net += s
        decay = np.exp(-dt / 250.0)
        total_path *= decay
        net *= decay
        prev = e

        if (total_path < mag * 2.0 or total_path < 12.0) and reversal < 2.0:
            stable = STABLE_TICKS
            recover()
            out.append(gain)
            continue

        eff = min(max(abs(net) / total_path, 0.0), 1.0)
        if eff >= EFFICIENT and reversal < 1.0:
            stable += 1
            recover()
        elif eff <= NOISY:
            stable = 0
            rate = min(max(dt / REF_DT, 0.25), 4.0)
            gain = max(MIN_GAIN, gain - GAIN_CUT * rate)
        else:
            stable = 0
            t = min(max((eff - NOISY) / (EFFICIENT - NOISY), 0.0), 1.0)
            target = MIN_GAIN + t * (MAX_GAIN - MIN_GAIN)
            gain += (target - gain) * 0.25
            gain = min(max(gain, MIN_GAIN), MAX_GAIN)
        out.append(gain)
    return np.array(out)


def play(targets, lever, gains):
    cam = 0.0
    errs = []
    for n in range(len(targets)):
        e = targets[n] - cam
        errs.append(e)
        u = K * gains[n] * e * DTS if abs(e) > DEADZONE else 0.0
        cam += u * lever * DTS
    return np.array(errs)


def scenario(lever, speed_px_per_s, noise, seed=5, ticks=1200):
    rng = np.random.default_rng(seed)
    # player strafes: target slides back and forth, plus detection noise
    t = np.array([speed_px_per_s * (DT / 1000.0) * n for n in range(ticks)])
    t = 220.0 * np.sin(t / 220.0)                       # smooth direction reversals
    if noise:
        t = t + rng.normal(0, noise, ticks)
    base = play(t, lever, np.ones(ticks))
    gains = governor_gains(base)
    on = play(t, lever, gains)
    sl = slice(300, None)
    return (np.abs(base[sl]).mean(), base[sl].max() - base[sl].min(), gains[sl].mean(),
            np.abs(on[sl]).mean(), on[sl].max() - on[sl].min())


print("4x-class scope (lever 9.3), 200 updates/s, realistic strafing speeds")
print()
print(" speed(px/s) | noise | gain 1.0 meanAbs   p2p   | governor meanAbs   p2p  meanGain")
for speed in (150, 300, 500, 800):
    for noise in (0.0, 2.0):
        a, b, g, c, d = scenario(9.3, speed, noise)
        print("%11d | %5.1f | %18.1f %6.1f | %17.1f %6.1f %8.3f" % (speed, noise, a, b, c, d, g))

print()
print("Same, but gain 1.0 loop is already unstable (lever 16):")
print()
print(" speed(px/s) | noise | gain 1.0 meanAbs   p2p   | governor meanAbs   p2p  meanGain")
for speed in (0, 150, 300, 500):
    for noise in (0.0, 2.0):
        a, b, g, c, d = scenario(16.0, speed, noise)
        print("%11d | %5.1f | %18.1f %6.1f | %17.1f %6.1f %8.3f" % (speed, noise, a, b, c, d, g))
