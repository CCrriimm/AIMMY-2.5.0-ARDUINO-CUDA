"""Closed-loop validation of the AimLoopController gain governor.

Sign convention
---------------
e_n  = target - camera          measurement the detector produces (screen px, positive = target
                                is to the right of the crosshair)
u_n  = +(1-sens) * gain * e_n   mouse counts the AI sends to move the crosshair right
camera moves right by u_n * CameraLever  ->  e_{n+1} = e_n - CameraLever * u_n

so the per-tick loop gain is  (1-sens) * CameraLever, and CameraLever grows with the scope
magnification - that is why 4x is unstable where 2x is not. The governor only ever sees e_n.
"""
import numpy as np

REF_DT = 16.0
MIN_GAIN, MAX_GAIN = 0.16, 1.0
GAIN_CUT = 0.12
GAIN_RECOVER_PER_S = 3.0
STABLE_TICKS_BEFORE_RECOVERY = 6
EFFICIENT, NOISY = 0.50, 0.15
DEADZONE = 3.0
REV_MIN_STEP, REV_GAIN, REV_CAP = 2.5, 0.35, 4.0
MIN_DT_SCALE, MAX_DT_SCALE = 0.70, 2.00


class Governor:
    def __init__(self):
        self.gain = MAX_GAIN
        self.prev = None
        self.total_path = self.net_err = 0.0
        self.ps = 0.0
        self.has_prev_step = False
        self.reversal = 0.0
        self.priming = True
        self.prime_ticks = 0
        self.stable_ticks = 0

    def recover(self, dt):
        if self.stable_ticks >= STABLE_TICKS_BEFORE_RECOVERY:
            self.gain = min(MAX_GAIN, self.gain + GAIN_RECOVER_PER_S * dt / 1000.0)

    def step_scale(self, dt):
        return min(max(dt / REF_DT, MIN_DT_SCALE), MAX_DT_SCALE)

    def evaluate(self, e, dt):
        # only used for diagnostics: what the real controller would do
        pass


def governor_gain_series(errors, dt):
    """Run the exact governor logic over a recorded error series; return the gain per tick."""
    gain = MAX_GAIN
    prev = None
    total_path = net_err = 0.0
    ps = 0.0
    has_prev_step = False
    reversal = 0.0
    priming = True
    prime_ticks = 0
    stable_ticks = 0
    out = []

    def recover():
        nonlocal gain
        if stable_ticks >= STABLE_TICKS_BEFORE_RECOVERY:
            gain = min(MAX_GAIN, gain + GAIN_RECOVER_PER_S * dt / 1000.0)

    for e in errors:
        mag = abs(e)
        if priming or prev is None:
            prime_ticks += 1
            if prime_ticks >= 3:
                priming = False
            prev = e
            total_path = net_err = 0.0
            stable_ticks = STABLE_TICKS_BEFORE_RECOVERY
            recover()
            out.append(gain)
            continue
        if mag < DEADZONE:
            total_path = net_err = 0.0
            prev = e
            has_prev_step = False
            reversal = max(0.0, reversal - 0.5)
            stable_ticks = STABLE_TICKS_BEFORE_RECOVERY
            recover()
            out.append(gain)
            continue

        s = e - prev
        if has_prev_step:
            if s * ps < 0 and abs(s) > REV_MIN_STEP:
                reversal = min(REV_CAP, reversal + 1.0)
            else:
                reversal = max(0.0, reversal - 0.25)
        ps = s
        has_prev_step = True

        if reversal >= 2.0:
            rate = min(max(dt / REF_DT, 0.25), 4.0)
            gain = max(MIN_GAIN, gain - REV_GAIN * rate)

        total_path += abs(s)
        net_err += s
        decay = np.exp(-dt / 250.0)
        total_path *= decay
        net_err *= decay
        prev = e

        if (total_path < mag * 2.0 or total_path < 12.0) and reversal < 2.0:
            stable_ticks = STABLE_TICKS_BEFORE_RECOVERY
            recover()
            out.append(gain)
            continue

        eff = min(max(abs(net_err) / total_path, 0.0), 1.0)
        if eff >= EFFICIENT and reversal < 1.0:
            stable_ticks += 1
            recover()
        elif eff <= NOISY:
            stable_ticks = 0
            rate = min(max(dt / REF_DT, 0.25), 4.0)
            gain = max(MIN_GAIN, gain - GAIN_CUT * rate)
        else:
            stable_ticks = 0
            t = min(max((eff - NOISY) / (EFFICIENT - NOISY), 0.0), 1.0)
            target = MIN_GAIN + t * (MAX_GAIN - MIN_GAIN)
            gain += (target - gain) * 0.25
            gain = min(max(gain, MIN_GAIN), MAX_GAIN)
        out.append(gain)
    return np.array(out)


def simulate(zoom, sens, fps, use_governor, ticks=600, target=250.0, sweep=None):
    lever = zoom
    k = 1.0 - sens
    dt = 1000.0 / fps
    dts = min(max(dt / REF_DT, MIN_DT_SCALE), MAX_DT_SCALE)
    cam = 0.0
    errors = []
    gains = []
    # first pass: generate the error series using gain 1.0 (what the loop looks like)
    for n in range(ticks):
        tgt = target if sweep is None else sweep(n)
        e = tgt - cam
        errors.append(e)
        g = 1.0
        u = k * g * e * dts if abs(e) > DEADZONE else 0.0
        cam += u * lever * dts
    errors = np.array(errors)

    if use_governor:
        gains = governor_gain_series(errors, dt)
    else:
        gains = np.ones(ticks)

    # second pass: replay with the governor applied
    cam = 0.0
    err2 = []
    for n in range(ticks):
        tgt = target if sweep is None else sweep(n)
        e = tgt - cam
        err2.append(e)
        u = k * gains[n] * e * dts if abs(e) > DEADZONE else 0.0
        cam += u * lever * dts
    return np.array(err2), gains


print("sens=0.79 (reported setting), static target 250 px away.")
print("peak-to-peak crosshair swing over the last 200 ticks (small = settled on target):")
print()
print(" zoom | fps | gain 1.0 (current code) | with governor")
print("      |     | p2p(px)   finalErr     | p2p(px)   finalErr  meanGain")
for zoom in (1, 2, 4, 6, 8):
    for fps in (60, 200, 400):
        e_off, g_off = simulate(zoom, 0.79, fps, False)
        e_on, g_on = simulate(zoom, 0.79, fps, True)
        p2p_off = e_off[400:].max() - e_off[400:].min()
        p2p_on = e_on[400:].max() - e_on[400:].min()
        print("%5d | %3d | %7.1f %12.1f | %7.1f %11.1f  %6.3f" % (
            zoom, fps, p2p_off, abs(e_off[-1]), p2p_on, abs(e_on[-1]), g_on[400:].mean()))

print()
print("Tracking a target sweeping +-150 px (worst error after settling), 4x zoom, 200 fps:")
sweep = lambda n: 150.0 * np.sin(2 * np.pi * n / 120.0)
for label, use in (("governor ON", True), ("gain 1.0", False)):
    e, g = simulate(4, 0.79, 200, use, ticks=900, sweep=sweep)
    print("  %-12s worst error = %6.1f px   meanGain = %.3f" % (label, np.abs(e[300:]).max(), g[300:].mean()))

print()
print("Slow deliberate aim move at 4x / 200 fps (target sweeps +-150 px over 2 s), show settle:")
sweep2 = lambda n: 150.0 * np.sin(2 * np.pi * n / 240.0)
e, g = simulate(4, 0.79, 200, True, ticks=900, sweep=sweep2)
print("  governor ON  worst error = %6.1f px   meanGain = %.3f" % (np.abs(e[300:]).max(), g[300:].mean()))
