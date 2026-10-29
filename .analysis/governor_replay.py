"""Replay the recovered aim-error signal through a faithful Python port of the C#
AimLoopController, to check that the governor actually detects the shake and reduces gain,
and that it stays at full gain when the aim is stable."""
import json
import numpy as np

base = r"C:\Users\Crim\Downloads\Compressed\AIMMY-2.5.0-ARDUINO-CUDA-Aimmy-V2-CUDA\.analysis"
err = json.load(open(base + r"\error_signal.json"))
idx = np.array([e[0] for e in err])
camx = np.concatenate([[0.0], np.cumsum([e[1] for e in err])])
camy = np.concatenate([[0.0], np.cumsum([e[2] for e in err])])

# camera also has real (deliberate) movement, so re-centre the error per 2 s window:
fps = 59.94
HW, HH = 960.0, 540.0
N = len(camx)

RefDtMs = 16.0
MinGain, MaxGain = 0.16, 1.0
GainCutPerTick = 0.12
GainRecoverPerSecond = 3.0
EfficientRatio, NoisyRatio = 0.50, 0.15
DeadZonePx = 3.0
RevMinStep, RevGain, RevCap = 2.5, 0.35, 4.0

gain = MaxGain
prev_x = prev_y = None
total_path = netx = nety = 0.0
psx = psy = 0.0
has_prev_step = False
reversal = 0.0
priming = True
prime_ticks = 0
stable_ticks = 0
last_ms = -1
dt = 1000.0 / fps
StableTicksBeforeRecovery = 6


def recover():
    global gain
    if stable_ticks >= StableTicksBeforeRecovery:
        gain = min(MaxGain, gain + GainRecoverPerSecond * dt / 1000.0)


log = []
for n in range(N):
    # error = (fixed world target) - camera;  the deliberate camera travel (hand aiming) is
    # removed by subtracting a slow moving average, leaving what the controller has to fix.
    ex = -(camx[n] - np.mean(camx[max(0, n - 120):n + 1]))
    ey = -(camy[n] - np.mean(camy[max(0, n - 120):n + 1]))
    mag = np.hypot(ex, ey)

    if priming or prev_x is None:
        prime_ticks += 1
        if prime_ticks >= 3:
            priming = False
        prev_x, prev_y = ex, ey
        total_path = netx = nety = 0.0
        stable_ticks = StableTicksBeforeRecovery
        recover()
        log.append((n, gain, 1.0, reversal))
        continue

    if mag < DeadZonePx:
        total_path = netx = nety = 0.0
        prev_x, prev_y = ex, ey
        has_prev_step = False
        reversal = max(0.0, reversal - 0.5)
        stable_ticks = StableTicksBeforeRecovery
        recover()
        log.append((n, gain, 1.0, reversal))
        continue

    sx = ex - prev_x
    sy = ey - prev_y
    step = np.hypot(sx, sy)

    if has_prev_step:
        if (sx * psx + sy * psy) < 0 and step > RevMinStep:
            reversal = min(RevCap, reversal + 1.0)
        else:
            reversal = max(0.0, reversal - 0.25)
    psx, psy = sx, sy
    has_prev_step = True

    if reversal >= 2.0:
        rate = min(max(dt / RefDtMs, 0.25), 4.0)
        gain = max(MinGain, gain - RevGain * rate)

    total_path += step
    netx += sx
    nety += sy
    decay = np.exp(-dt / 250.0)
    total_path *= decay
    netx *= decay
    nety *= decay
    prev_x, prev_y = ex, ey

    net = np.hypot(netx, nety)
    if (total_path < mag * 2.0 or total_path < 12.0) and reversal < 2.0:
        log.append((n, gain, 1.0, reversal))
        stable_ticks = StableTicksBeforeRecovery
        recover()
        continue

    eff = min(max(net / total_path, 0.0), 1.0)
    if eff >= EfficientRatio and reversal < 1.0:
        stable_ticks += 1
        recover()
    elif eff <= NoisyRatio:
        stable_ticks = 0
        rate = min(max(dt / RefDtMs, 0.25), 4.0)
        gain = max(MinGain, gain - GainCutPerTick * rate)
    else:
        stable_ticks = 0
        t = min(max((eff - NoisyRatio) / (EfficientRatio - NoisyRatio), 0.0), 1.0)
        target = MinGain + t * (MaxGain - MinGain)
        gain += (target - gain) * 0.25
        gain = min(max(gain, MinGain), MaxGain)
    log.append((n, gain, eff, reversal))

L = np.array(log)
print("frames:", N)
print()
print(" t(s) | gain  | efficiency | reversal")
win = 30
for s in range(0, N - win, win):
    e = s + win
    print("%6.2f | %5.2f | %10.2f | %8.2f" % (
        s / fps, L[s:e, 1].mean(), L[s:e, 2].mean(), L[s:e, 3].mean()))

print()
for a, b, label in [(60, 210, "calm 1.0-3.5s"), (280, 620, "shake 4.7-10.3s"),
                    (930, 1090, "calm 15.5-18.2s"), (1200, 1360, "shake 20.0-22.7s")]:
    print("%-20s meanGain=%.3f  frac(gain<0.9)=%.2f" % (
        label, L[a:b, 1].mean(), (L[a:b, 1] < 0.9).mean()))
