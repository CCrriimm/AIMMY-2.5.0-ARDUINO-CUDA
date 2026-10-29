"""Clean, bounded comparison: close a 40 px error, then track a target moving at 4 px/update.
No overflow - the loop is capped and monitored. CombinedGain is the *real* loop gain including
the game's per-optic mouse scaling, which the Ai code cannot observe."""
import numpy as np


def simulate(combined_gain, use_cap, target_speed=0.0, ticks=600, max_step_ratio=0.85,
             min_step_cap=6.0, deadzone=3.0):
    cam = 0.0
    errs = []
    for n in range(ticks):
        target = 40.0 + target_speed * n
        e = target - cam
        errs.append(e)
        if abs(e) <= deadzone:
            u = 0.0
        else:
            u = e
            if use_cap:
                cap = max(min_step_cap, abs(e) * max_step_ratio)
                u = float(np.clip(u, -cap, cap))
        cam += combined_gain * u
        if not np.isfinite(cam) or abs(cam - target) > 5000:
            errs.extend([np.nan] * (ticks - len(errs)))
            break
    errs = np.array(errs)
    sl = slice(50, None)
    return np.nanmean(np.abs(errs[sl])), np.nanmax(np.abs(errs[sl])), np.isnan(errs).any()


print("A) Static target 40 px away - does the crosshair overshoot and chatter?")
print()
print(" combinedGain | original: meanAbs maxAbs unstable | capped: meanAbs maxAbs unstable")
for gain in (0.5, 0.84, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0):
    a1, a2, au = simulate(gain, False)
    b1, b2, bu = simulate(gain, True)
    print("%13.2f | %17.1f %6.1f %8s | %15.1f %6.1f %8s" % (gain, a1, a2, au, b1, b2, bu))

print()
print("B) Target moving 4 px/update (240 px/s at 60 fps) - steady-state tracking error")
print()
print(" combinedGain | original: meanAbs unstable | capped: meanAbs unstable")
for gain in (0.5, 0.84, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0):
    a1, a2, au = simulate(gain, False, target_speed=4.0)
    b1, b2, bu = simulate(gain, True, target_speed=4.0)
    print("%13.2f | %17.1f %8s | %15.1f %8s" % (gain, a1, au, b1, bu))

print()
print("C) Highest real gain the loop can survive, and the worst-case chatter amplitude")
print()
for label, use in (("original", False), ("capped", True)):
    survived = []
    for gain in np.arange(0.2, 6.0, 0.1):
        _, _, unstable = simulate(gain, use, target_speed=4.0)
        if not unstable:
            survived.append(round(float(gain), 2))
    print("  %-9s stable for combinedGain in 0.2 .. %.2f" % (label, max(survived)))
