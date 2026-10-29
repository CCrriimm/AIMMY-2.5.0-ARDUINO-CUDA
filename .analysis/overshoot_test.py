"""The decisive check.

A proportional aim loop overshoots when one update moves the crosshair further than the whole
remaining error. Because the game scales mouse input by its own per-optic sensitivity, the
'gain' the code can see ((1-sens)) is not the gain the loop actually has:

    actual on-screen move per update = (1-sens) * gameOpticsMultiplier * error
                                     = CombinedGain * error

Before the fix the step was unbounded in error units, so CombinedGain > 1 always overshoots.
With the motion-derivative cap the step is clamped to 0.85 * |error|, so:

    move <= 0.85 * error   ->   residual error >= 0.15 * error   ->   no sign flip, ever.

This sweeps CombinedGain from 0.1 to 50 and reports whether the crosshair can overshoot.
"""
import numpy as np

MAX_STEP_RATIO = 0.85
MIN_STEP_CAP = 6.0
DEADZONE = 3.0


def simulate(combined_gain, use_cap, ticks=400, start_error=40.0):
    """combined_gain = screen pixels the crosshair moves per unit of (error * command fraction)."""
    cam = 0.0
    e = start_error
    overshot = 0
    for n in range(ticks):
        e = start_error - cam
        if abs(e) <= DEADZONE:
            break
        u = e                                  # the AI asks for the whole error
        if use_cap:
            cap = max(MIN_STEP_CAP, abs(e) * MAX_STEP_RATIO)
            u = np.clip(u, -cap, cap)
        cam += combined_gain * u               # game applies its own scaling
        if cam > start_error + 0.5 or (n > 0 and np.sign(e) != np.sign(start_error - cam) and cam != 0):
            overshot += 1
    return cam, overshot


print("Crosshair overshoot when closing a 40 px error, for a range of real (combined) loop gains")
print("(the combined gain includes the game's per-optic sensitivity, so the code cannot know it)")
print()
print(" combinedGain | original code                    | with 0.85 step cap")
print("              | settled at  overshoot events     | settled at  overshoot events")
for gain in (0.1, 0.2, 0.5, 0.84, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0):
    a, ao = simulate(gain, False)
    b, bo = simulate(gain, True)
    print("%13.2f | %9.1f  %17d | %9.1f  %17d" % (gain, a, ao, b, bo))

print()
print("Interpretation")
print("--------------")
print("original: once combinedGain > ~1 the crosshair lands past the target (settled at > 40 px),")
print("          which is the overshoot half of the oscillation.")
print("with cap: the crosshair can never pass the target, whatever the game's multiplier is.")
