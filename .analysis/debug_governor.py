import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location("tg", r"C:\Users\Crim\Downloads\Compressed\AIMMY-2.5.0-ARDUINO-CUDA-Aimmy-V2-CUDA\.analysis\tune_governor.py")
# reuse the classes without executing the report
src = open(spec.origin).read().split("CANDIDATES = {")[0]
ns = {}
exec(compile(src, "tg", "exec"), ns)
Governor = ns["Governor"]

REF_DT, MIN_GAIN, MAX_GAIN = 16.0, 0.16, 1.0
MIN_DT_SCALE, MAX_DT_SCALE = 0.70, 2.00
MAX_STEP_RATIO, MIN_STEP_CAP, DEADZONE, SENS = 0.85, 6.0, 3.0, 0.79

# scenario A: lever 4, 60 fps, noise 3
dt = 1000 / 60
dts = min(max(dt / REF_DT, MIN_DT_SCALE), MAX_DT_SCALE)
k = 1 - SENS
print("dt=%.2fms dts=%.3f  perTickGain k*dts=%.4f  loopGain=%.3f" % (dt, dts, k * dts, k * dts * 4.0))
rng = np.random.default_rng(11)
ticks = 1200
amp = 300 * 240 * (dt / 1000) / 2
t = np.zeros(ticks)
v = 300.0
for n in range(1, ticks):
    t[n] = t[n - 1] + v * (dt / 1000.0)
    if n % 240 == 0:
        v = -v
kern = np.hanning(11); kern /= kern.sum()
t = np.convolve(t, kern, mode="same"); t -= t.mean()
t = t + rng.normal(0, 3.0, ticks)

gov = Governor(2.5, 0.0, 0.35, 12.0, 2.0)
cam = 0.0
for n in range(120):
    e = t[n] - cam
    gov.evaluate(e, dt)
    g = gov.gain
    cap = max(MIN_STEP_CAP, abs(e) * MAX_STEP_RATIO)
    u = 0.0 if abs(e) <= DEADZONE else np.clip(k * g * e * dts, -cap, cap)
    cam += u * 4.0 * dts
    if n < 60:
        print("n=%3d e=%9.3f g=%.3f rev=%.2f path=%7.2f net=%7.2f u=%8.3f cam=%9.3f" % (
            n, e, g, gov.rev, gov.total_path, gov.net, u, cam))
