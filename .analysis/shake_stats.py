import json
import numpy as np

sh = json.load(open(r"C:\Users\Crim\Downloads\Compressed\AIMMY-2.5.0-ARDUINO-CUDA-Aimmy-V2-CUDA\.analysis\shake.json"))
idx = np.array([s[0] for s in sh])
dx = np.array([s[1] for s in sh])
dy = np.array([s[2] for s in sh])
peak = np.array([s[3] for s in sh])
fps = 59.94

# signed cumulative camera position
px = np.concatenate([[0], np.cumsum(dx)])
py = np.concatenate([[0], np.cumsum(dy)])

# Detrend with a 0.5s (30 frame) moving average to isolate oscillation from real aiming
def movavg(a, w):
    k = np.ones(w) / w
    return np.convolve(a, k, mode="same")

smooth_x = movavg(px, 31)
smooth_y = movavg(py, 31)
osc_x = px - smooth_x
osc_y = py - smooth_y

print("Global: mean |d| per frame: x=%.3f y=%.3f px" % (np.abs(dx).mean(), np.abs(dy).mean()))
print("Global: reversals x=%d/%d (%.1f%%)  y=%d/%d (%.1f%%)" % (
    ((np.diff(np.sign(dx)) != 0).sum()), len(dx), 100 * (np.diff(np.sign(dx)) != 0).sum() / len(dx),
    ((np.diff(np.sign(dy)) != 0).sum()), len(dy), 100 * (np.diff(np.sign(dy)) != 0).sum() / len(dy)))
print("Global: residual oscillation RMS: x=%.2f y=%.2f px" % (osc_x.std(), osc_y.std()))
print()

win = 30
print(" t(s) | movX movY (px/s) | meanAbsDx meanAbsDy | revX% revY% | OScRMS_x OScRMS_y | match")
for s in range(0, len(dx) - win, win):
    e = s + win
    t = idx[s] / fps
    mx = (px[e] - px[s]) / (win / fps)
    my = (py[e] - py[s]) / (win / fps)
    rx = ((np.diff(np.sign(dx[s:e])) != 0).sum()) / (e - s) * 100
    ry = ((np.diff(np.sign(dy[s:e])) != 0).sum()) / (e - s) * 100
    print("%6.2f | %7.0f %7.0f | %8.2f %8.2f | %4.0f %4.0f | %7.2f %7.2f | %.2f" % (
        t, mx, my, np.abs(dx[s:e]).mean(), np.abs(dy[s:e]).mean(), rx, ry,
        osc_x[s + 1:e + 1].std(), osc_y[s + 1:e + 1].std(), peak[s:e].mean()))

np.save(r"C:\Users\Crim\Downloads\Compressed\AIMMY-2.5.0-ARDUINO-CUDA-Aimmy-V2-CUDA\.analysis\campos.npy",
        np.stack([idx, dx, dy, px[1:], py[1:]]))
