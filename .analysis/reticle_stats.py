import json
import numpy as np
import sys

p = r"C:\Users\Crim\Downloads\Compressed\AIMMY-2.5.0-ARDUINO-CUDA-Aimmy-V2-CUDA\.analysis\reticle.json"
d = json.load(open(p))
d = [x for x in d if x[3] > 0]
idx = np.array([x[0] for x in d])
cx = np.array([x[1] for x in d])
cy = np.array([x[2] for x in d])
n = np.array([x[3] for x in d])

fps = 59.94
print("frames with red pixels:", len(d), "of", d[-1][0] + 1)
print("centroid x range: %.1f .. %.1f  y range: %.1f .. %.1f" % (cx.min(), cx.max(), cy.min(), cy.max()))

# per 0.5s window: jitter metric = std of frame-to-frame delta of centroid, and mean abs delta
win = 30  # ~0.5 s
print()
print(" t(s) |  npx |  meanX |  meanY |  MAD_dx |  MAD_dy | std_dx | std_dy | spd(px/s)")
for s in range(0, len(d), win):
    e = min(s + win, len(d))
    if e - s < 10:
        continue
    xs, ys = cx[s:e], cy[s:e]
    dx = np.diff(xs); dy = np.diff(ys)
    t = idx[s] / fps
    print("%6.2f | %4.0f | %6.1f | %6.1f | %6.2f | %6.2f | %6.2f | %6.2f | %8.1f" % (
        t, n[s:e].mean(), xs.mean(), ys.mean(),
        np.abs(dx).mean(), np.abs(dy).mean(), dx.std(), dy.std(),
        np.sqrt(dx**2 + dy**2).mean() * fps))
