import json
import numpy as np

base = r"C:\Users\Crim\Downloads\Compressed\AIMMY-2.5.0-ARDUINO-CUDA-Aimmy-V2-CUDA\.analysis"
m = json.load(open(base + r"\metrics.json"))
sh = json.load(open(base + r"\shake.json"))

midx = np.array([x[0] for x in m])
rx = np.array([x[1] for x in m])
ry = np.array([x[2] for x in m])
npix = np.array([x[3] for x in m])
ring = np.array([x[4] for x in m])

sidx = np.array([s[0] for s in sh])
sdx = np.array([s[1] for s in sh])
sdy = np.array([s[2] for s in sh])
fps = 59.94

print("ring radius: min %.0f max %.0f median %.0f" % (ring.min(), ring.max(), np.median(ring)))
print()
print(" t(s) | ringR | redpx | reticleX reticleY | |d|/frame | revX%% revY%% | reticle disp/frame")
win = 30
for s in range(0, len(midx) - win, win):
    e = s + win
    t = midx[s] / fps
    rr = np.median(ring[s:e])
    rxs = rx[s:e]
    rys = ry[s:e]
    rdx = np.diff(rxs[~np.isnan(rxs)])
    rdy = np.diff(rys[~np.isnan(rys)])
    sdxw = sdx[s:e]
    sdyw = sdy[s:e]
    revx = (np.diff(np.sign(sdxw)) != 0).mean() * 100
    revy = (np.diff(np.sign(sdyw)) != 0).mean() * 100
    print("%6.2f | %5.0f | %5.0f | %8.1f %8.1f | %8.2f | %4.0f %4.0f | rx %.2f ry %.2f  rstd %.2f %.2f" % (
        t, rr, npix[s:e].mean(),
        np.nanmean(rxs), np.nanmean(rys),
        np.abs(sdxw).mean(),
        revx, revy,
        np.abs(rdx).mean() if len(rdx) else 0, np.abs(rdy).mean() if len(rdy) else 0,
        np.nanstd(rxs), np.nanstd(rys)))
