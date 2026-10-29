"""Per-frame metrics directly from the video:
 - red reticle centroid inside the middle of the viewport (crosshair path)
 - scope ring radius (proxy for the game's zoom level: 2x vs 4x)
"""
import json
import numpy as np
import imageio.v3 as iio

VIDEO = r"C:\Users\Crim\Videos\clip_1,791,123,561,479.mp4"
OUT = r"C:\Users\Crim\Downloads\Compressed\AIMMY-2.5.0-ARDUINO-CUDA-Aimmy-V2-CUDA\.analysis\metrics.json"

CX, CY = 960, 540
R_CROP = 60          # reticle search crop
R_SCAN = 500         # scope ring radial scan limit

rows = []
for idx, frame in enumerate(iio.imiter(VIDEO, plugin="FFMPEG")):
    sub = frame[CY - R_CROP:CY + R_CROP, CX - R_CROP:CX + R_CROP].astype(np.int16)
    r, g, b = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
    mask = (r > 140) & (r - g > 50) & (r - b > 50)
    n = int(mask.sum())
    if n > 0:
        ys, xs = np.nonzero(mask)
        rx = float(xs.mean()) - R_CROP
        ry = float(ys.mean()) - R_CROP
    else:
        rx = ry = float("nan")

    # radial darkness profile: mean luminance at each radius, sampled on 16 rays
    g_full = frame.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    prof = []
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    radii = np.arange(120, R_SCAN, 4)
    for rad in radii:
        xs2 = (CX + rad * np.cos(angles)).astype(int)
        ys2 = (CY + rad * np.sin(angles)).astype(int)
        prof.append(float(g_full[ys2, xs2].mean()))
    prof = np.array(prof)
    # ring = darkest radius (scope housing is much darker than the scene)
    darkest = float(radii[int(np.argmin(prof))])
    rows.append((idx, rx, ry, n, darkest))
    if idx % 300 == 0:
        print("frame", idx, flush=True)

json.dump(rows, open(OUT, "w"))
print("done", len(rows))
