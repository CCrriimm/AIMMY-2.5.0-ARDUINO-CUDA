"""Local motion field: is the 30-50px/frame jitter a global camera shift (real aim shake)
or local content motion (weapon sway / animation / detection artefacts)?"""
import json
import numpy as np
import imageio.v3 as iio

VIDEO = r"C:\Users\Crim\Videos\clip_1,791,123,561,479.mp4"
OUT = r"C:\Users\Crim\Downloads\Compressed\AIMMY-2.5.0-ARDUINO-CUDA-Aimmy-V2-CUDA\.analysis\motionfield.json"

# 3x3 grid of 160x160 patches spread over the central 1000x1000 area
P = 160
CENTERS = [(x, y) for y in (300, 540, 780) for x in (700, 960, 1220)]

win1 = np.hanning(P)
window = np.outer(win1, win1).astype(np.float32)

prev = None
rows = []
for idx, frame in enumerate(iio.imiter(VIDEO, plugin="FFMPEG")):
    g = frame.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    patches = []
    for (cx, cy) in CENTERS:
        patch = g[cy - P // 2:cy + P // 2, cx - P // 2:cx + P // 2].copy()
        patch -= patch.mean()
        patches.append(patch)
    if prev is not None:
        shifts = []
        for a, b in zip(prev, patches):
            fa = np.fft.rfft2(a * window)
            fb = np.fft.rfft2(b * window)
            cc = np.fft.irfft2(fa * np.conj(fb), s=a.shape)
            py, px = np.unravel_index(int(np.argmax(cc)), cc.shape)
            def refine(arr, i, n):
                im1, i0, ip1 = arr[(i - 1) % n], arr[i], arr[(i + 1) % n]
                d = im1 - 2 * i0 + ip1
                return 0.5 * (im1 - ip1) / d if d != 0 else 0.0
            sx = px + refine(cc[py, :], px, cc.shape[1])
            sy = py + refine(cc[:, px], py, cc.shape[0])
            if sx > P / 2: sx -= P
            if sy > P / 2: sy -= P
            shifts.append((-sx, -sy))
        rows.append((idx, shifts))
    prev = patches
    if idx % 300 == 0:
        print("frame", idx, flush=True)

json.dump(rows, open(OUT, "w"))
print("done", len(rows))
