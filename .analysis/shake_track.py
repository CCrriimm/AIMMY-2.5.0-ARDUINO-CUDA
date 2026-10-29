"""Measure frame-to-frame camera (aim) jitter from the video using phase correlation.

The crosshair is drawn at a FIXED screen position; what oscillates when the aim
"shakes" is the GAME CAMERA, i.e. the world content. So the frame-to-frame shift
of the world inside the central viewport is exactly the aim movement.
"""
import json
import sys
import numpy as np
import imageio.v3 as iio

VIDEO = r"C:\Users\Crim\Videos\clip_1,791,123,561,479.mp4"
OUT = r"C:\Users\Crim\Downloads\Compressed\AIMMY-2.5.0-ARDUINO-CUDA-Aimmy-V2-CUDA\.analysis\shake.json"

R = 320          # half-size of analysed square around screen centre
CX, CY = 960, 540

win1 = np.hanning(2 * R)
window = np.outer(win1, win1).astype(np.float32)

prev = None
shifts = []   # (frame_index, dx, dy, peak)

for idx, frame in enumerate(iio.imiter(VIDEO, plugin="FFMPEG")):
    sub = frame[CY - R:CY + R, CX - R:CX + R]
    g = sub.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    g -= g.mean()

    if prev is not None:
        a = np.fft.rfft2(prev * window)
        b = np.fft.rfft2(g * window)
        cc = np.fft.irfft2(a * np.conj(b), s=prev.shape)
        peak_idx = int(np.argmax(cc))
        py, px = np.unravel_index(peak_idx, cc.shape)
        peak = float(cc[py, px])
        # sub-pixel parabola refinement
        def refine(arr, i, n):
            im1 = arr[(i - 1) % n]
            i0 = arr[i]
            ip1 = arr[(i + 1) % n]
            denom = (im1 - 2 * i0 + ip1)
            return 0.5 * (im1 - ip1) / denom if denom != 0 else 0.0
        ddx = refine(cc[py, :], px, cc.shape[1])
        ddy = refine(cc[:, px], py, cc.shape[0])
        sx = px + ddx
        sy = py + ddy
        if sx > R: sx -= 2 * R
        if sy > R: sy -= 2 * R
        # sign: cc peak at (px,py) means current frame = prev shifted by (+px,+py)?
        shifts.append((idx, float(-sx), float(-sy), peak))
    prev = g
    if idx % 300 == 0:
        print("frame", idx, flush=True)

json.dump(shifts, open(OUT, "w"))
print("done", len(shifts))
