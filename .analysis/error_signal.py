"""Recover the aim error signal the AI saw, from the recorded video.

The crosshair is drawn at a fixed screen position and the target is a static firing-range
dummy, so the AI's error signal is:

    error_n = (world target position) - (camera position at frame n)
            = K - cumulative_camera_shift_n

cumulative camera shift is obtained by integrating the frame-to-frame phase-correlation
shift of the world (sampled in a textured region and median-combined).
"""
import json
import numpy as np
import imageio.v3 as iio

VIDEO = r"C:\Users\Crim\Videos\clip_1,791,123,561,479.mp4"
OUT = r"C:\Users\Crim\Downloads\Compressed\AIMMY-2.5.0-ARDUINO-CUDA-Aimmy-V2-CUDA\.analysis\error_signal.json"

# Texture-rich horizontal band (mid-field architecture) - avoids HUD, sky and weapon model.
BAND_Y0, BAND_Y1 = 600, 900
H = BAND_Y1 - BAND_Y0
W = 1920
window = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)

prev = None
shifts = []
for idx, frame in enumerate(iio.imiter(VIDEO, plugin="FFMPEG")):
    band = frame[BAND_Y0:BAND_Y1, :].astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    band = band - band.mean() if prev is None else band - band.mean()
    if prev is not None:
        fa = np.fft.rfft2(prev * window)
        fb = np.fft.rfft2(band * window)
        cc = np.fft.irfft2(fa * np.conj(fb), s=(H, W))
        # restrict search to +-80 px horizontally, +-40 vertically
        cc2 = np.zeros_like(cc)
        cc2[:41, :81] = cc[:41, :81]
        cc2[-40:, :81] = cc[-40:, :81]
        cc2[:41, -80:] = cc[:41, -80:]
        cc2[-40:, -80:] = cc[-40:, -80:]
        py, px = np.unravel_index(int(np.argmax(cc2)), cc2.shape)
        def refine(arr, i, n):
            im1, i0, ip1 = arr[(i - 1) % n], arr[i], arr[(i + 1) % n]
            d = im1 - 2 * i0 + ip1
            return 0.5 * (im1 - ip1) / d if d != 0 else 0.0
        sx = px + refine(cc2[py, :], px, W)
        sy = py + refine(cc2[:, px], py, H)
        if sx > W / 2: sx -= W
        if sy > H / 2: sy -= H
        shifts.append((idx, float(-sx), float(-sy)))
    prev = band
    if idx % 300 == 0:
        print("frame", idx, flush=True)

json.dump(shifts, open(OUT, "w"))
print("done", len(shifts))
