import sys, json
import numpy as np
import imageio.v3 as iio

VIDEO = r"C:\Users\Crim\Videos\clip_1,791,123,561,479.mp4"

# Only analyse the central region of the 1920x1080 frame (the crosshair lives there)
CX, CY = 960, 540
HALF = 450

centroids = []  # (frame_index, cx, cy, npix)

# Use ffmpeg plugin via imageio, grayscale=False
for idx, frame in enumerate(iio.imiter(VIDEO, plugin="FFMPEG")):
    sub = frame[CY - HALF:CY + HALF, CX - HALF:CX + HALF]
    r = sub[:, :, 0].astype(np.int16)
    g = sub[:, :, 1].astype(np.int16)
    b = sub[:, :, 2].astype(np.int16)
    mask = (r > 150) & (r - g > 60) & (r - b > 60)
    n = int(mask.sum())
    if n > 0:
        ys, xs = np.nonzero(mask)
        centroids.append((idx, float(xs.mean()) + CX - HALF, float(ys.mean()) + CY - HALF, n))
    else:
        centroids.append((idx, float("nan"), float("nan"), 0))
    if idx % 200 == 0:
        print(f"frame {idx}", flush=True)

with open(r"C:\Users\Crim\Downloads\Compressed\AIMMY-2.5.0-ARDUINO-CUDA-Aimmy-V2-CUDA\.analysis\reticle.json", "w") as f:
    json.dump(centroids, f)
print("frames:", len(centroids))
