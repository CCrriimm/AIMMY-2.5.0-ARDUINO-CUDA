import numpy as np
import imageio.v3 as iio

VIDEO = r"C:\Users\Crim\Videos\clip_1,791,123,561,479.mp4"

want = {60, 300, 500, 700, 900, 1100, 1400}
for idx, frame in enumerate(iio.imiter(VIDEO, plugin="FFMPEG")):
    if idx not in want:
        continue
    sub = frame[540-90:540+90, 960-90:960+90].astype(np.int16)
    r, g, b = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
    mask = (r > 140) & (r - g > 50) & (r - b > 50)
    print(f"--- frame {idx}: red px in 180x180 center crop = {mask.sum()}")
    ys, xs = np.nonzero(mask)
    if len(ys):
        print("    centroid offset from center: x=%.1f y=%.1f" % (xs.mean()-90, ys.mean()-90))
        print("    bbox x %d..%d  y %d..%d" % (xs.min()-90, xs.max()-90, ys.min()-90, ys.max()-90))
    # print a coarse ascii map (every 4 px)
    H, W = mask.shape
    for y in range(0, H, 4):
        row = "".join("#" if mask[y:y+4, x:x+4].any() else "." for x in range(0, W, 4))
        print("   ", row)
    if idx == max(want):
        break
