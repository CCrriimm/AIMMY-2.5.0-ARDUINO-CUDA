import json
import numpy as np

base = r"C:\Users\Crim\Downloads\Compressed\AIMMY-2.5.0-ARDUINO-CUDA-Aimmy-V2-CUDA\.analysis"
sh = json.load(open(base + r"\shake.json"))
m = json.load(open(base + r"\metrics.json"))

sidx = np.array([s[0] for s in sh])
sdx = np.array([s[1] for s in sh])
sdy = np.array([s[2] for s in sh])
peak = np.array([s[3] for s in sh])
mag = np.sqrt(sdx ** 2 + sdy ** 2)

npix = np.array([x[3] for x in m])[1:]

def show(a, b, label):
    sl = slice(a, b)
    print("=== %s  frames %d..%d (t %.2f..%.2f s)" % (label, a, b, a / 59.94, b / 59.94))
    print("   mag mean %.2f  p90 %.2f  peak(score) mean %.3g" % (
        mag[sl].mean(), np.percentile(mag[sl], 90), peak[sl].mean()))
    print("   |dx| %.2f |dy| %.2f" % (np.abs(sdx[sl]).mean(), np.abs(sdy[sl]).mean()))

show(60, 210, "calm 1.0-3.5s")
show(280, 620, "shake 4.7-10.3s")
show(930, 1090, "calm 15.5-18.2s")
show(1200, 1360, "shake 20.0-22.7s")

print()
# find transitions: rolling 30-frame mean of magnitude
roll = np.convolve(mag, np.ones(30) / 30, mode="same")
state = roll > 8
edges = np.nonzero(np.diff(state.astype(int)) != 0)[0]
print("transitions (frame, t, ->state):")
for e in edges:
    print("  f=%4d t=%5.2f -> %s" % (e, e / 59.94, "SHAKE" if state[e + 1] else "calm"))

print()
print("calm rolling mean: %.2f   shake rolling mean: %.2f" % (roll[~state].mean(), roll[state].mean()))
print("match score calm: %.3g   shake: %.3g" % (peak[~state].mean(), peak[state].mean()))
# normalised correlation coefficient quality
print("red px near crosshair calm: %.0f  shake: %.0f" % (npix[~state[1:]].mean(), npix[state[1:]].mean()))
