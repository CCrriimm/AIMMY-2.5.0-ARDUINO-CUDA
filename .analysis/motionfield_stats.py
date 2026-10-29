import json
import numpy as np

base = r"C:\Users\Crim\Downloads\Compressed\AIMMY-2.5.0-ARDUINO-CUDA-Aimmy-V2-CUDA\.analysis"
mf = json.load(open(base + r"\motionfield.json"))
sh = json.load(open(base + r"\shake.json"))

mag_global = np.array([np.hypot(s[1], s[2]) for s in sh])

idx = np.array([r[0] for r in mf])
# shifts[9][2]
S = np.array([[p for p in r[1]] for r in mf])  # (N,9,2)
print("shape", S.shape)

def summarise(a, b, label):
    sl = slice(a, b)
    m = np.hypot(S[sl, :, 0], S[sl, :, 1])  # (n,9)
    mean_patch = m.mean(axis=1)
    # spread across patches
    spread = m.std(axis=1)
    # coherent (global) component = mean vector length
    mvx = S[sl, :, 0].mean(axis=1)
    mvy = S[sl, :, 1].mean(axis=1)
    coh = np.hypot(mvx, mvy)
    print("%-22s meanPatch=%.2f  coherent=%.2f  spread=%.2f  ratio_coh/patch=%.2f" % (
        label, mean_patch.mean(), coh.mean(), spread.mean(), coh.mean() / max(mean_patch.mean(), 1e-9)))

summarise(60, 210, "calm 1.0-3.5s")
summarise(280, 620, "shake 4.7-10.3s")
summarise(930, 1090, "calm 15.5-18.2s")
summarise(1200, 1360, "shake 20.0-22.7s")

print()
print("per-patch mean |shift| during shake (patches ordered top-left..bottom-right):")
m = np.hypot(S[280:620, :, 0], S[280:620, :, 1]).mean(axis=0)
print(np.round(m.reshape(3, 3), 1))
print("per-patch mean |shift| during calm:")
m2 = np.hypot(S[930:1090, :, 0], S[930:1090, :, 1]).mean(axis=0)
print(np.round(m2.reshape(3, 3), 1))
