"""Measure fidelity, gradient correctness and cost; write benchmarks/evidence.md.

    python benchmarks/evidence.py [--gpu]
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ecma418_torch import fluctuation, fluctuation_strength, roughness

HERE = Path(__file__).parent
FIX = HERE.parent / "tests" / "fixtures"
FS = 48000
out = []


def table(head, rows):
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))
    out.extend("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    out.append("")


def tone(fm, seconds=10.0):
    t = torch.arange(int(seconds * FS), dtype=torch.float64) / FS
    x = (1 + torch.sin(2 * math.pi * fm * t - math.pi / 2)) * torch.sin(2 * math.pi * 1000 * t)
    return x / x.pow(2).mean().sqrt() * 0.02


def load(name):
    d = np.load(FIX / f"{name}.npz")
    return torch.tensor(d["signal"].astype(np.float64)), d


out.append("## Calibration, 10 s, against ArtemiS v17\n")
r, f = roughness(tone(70), FS), fluctuation_strength(tone(4), FS)
table(["quantity", "measured", "ArtemiS v17"], [
    ["R (asper)", f"{float(r['single']):.7f}", "1.0"],
    ["R' avg, band 17", f"{float(r['specific_avg'][17]):.5f}", "0.374"],
    ["F (vacil)", f"{float(f['single']):.7f}", "1.00"],
    ["F' avg, band 17", f"{float(f['specific_avg'][17]):.5f}", "0.395"],
])

out.append("## Agreement with sottek-hearing-model (oracle fixtures)\n")
rows = []
for name in ("r_cal", "r_noise", "r_mix"):
    x, d = load(name)
    q = roughness(x, FS)
    rows.append([name, "default", f"{np.abs(q['time_dependent'].numpy() - d['ref_roughness_t']).max():.2e}",
                 f"{float(q['single']):.6f}", f"{float(d['ref_roughness90pc'][0]):.6f}"])
for name in ("f_cal", "f_mix"):
    x, d = load(name)
    for label, kw, eps in (("printed", {}, fluctuation.EPS),
                           ("reference choices", {"newton_cap": 0.2, "require_fundamental": True}, 1e-12)):
        saved, fluctuation.EPS = fluctuation.EPS, eps
        q = fluctuation_strength(x, FS, **kw)
        fluctuation.EPS = saved
        rows.append([name, label, f"{np.abs(q['time_dependent'].numpy() - d['ref_fluctuation_t']).max():.2e}",
                     f"{float(q['single']):.6f}", f"{float(d['ref_fluctuation90pc'][0]):.6f}"])
table(["signal", "settings", "max abs diff, time-dependent", "single", "oracle single"], rows)

out.append("## Gradient: central differences and Taylor remainder\n")
out.append("f = w . [single, time_dependent after the discarded frames], w random; "
           "v random with |v| = |x|; rem(h) = |f(x + h v) - f(x) - h <grad f, v>|.\n")
for metric, skip, name in ((roughness, 16, "r_cal"), (roughness, 16, "r_mix"), (fluctuation_strength, 36, "f_mix")):
    x0, _ = load(name)
    g = torch.Generator().manual_seed(7)
    fn = lambda x: torch.cat([(q := metric(x, FS))["single"][None], q["time_dependent"][skip:]])  # noqa: E731
    w = torch.randn(fn(x0).shape, generator=g, dtype=torch.float64)
    x = x0.clone().requires_grad_(True)
    t0 = time.time()
    (fn(x) * w).sum().backward()
    rows = []
    for k in range(3):
        v = torch.randn(x0.shape, generator=g, dtype=torch.float64)
        v = v / v.norm() * x0.norm()
        ad = float((x.grad * v).sum())
        with torch.no_grad():
            f0 = float((fn(x0) * w).sum())
            prev = None
            for h in (1e-4, 1e-5, 1e-6, 1e-7):
                fp, fm = float((fn(x0 + h * v) * w).sum()), float((fn(x0 - h * v) * w).sum())
                rem = abs(fp - f0 - h * ad)
                rows.append([k, f"{h:.0e}", f"{ad:.9f}", f"{(fp - fm) / (2 * h):.9f}",
                             f"{abs((fp - fm) / (2 * h) - ad) / abs(ad):.1e}", f"{rem:.2e}",
                             f"{prev / rem:.0f}" if prev else ""])
                prev = rem
    out.append(f"### {metric.__name__}, {name}\n")
    table(["dir", "h", "autograd", "central diff", "rel err", "rem(h)", "rem(10h)/rem(h)"], rows)

out.append("## Cost, forward + backward of R + F\n")
rows = []
devices = ["cpu"] + (["cuda"] if "--gpu" in sys.argv and torch.cuda.is_available() else [])
for dev in devices:
    for seconds, ch in ((10, 1), (30, 2)) + (((240, 2),) if dev == "cuda" else ()):
        n = int(seconds * 44100)
        tt = torch.arange(n, dtype=torch.float64) / 44100
        gen = torch.Generator().manual_seed(0)
        x = (0.3 * torch.randn(ch, n, generator=gen, dtype=torch.float64) * (1 + 0.5 * torch.sin(2 * math.pi * 2 * tt))
             + 0.5 * torch.sin(2 * math.pi * 220 * tt)) * 0.05
        x = x.to(dev).requires_grad_(True)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        total = roughness(x, 44100)["single"] + fluctuation_strength(x, 44100)["single"]
        if dev == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        total.backward()
        if dev == "cuda":
            torch.cuda.synchronize()
        t2 = time.time()
        mem = f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if dev == "cuda" else ""
        rows.append([dev, f"{seconds} s x {ch} ch", f"{t1 - t0:.1f} s", f"{t2 - t1:.1f} s", mem])
table(["device", "input at 44.1 kHz", "forward", "backward", "peak memory"], rows)

(HERE / "evidence.md").write_text("# Evidence\n\n" + "\n".join(out))
print("\n".join(out))
