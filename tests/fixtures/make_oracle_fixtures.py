"""Regenerate the fixtures: run the GPL reference implementation
(https://github.com/mlotinga/sottek-hearing-model, commit recorded in each file)
as a black box and store its outputs next to the exact input signals.

    python make_oracle_fixtures.py <sottek-hearing-model>/src <out_dir> [case ...]
"""
import subprocess
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
from sottek_hearing_model import shm_fluctuation_ecma, shm_roughness_ecma

fs = 48000
p0 = 2e-5
def am(fc, fm, spl, dur, m=1.0):
    t = np.arange(int(dur * fs)) / fs
    x = (1 + m * np.sin(2 * np.pi * fm * t)) * np.sin(2 * np.pi * fc * t)
    return (x / np.sqrt(np.mean(x ** 2)) * p0 * 10 ** (spl / 20)).astype(np.float32).astype(np.float64)

rng = np.random.default_rng(418)
def noise(spl, dur):
    x = rng.standard_normal(int(dur * fs))
    return (x / np.sqrt(np.mean(x ** 2)) * p0 * 10 ** (spl / 20)).astype(np.float32).astype(np.float64)

def mix(dur):
    t = np.arange(int(dur * fs)) / fs
    x = (0.6 * (1 + 0.8 * np.sin(2 * np.pi * 43 * t)) * np.sin(2 * np.pi * 2500 * t)
         + 0.4 * (1 + 0.9 * np.sin(2 * np.pi * 3 * t)) * np.sin(2 * np.pi * 300 * t)
         + 0.1 * rng.standard_normal(t.size))
    return (x / np.sqrt(np.mean(x ** 2)) * p0 * 10 ** (70 / 20)).astype(np.float32).astype(np.float64)

rev = subprocess.run(["git", "-C", sys.argv[1] + "/..", "rev-parse", "HEAD"],
                     capture_output=True, text=True).stdout.strip()
cases = {
    "r_cal": (am(1000, 70, 60, 2.0), "roughness"),
    "r_noise": (noise(70, 2.0), "roughness"),
    "r_mix": (mix(2.0), "roughness"),
    "f_cal": (am(1000, 4, 60, 4.0), "fluctuation"),
    "f_mix": (mix(4.0), "fluctuation"),
}
only = sys.argv[3:] or list(cases)
for name in only:
    x, kind = cases[name]
    fn = shm_roughness_ecma if kind == "roughness" else shm_fluctuation_ecma
    out = fn(x, fs, wait_bar=False, out_plot=False, binaural=False, parallel_cores=1)
    arrays = {k: np.asarray(v) for k, v in out.items() if isinstance(v, (np.ndarray, float, int, np.floating))}
    np.savez_compressed(f"{sys.argv[2]}/{name}.npz", signal=x.astype(np.float32), fs=fs,
                        oracle_commit=rev, **{"ref_" + k: v for k, v in arrays.items()})
    print(name, {k: np.shape(v) for k, v in arrays.items()}, flush=True)
