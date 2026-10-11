"""Agreement with the standard's calibration, with the commercial implementation
of the standard's authors, and with an independent open implementation.

ArtemiS v17 values (HEAD acoustics) are those quoted, with the same tolerances,
by the reference implementation's tests:
https://github.com/mlotinga/sottek-hearing-model/tree/main/tests
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ecma418_torch import fluctuation, fluctuation_strength, roughness

FIX = Path(__file__).parent / "fixtures"
FS = 48000


def calibration_tone(fm, seconds=10.0, fs=FS):
    """1 kHz carrier, 100 % sinusoidal AM starting at its minimum, 60 dB SPL."""
    t = torch.arange(int(seconds * fs), dtype=torch.float64) / fs
    x = (1 + torch.sin(2 * math.pi * fm * t - math.pi / 2)) * torch.sin(2 * math.pi * 1000 * t)
    return x / x.pow(2).mean().sqrt() * 0.02


def load(name):
    d = np.load(FIX / f"{name}.npz")
    return torch.tensor(d["signal"].astype(np.float64)), d


def test_roughness_calibration_is_one_asper():
    r = roughness(calibration_tone(70), FS)
    assert float(r["single"]) == pytest.approx(1.0, abs=1e-4)
    assert float(r["specific_avg"][17]) == pytest.approx(0.374, abs=1e-3)


def test_fluctuation_calibration_is_one_vacil():
    f = fluctuation_strength(calibration_tone(4), FS)
    assert float(f["single"]) == pytest.approx(1.0, abs=1e-2)
    assert float(f["specific_avg"][17]) == pytest.approx(0.395, abs=1e-2)


@pytest.mark.parametrize("name", ["r_cal", "r_noise", "r_mix"])
def test_roughness_matches_reference(name):
    x, d = load(name)
    r = roughness(x, FS)
    assert r["time_dependent"].numpy() == pytest.approx(d["ref_roughness_t"], abs=5e-5)
    assert r["specific"].numpy() == pytest.approx(d["ref_spec_roughness"], abs=2e-5)
    assert float(r["single"]) == pytest.approx(float(d["ref_roughness90pc"][0]), abs=5e-5)


def test_fluctuation_matches_reference(monkeypatch):
    """Same reading of the three points the reference decides differently from
    the print (its 1e-12 in place of machine epsilon, Newton cap 0.2, fundamental
    required): then the two implementations agree to the outer-ear
    coefficient rounding."""
    monkeypatch.setattr(fluctuation, "EPS", 1e-12)
    x, d = load("f_cal")
    f = fluctuation_strength(x, FS, newton_cap=0.2, require_fundamental=True)
    assert f["time_dependent"].numpy() == pytest.approx(d["ref_fluctuation_t"], abs=1e-4)
    assert float(f["single"]) == pytest.approx(float(d["ref_fluctuation90pc"][0]), abs=1e-4)


def test_fluctuation_complex_signal_matches_reference():
    x, d = load("f_mix")
    f = fluctuation_strength(x, FS, newton_cap=0.2, require_fundamental=True)
    assert float(f["single"]) == pytest.approx(float(d["ref_fluctuation90pc"][0]), rel=1e-3)


def test_diotic_stereo_equals_mono():
    x = calibration_tone(70, 2.0)
    mono = roughness(x, FS)["single"]
    stereo = roughness(torch.stack([x, x]), FS)["single"]
    assert float(stereo) == pytest.approx(float(mono), rel=1e-12)


def test_resampled_input_matches_48k():
    t44 = calibration_tone(70, 3.0, 44100)
    t48 = calibration_tone(70, 3.0, 48000)
    assert float(roughness(t44, 44100)["single"]) == pytest.approx(float(roughness(t48, FS)["single"]), rel=2e-3)
