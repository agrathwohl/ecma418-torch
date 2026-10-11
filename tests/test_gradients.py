"""The gradients autograd returns are the derivatives of the metrics.

1. gradcheck: autograd's Jacobian-vector product against central differences.
2. Taylor remainder: |f(x + h v) - f(x) - h <grad f, v>| shrinks as h^2.
   A wrong gradient leaves an O(h) remainder, so a ratio near 100 per decade
   of h is only possible when the gradient is the true first derivative.
3. Where a metric is exactly zero on a neighbourhood (silence, unmodulated
   noise), the gradient is exactly zero and finite.

The input is parametrised as x0 + sum_i theta_i v_i with |v_i| = |x0|, so
theta is a relative perturbation along random directions of the signal.
"""

from pathlib import Path

import numpy as np
import torch

from ecma418_torch import fluctuation_strength, roughness

FIX = Path(__file__).parent / "fixtures"
FS = 48000


def signal(name):
    return torch.tensor(np.load(FIX / f"{name}.npz")["signal"].astype(np.float64))


def directions(x0, k, seed):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(k, x0.numel(), generator=g, dtype=torch.float64)
    return v / v.norm(dim=1, keepdim=True) * x0.norm()


def outputs(metric, skip):
    return lambda x: torch.cat([(r := metric(x, FS))["single"][None], r["time_dependent"][skip:]])


def check_gradcheck(metric, skip, name, eps, rtol):
    x0 = signal(name)
    v = directions(x0, 3, 0)
    f = outputs(metric, skip)
    theta = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda th: f(x0 + th @ v), (theta,), eps=eps, atol=1e-7, rtol=rtol,
                                    fast_mode=True, nondet_tol=0.0)


def test_roughness_gradcheck():
    check_gradcheck(roughness, 16, "r_mix", 1e-6, 1e-3)


def test_fluctuation_gradcheck():
    check_gradcheck(fluctuation_strength, 36, "f_mix", 1e-5, 1e-3)


def taylor_ratios(metric, skip, name, seed):
    x0 = signal(name)
    f = outputs(metric, skip)
    w = torch.randn(f(x0).shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    x = x0.clone().requires_grad_(True)
    (f(x) * w).sum().backward()
    ratios = []
    for v in directions(x0, 2, seed):
        slope = float((x.grad * v).sum())
        with torch.no_grad():
            f0 = float((f(x0) * w).sum())
            rem = [abs(float((f(x0 + h * v) * w).sum()) - f0 - h * slope) for h in (1e-5, 1e-6)]
        ratios.append(rem[0] / rem[1])
    return ratios


def test_roughness_taylor_remainder_is_second_order():
    for name in ("r_cal", "r_mix"):
        assert min(taylor_ratios(roughness, 16, name, 0)) > 30


def test_fluctuation_directional_derivative():
    x0 = signal("f_mix")
    f = outputs(fluctuation_strength, 36)
    x = x0.clone().requires_grad_(True)
    f(x).sum().backward()
    for v in directions(x0, 2, 1):
        with torch.no_grad():
            fd = float((f(x0 + 1e-5 * v).sum() - f(x0 - 1e-5 * v).sum()) / 2e-5)
        ad = float((x.grad * v).sum())
        assert abs(fd - ad) <= 1e-3 * abs(ad)


def test_zero_metric_has_zero_finite_gradient():
    g = torch.Generator().manual_seed(3)
    noise = torch.randn(2 * FS, generator=g, dtype=torch.float64) * 0.063  # 70 dB SPL, unmodulated
    for x0 in (torch.zeros(2 * FS, dtype=torch.float64), noise):
        x = x0.clone().requires_grad_(True)
        r = roughness(x, FS)["single"]
        r.backward()
        assert float(r.detach()) == 0.0
        assert torch.isfinite(x.grad).all() and (x.grad == 0).all()


def test_level_gradient_matches_difference():
    """d/dg of R(g x): the loudness nonlinearity makes roughness level dependent."""
    x0 = signal("r_mix")
    g = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    roughness(g * x0, FS)["single"].backward()
    with torch.no_grad():
        fd = (roughness((1 + 1e-6) * x0, FS)["single"] - roughness((1 - 1e-6) * x0, FS)["single"]) / 2e-6
    assert abs(float(g.grad) - float(fd)) <= 1e-4 * abs(float(fd))
