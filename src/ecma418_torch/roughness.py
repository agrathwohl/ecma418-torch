"""ECMA-418-2 (4th ed., 2025) clause 7.1: roughness. Formula numbers refer to
the standard. Discrete choices (peak positions, which peaks survive, the
harmonic set) are piecewise constant in the input, so their derivative is zero
wherever it exists; the values those choices select stay in the graph."""

import math

import torch
from torch.utils.checkpoint import checkpoint

from .hearing import (
    FC,
    N_BANDS,
    band_features,
    basis_loudness,
    calibrate,
    finish,
    pchip,
    prepare,
    safe_div,
    safe_pow,
    smooth,
)

SB, SH = 16384, 4096
NK = 512  # envelope block length after decimation to 1500 Hz
DF = 1500.0 / NK  # Formula (76)
C_R = 0.0180685
SKIP = 16  # l50 <= 15 discarded, clauses 7.1.8 and 7.1.10

# Table 10, theta = 0..33.
E_THETA = torch.tensor([
    0.0000, 0.0457, 0.0907, 0.1346, 0.1765, 0.2157, 0.2515, 0.2828, 0.3084,
    0.3269, 0.3364, 0.3348, 0.3188, 0.2844, 0.2259, 0.1351, 0.0000, -0.1351,
    -0.2259, -0.2844, -0.3188, -0.3348, -0.3364, -0.3269, -0.3084, -0.2828, -0.2515,
    -0.2157, -0.1765, -0.1346, -0.0907, -0.0457, 0.0000, 0.0000,
], dtype=torch.float64)


def _band_constants(device):
    fk = (FC / 1000).to(device)
    lg = torch.log2(fk)
    low = fk < 1
    r_max = 1 / (1 + torch.where(low, 0.3560, 0.8024) * lg.abs() ** torch.where(low, 0.8049, 0.9333))  # (84)
    f_max = 72.6937 * (1 - 1.1739 * torch.exp(-5.4583 * fk))  # (86)
    q2_high = torch.where(fk < 2 ** -3.4253, 0.2471, 0.2471 + 0.0129 * (lg + 3.4253) ** 2)  # (87)
    q2_low = 1.0967 - 0.0640 * lg  # (96)
    return r_max, f_max, q2_high, q2_low


def _weight(f, f_max, q1, q2):
    """Formula (85)."""
    return 1 / (1 + ((f / f_max - f_max / f) * q1) ** 2) ** q2


def prominent_peaks(x, n_keep):
    """Indices of the n_keep most prominent strict local maxima of each row of
    x (M, n), ascending, padded with -1. Prominence as in clause 7.1.5.1: the
    peak height minus the higher of the lowest points on each side, searched
    until the data rise above the peak or end."""
    with torch.no_grad():
        m, n = x.shape
        pos = torch.arange(n, device=x.device)
        is_pk = torch.zeros_like(x, dtype=torch.bool)
        is_pk[:, 1:-1] = (x[:, 1:-1] > x[:, :-2]) & (x[:, 1:-1] > x[:, 2:])
        p_max = int(is_pk.sum(-1).max()) if m else 0
        out = torch.full((m, n_keep), -1, dtype=torch.long, device=x.device)
        if p_max == 0:
            return out
        chunk = max(1, 2**25 // (p_max * n))
        for s in range(0, m, chunk):
            xs = x[s:s + chunk]
            pk = torch.where(is_pk[s:s + chunk], pos, n).sort(-1).values[:, :p_max]
            valid = pk < n
            pk = pk.clamp(max=n - 1)
            h = xs.gather(-1, pk)[..., None]
            above = xs[:, None, :] > h
            q, p = pos[None, None, :], pk[..., None]
            jl = torch.where(above & (q < p), q, -1).amax(-1, keepdim=True)
            jr = torch.where(above & (q > p), q, n).amin(-1, keepdim=True)
            inf = torch.inf
            lmin = torch.where((q > jl) & (q <= p), xs[:, None, :], inf).amin(-1)
            rmin = torch.where((q >= p) & (q < jr), xs[:, None, :], inf).amin(-1)
            prom = torch.where(valid, h[..., 0] - torch.maximum(lmin, rmin), -inf)
            k = min(n_keep, p_max)
            top = prom.topk(k, -1)
            sel = torch.where(top.values > -inf, pk.gather(-1, top.indices), n)
            out[s:s + chunk, :k] = torch.where(sel < n, sel, -1)
        key = torch.where(out >= 0, out, n)
        return torch.where(key.sort(-1).values < n, key.sort(-1).values, -1)


def _refined_rate(a, b, c, kp, valid):
    """Formulae (73)-(81): quadratic-fit peak rate plus the Hann bias term."""
    curv = torch.where(valid, 2 * b - a - c, 1.0)
    ft = (kp + (c - a) / (2 * curv)) * DF  # (76), vertex of the fit through 3 points
    e = E_THETA.to(ft.device)
    theta = torch.arange(34, dtype=ft.dtype, device=ft.device)
    fl = torch.floor(ft.detach() / DF)[..., None]
    beta = (fl + theta / 32) * DF - (ft[..., None] + e)  # (79)
    tmin = beta[..., :33].abs().argmin(-1, keepdim=True)  # (80)
    b_min = beta.gather(-1, tmin)
    b_prev = beta.gather(-1, (tmin - 1).clamp(min=0))
    tcorr = torch.where((tmin > 0) & (b_min * b_prev < 0), tmin, tmin + 1)  # (81)
    b0, b1 = beta.gather(-1, tcorr - 1)[..., 0], beta.gather(-1, tcorr)[..., 0]
    e0, e1 = e[tcorr - 1][..., 0], e[tcorr][..., 0]
    return ft + e0 - (e1 - e0) * b0 / (b1 - b0)  # (77), (78)


def _fundamental(fp, at, valid):
    """Clause 7.1.5.3 over peaks (..., P). Returns the index of the fundamental
    i_max and the membership mask of its harmonic complex I_max."""
    with torch.no_grad():
        ratio = fp[..., None, :] / fp[..., :, None]  # [i0, i]
        r = torch.round(ratio)
        dev = torch.where(r > 0, (ratio / r - 1).abs(), torch.inf)  # (89), (90)
        pair = valid[..., None, :] & valid[..., :, None]
        dev = torch.where(pair, dev, torch.inf)
        # (89): among peaks with the same integer ratio keep the closest one.
        idx = torch.arange(fp.shape[-1], device=fp.device)
        same = r[..., :, None] == r[..., None, :]  # [i0, i, j]
        beaten = same & ((dev[..., None, :] < dev[..., :, None])
                         | ((dev[..., None, :] == dev[..., :, None]) & (idx[None, :] < idx[:, None])))
        member = (dev < 0.04) & ~beaten.any(-1)  # (90)
        energy = torch.where(member, at.detach()[..., None, :], 0.0).sum(-1)  # (91)
        energy = torch.where(valid, energy, -torch.inf)
        i_max = energy.argmax(-1, keepdim=True)
        in_max = member.gather(-2, i_max[..., None].expand(*i_max.shape, fp.shape[-1]))[..., 0, :]
    return i_max, in_max


def block_roughness(env, nb):
    """Clauses 7.1.3-7.1.5 for one set of blocks. env (B, 53, 512) decimated
    envelopes, nb (B, 53) specific basis loudness. Returns A(l, z) (B, 53)."""
    dev = env.device
    r_max, f_max, q2_high, q2_low = _band_constants(dev)
    n = torch.arange(NK, dtype=env.dtype, device=dev)
    xw = env * ((0.5 - 0.5 * torch.cos(2 * math.pi * n / NK)) / math.sqrt(0.375))
    phi0 = (xw**2).sum(-1)  # (67)
    spec = torch.fft.rfft(xw).abs() ** 2  # k = 0..256 of the 512-point DFT
    nmax = nb.amax(-1, keepdim=True)
    phi = safe_div(nb**2, nmax * phi0)[..., None] * spec  # (66)

    phib = torch.cat([phi[:, :1], (phi[:, :-2] + phi[:, 1:-1] + phi[:, 2:]) / 3, phi[:, -1:]], 1)
    s = phib.sum(1)  # (68)
    mid = s[:, 2:256].sort(-1).values
    med = (mid[:, 126] + mid[:, 127]) / 2
    k = torch.arange(NK // 2 + 1, dtype=env.dtype, device=dev)
    wt = 0.0856 * s / (med[:, None] + 1e-10) * torch.clamp(0.1891 * torch.exp(0.0120 * k), 0, 1)  # (71)
    wmax = wt[:, 2:256].amax(-1, keepdim=True)
    w = torch.where(wt >= 0.05 * wmax, torch.clamp(wt - 0.1407, 0, 1), 0.0)  # (70)
    phih = phib * w[:, None, :]  # (69)

    flat = phih.reshape(-1, phih.shape[-1])
    kp = prominent_peaks(flat[:, 2:256], 10).reshape(*phih.shape[:2], 10)
    valid = kp >= 0
    kp = torch.where(valid, kp + 2, 3)
    b = phih.gather(-1, kp)
    bmax = torch.where(valid, b, -torch.inf).amax(-1, keepdim=True)
    valid = valid & (b > 0.05 * bmax)  # (72)
    a, c = phih.gather(-1, kp - 1), phih.gather(-1, kp + 1)
    fp = torch.where(valid, _refined_rate(a, b, c, kp.to(env.dtype), valid), 1.0)
    amp = a + b + c  # (82)
    fmx = f_max[None, :, None]
    at = amp * r_max[None, :, None] * torch.where(fp > fmx, _weight(fp, fmx, 1.2822, q2_high[None, :, None]), 1.0)
    at = torch.where(valid & (fp > DF), at, 0.0)  # (83)

    i_max, in_max = _fundamental(fp, at, valid)
    i_peak = torch.where(in_max, at, -torch.inf).argmax(-1, keepdim=True)  # (94)
    sa = torch.where(in_max, at, 0.0).sum(-1, keepdim=True)
    centroid = torch.where(in_max, fp * at, 0.0).sum(-1, keepdim=True) / (sa + torch.finfo(env.dtype).eps)
    w_peak = 1 + 0.1 * safe_pow((centroid - fp.gather(-1, i_peak)).abs(), 0.749)  # (93)
    total = (w_peak * sa)[..., 0]  # (92) summed over I_max
    f0 = fp.gather(-1, i_max)[..., 0]
    g_low = torch.where(f0 <= f_max, _weight(f0, f_max, 0.7066, q2_low), 1.0)
    out = torch.where(valid.any(-1) & (f0 > DF), g_low * total, 0.0)  # (95)
    return torch.where(out >= 0.074376, out, 0.0)


def _chunked(fn, env, nb, chunk):
    outs = []
    for s in range(0, env.shape[0], chunk):
        e, v = env[s:s + chunk], nb[s:s + chunk]
        outs.append(checkpoint(fn, e, v, use_reentrant=False) if e.requires_grad else fn(e, v))
    return torch.cat(outs, 0)


def roughness(x, fs, soundfield="free", binaural=True, block_chunk=64):
    """ECMA-418-2 roughness of `x`, sound pressure in Pa, shape (n,) or (C, n)
    with C = 1 or 2, sampled at `fs`. Returns a dict with the specific roughness
    R'(l50, z) ("specific", asper/Bark_HMS), its average over time
    ("specific_avg"), R(l50) ("time_dependent", asper), the 90th-percentile
    single value R ("single"), and the 50 Hz time axis."""
    xp, n = prepare(x, fs, SB)
    (env, rms, t), = band_features(xp, soundfield, [(SB, SH, True)], SB)
    nb = basis_loudness(rms)
    c, nl = env.shape[:2]
    a = _chunked(block_roughness, env.reshape(c * nl, N_BANDS, NK), nb.reshape(c * nl, N_BANDS), block_chunk)
    a = a.reshape(c, nl, N_BANDS).transpose(-1, -2)
    t50 = torch.arange(n * 50 // 48000 + 1, dtype=torch.float64) / 50  # l50 up to floor(t(l_last) r_s50)
    est = torch.relu(pchip(t, a, t50)).transpose(-1, -2)  # (C, T, 53)
    spec = smooth(calibrate(est, C_R), 0.0625, 0.5)
    return finish(spec, SKIP, binaural)
