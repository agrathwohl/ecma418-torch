"""ECMA-418-2 (4th ed., 2025) clause 5: the Sottek Hearing Model front end.

Every stage is a torch operation on float64 tensors, so autograd differentiates
the whole chain. Linear stages (resampling, fade-in, outer/middle-ear filter,
auditory filter bank, segmentation, Hilbert envelope, decimation) are applied
exactly in the frequency domain. Formula numbers refer to the standard.
"""

import math

import torch
from torch.utils.checkpoint import checkpoint

FS = 48000
P0 = 2e-5
N_BANDS = 53
DZ = 0.5
Z = torch.arange(1, N_BANDS + 1, dtype=torch.float64) * DZ  # 0.5 ... 26.5 Bark_HMS
DF0, C_BARK = 81.9289, 0.1618
FC = DF0 / C_BARK * torch.sinh(C_BARK * Z)  # Formula (9)
BW = torch.sqrt(DF0**2 + (C_BARK * FC) ** 2)  # Formula (10)
ORDER = 5
EULER = (1.0, 11.0, 11.0, 1.0)  # e_1..e_4 for k = 5, Formula (15)

# Table 1: outer and middle/inner ear, (b0, b1, b2, a1, a2) per second-order section.
EAR_SOS = (
    (1.015896, -1.925299, 0.922118, -1.925299, 0.938014),
    (0.958943, -1.806088, 0.876439, -1.806088, 0.835382),
    (0.961372, -1.763632, 0.821788, -1.763632, 0.783160),
    (2.225804, -1.434650, -0.498204, -1.434650, 0.727599),
    (0.471735, -0.366092, 0.244145, -0.366092, -0.284120),
    (0.115267, 0.000000, -0.115267, -1.796003, 0.805838),
    (0.988029, -1.912434, 0.926132, -1.912434, 0.914161),
    (1.952238, 0.162320, -0.667994, 0.162320, 0.284244),
)

# Table 2 and Formula (23).
C_N = 0.0211964
ALPHA = 1.5
NU = (1.0, 0.6602, 0.0864, 0.6384, 0.0328, 0.4068, 0.2082, 0.3994, 0.6434)
PT = tuple(P0 * 10 ** (db / 20) for db in (15, 25, 35, 45, 55, 65, 75, 85))

# Table 3.
LTQ = torch.tensor([
    0.3310, 0.1625, 0.1051, 0.0757, 0.0576, 0.0453, 0.0365, 0.0298, 0.0247, 0.0207, 0.0176,
    0.0151, 0.0131, 0.0115, 0.0103, 0.0093, 0.0086, 0.0081, 0.0077, 0.0074, 0.0073, 0.0072,
    0.0071, 0.0072, 0.0073, 0.0074, 0.0076, 0.0079, 0.0082, 0.0086, 0.0092, 0.0100, 0.0109,
    0.0122, 0.0138, 0.0157, 0.0172, 0.0180, 0.0180, 0.0177, 0.0176, 0.0177, 0.0182, 0.0190,
    0.0202, 0.0217, 0.0237, 0.0263, 0.0296, 0.0339, 0.0398, 0.0485, 0.0622,
], dtype=torch.float64)


def safe_sqrt(x):
    """sqrt with gradient 0 at 0. Every zero this sees comes from a clamp or a
    threshold that holds it at zero on an open set, where 0 is the derivative."""
    pos = x > 0
    return torch.where(pos, torch.sqrt(torch.where(pos, x, 1.0)), 0.0)


def safe_pow(x, e):
    """x**e for x >= 0 with gradient 0 at 0, same reasoning as safe_sqrt."""
    pos = x > 0
    return torch.where(pos, torch.where(pos, x, 1.0) ** e, 0.0)


def safe_div(num, den):
    """num/den where den != 0, else 0 (Formula (66) defines the zero case)."""
    nz = den != 0
    return torch.where(nz, num / torch.where(nz, den, 1.0), 0.0)


def _fast_len(n):
    """Smallest 2^a 3^b 5^c >= n."""
    best = 1 << (n - 1).bit_length()
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            m = p35
            while m < n:
                m *= 2
            best = min(best, m)
            p35 *= 3
        p5 *= 5
    return best


def resample(x, fs):
    """Band-limited resampling to 48 kHz (clause 5.1.1, note 2), exact rational
    ratio, by the Fourier method. The input is zero-padded first so the periodic
    extension does not wrap the end of the signal onto its start."""
    if fs == FS:
        return x
    g = math.gcd(int(fs), FS)
    up, down = FS // g, int(fs) // g
    n = x.shape[-1]
    n_pad = -(-(n + 4096) // down) * down
    m = n_pad * up // down
    X = torch.fft.rfft(x, n_pad)
    if m > n_pad and n_pad % 2 == 0:
        X = torch.cat([X[..., :-1], X[..., -1:] / 2], -1)
    return (torch.fft.irfft(X, m) * (m / n_pad))[..., : -(-n * up // down)]


def fade_in(x):
    """Formula (1)."""
    n = torch.arange(240, dtype=x.dtype, device=x.device)
    w = 0.5 - 0.5 * torch.cos(math.pi * n / 240)
    return torch.cat([x[..., :240] * w, x[..., 240:]], -1)


def _ear_response(omega, soundfield):
    zi = torch.exp(-1j * omega)
    h = torch.ones_like(zi)
    for b0, b1, b2, a1, a2 in EAR_SOS[2:] if soundfield == "diffuse" else EAR_SOS:
        h = h * (b0 + b1 * zi + b2 * zi * zi) / (1 + a1 * zi + a2 * zi * zi)
    return h


def _band_response(omega, band):
    """DTFT of the real band-pass impulse response 2*Re(h_LP[n] e^{j w0 n}),
    Formulae (11), (12). h_LP[n] = c n^4 d^n has the closed-form transform
    c (sum e_i d^i z^-i) / (1 - d z^-1)^5, so no truncation is involved."""
    tau = math.comb(2 * ORDER - 2, ORDER - 1) / 2 ** (2 * ORDER - 1) / float(BW[band])
    d = math.exp(-1.0 / (FS * tau))
    c = (1 - d) ** ORDER / sum(e * d ** (i + 1) for i, e in enumerate(EULER))
    w0 = 2 * math.pi * float(FC[band]) / FS

    def g(w):
        zi = torch.exp(-1j * w)
        num = sum(e * d ** (i + 1) * zi ** (i + 1) for i, e in enumerate(EULER))
        return c * num / (1 - d * zi) ** ORDER

    return g(omega - w0) + torch.conj(g(-omega - w0))


def _omega(nfft, device):
    return 2 * math.pi * torch.arange(nfft // 2 + 1, dtype=torch.float64, device=device) / nfft


def segment(x, sb, sh):
    """Clause 5.1.5.2: blocks of sb every sh from the start, plus one last block
    holding the final sb samples when the regular blocks leave a remainder.
    Returns the blocks (..., L, sb) and each block's end time in seconds on the
    unpadded time axis (Formula (103))."""
    n = x.shape[-1]
    n_reg = (n - sb) // sh + 1
    blocks = x.unfold(-1, sb, sh)[..., :n_reg, :]
    t = torch.arange(n_reg, dtype=torch.float64) * sh / FS
    if (n_reg - 1) * sh + sb < n:
        blocks = torch.cat([blocks, x[..., None, n - sb:]], -2)
        t = torch.cat([t, torch.tensor([(n - sb) / FS])])
    return blocks, t


def block_rms(blocks):
    """Formulae (21), (22)."""
    return safe_sqrt(2.0 / blocks.shape[-1] * (torch.relu(blocks) ** 2).sum(-1))


def loudness_nonlinearity(p):
    """Formula (23), specific loudness from an RMS pressure in Pa."""
    out = C_N * p / P0
    for i, pt in enumerate(PT):
        out = out * (1 + (p / pt) ** ALPHA) ** ((NU[i + 1] - NU[i]) / ALPHA)
    return out


def basis_loudness(p):
    """Formulae (24), (25). p (..., 53)."""
    return torch.relu(loudness_nonlinearity(p) - LTQ.to(p.device))


def envelope(blocks, decim=32):
    """Formula (65)/(119) with the factor-32 decimation: |analytic signal| of
    each block at samples 0, 32, 64, ... The analytic spectrum is folded to the
    decimated length before the inverse transform, which returns exactly those
    samples of the full-length inverse."""
    sb = blocks.shape[-1]
    half = sb // 2
    xb = torch.fft.rfft(blocks)
    # Analytic spectrum: 2 X[k] for 0 < k < sb/2, X[0] and X[sb/2] once, zero above.
    folded = 2 * xb[..., :half].reshape(*xb.shape[:-1], decim // 2, sb // decim).sum(-2)
    folded = torch.cat([folded[..., :1] - xb[..., :1] + xb[..., half:], folded[..., 1:]], -1)
    return torch.fft.ifft(folded).abs() / decim


def band_features(x, soundfield, specs, pad):
    """Run the front end once per band and return, per spec (sb, sh, with_rms),
    the decimated envelopes (C, L, 53, sb/32), the block RMS (C, L, 53) or None,
    and block times. `x` (C, n) is already at 48 kHz, faded in and zero-padded at
    the start by `pad`; each spec slices its own start padding (sb) from the same
    filter output, which is exact because the filters are causal. Each band is
    checkpointed, so backward recomputes it instead of storing its blocks."""
    n = x.shape[-1]
    # Circular convolution over n + 2^16 samples: the impulse-response tail past
    # 2^16 samples is below 1e-60 of its peak even for the narrowest filter.
    nfft = _fast_len(n + 65536)
    X = torch.fft.rfft(x, nfft) * _ear_response(_omega(nfft, x.device), soundfield)

    def one_band(X, band):
        y = torch.fft.irfft(X * _band_response(_omega(nfft, X.device), band), nfft)[..., :n]
        outs = []
        for sb, sh, with_rms in specs:
            blocks, _ = segment(y[..., pad - sb:], sb, sh)
            env = envelope(blocks)
            outs.append(env)
            outs.append(block_rms(blocks) if with_rms else env.new_zeros(()))
        return tuple(outs)

    per_band = [
        checkpoint(one_band, X, b, use_reentrant=False) if X.requires_grad else one_band(X, b)
        for b in range(N_BANDS)
    ]
    results = []
    for j, (sb, sh, with_rms) in enumerate(specs):
        env = torch.stack([pb[2 * j] for pb in per_band], -2)
        rms = torch.stack([pb[2 * j + 1] for pb in per_band], -1) if with_rms else None
        _, t = segment(x.new_zeros(n - pad + sb), sb, sh)
        results.append((env, rms, t))
    return results


def prepare(x, fs, pad):
    """(C, n) or (n,) Pa at fs -> float64 (C, n48 + pad) at 48 kHz, faded in
    and zero-padded at the start, plus the unpadded length."""
    x = torch.as_tensor(x)
    x = (x if x.ndim == 2 else x[None]).to(torch.float64)
    x = fade_in(resample(x, fs))
    return torch.nn.functional.pad(x, (pad, 0)), x.shape[-1]


def pchip(t, y, tq):
    """Piecewise cubic Hermite interpolation of y (..., len(t)) along its last
    axis at tq, with the Fritsch-Carlson derivatives and end conditions of
    SciPy's PchipInterpolator (clauses 7.1.7, 9.1.11)."""
    h = (t[1:] - t[:-1]).to(y)
    delta = (y[..., 1:] - y[..., :-1]) / h
    w1 = 2 * h[1:] + h[:-1]
    w2 = h[1:] + 2 * h[:-1]
    d0, d1 = delta[..., :-1], delta[..., 1:]
    same = (d0 * d1) > 0
    hm = safe_div(w1 + w2, torch.where(same, w1 / torch.where(same, d0, 1.0) + w2 / torch.where(same, d1, 1.0), 0.0))
    inner = torch.where(same, hm, 0.0)

    def edge(h0, h1, m0, m1):
        d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
        d = torch.where(torch.sign(d) != torch.sign(m0), 0.0, d)
        return torch.where((torch.sign(m0) != torch.sign(m1)) & (d.abs() > 3 * m0.abs()), 3 * m0, d)

    if y.shape[-1] > 2:
        dl = edge(h[0], h[1], delta[..., 0], delta[..., 1])
        dr = edge(h[-1], h[-2], delta[..., -1], delta[..., -2])
        deriv = torch.cat([dl[..., None], inner, dr[..., None]], -1)
    else:
        deriv = torch.cat([delta, delta], -1)
    tq = tq.to(y)
    idx = torch.clamp(torch.searchsorted(t.to(y), tq, right=True) - 1, 0, len(t) - 2)
    hk = h[idx]
    s = (tq - t.to(y)[idx]) / hk
    h00 = (1 + 2 * s) * (1 - s) ** 2
    h10 = s * (1 - s) ** 2
    h01 = s * s * (3 - 2 * s)
    h11 = s * s * (s - 1)
    return (h00 * y[..., idx] + h10 * hk * deriv[..., idx]
            + h01 * y[..., idx + 1] + h11 * hk * deriv[..., idx + 1])


def moving_median(x, n):
    """Centred moving median of odd length n along the last axis; near the ends
    the window shrinks to the samples that exist (clause 9.1.3.2, note 37)."""
    h = n // 2
    t = x.shape[-1]
    srt = torch.nn.functional.pad(x, (h, h), value=torch.inf).unfold(-1, n, 1).sort(-1).values
    i = torch.arange(t, device=x.device)
    cnt = torch.clamp(i, max=h) + torch.clamp(t - 1 - i, max=h) + 1
    lo = ((cnt - 1) // 2).expand(*x.shape[:-1], t)[..., None]
    hi = (cnt // 2).expand(*x.shape[:-1], t)[..., None]
    return (srt.gather(-1, lo) + srt.gather(-1, hi))[..., 0] / 2


def calibrate(est, c, median_len=None):
    """Band-distribution exponent and calibration, est (..., T, 53) >= 0:
    Formulae (104)-(108) for roughness; with median_len, (163)-(167) for
    fluctuation strength, whose B-hat adds epsilon to its denominator (165) and
    is median-smoothed."""
    rms = safe_sqrt((est**2).mean(-1))
    if median_len is None:
        b = safe_div(rms, est.mean(-1))  # (106)
    else:
        b = moving_median(rms / (est.mean(-1) + torch.finfo(est.dtype).eps), median_len)  # (165)
    e = 0.37106 * (torch.tanh(1.6407 * (b - 2.5804)) + 1) * 0.5 + 0.58449
    return c * safe_pow(est, e[..., None])


def smooth(x, tau_rise, tau_fall):
    """Formulae (109), (110) / (168): first-order low-pass along axis -2 at
    50 Hz, rise and fall constants chosen per sample."""
    a_r, a_f = math.exp(-1 / (50 * tau_rise)), math.exp(-1 / (50 * tau_fall))
    out = [x[..., 0, :]]
    for t in range(1, x.shape[-2]):
        xt, prev = x[..., t, :], out[-1]
        a = torch.where(xt >= prev, a_r, a_f)
        out.append(xt * (1 - a) + prev * a)
    return torch.stack(out, -2)


def finish(spec, skip, binaural):
    """Clauses 7.1.8-7.1.11 / 9.1.12-9.1.15 from the specific values spec
    (C, T, 53)."""
    if binaural and spec.shape[0] == 2:
        spec = safe_sqrt((spec[0] ** 2 + spec[1] ** 2) / 2)[None]  # (112) / (170)
    total = DZ * spec.sum(-1)  # (111) / (169)
    single = torch.quantile(total[..., skip:], 0.9, dim=-1)
    return {
        "specific": spec.squeeze(0) if spec.shape[0] == 1 else spec,
        "specific_avg": spec[..., skip:, :].mean(-2).squeeze(0) if spec.shape[0] == 1 else spec[..., skip:, :].mean(-2),
        "time_dependent": total.squeeze(0) if total.shape[0] == 1 else total,
        "single": single.squeeze(0) if single.shape[0] == 1 else single,
        "time": torch.arange(total.shape[-1], dtype=torch.float64) / 50,
        "band_centre_freqs": FC.clone(),
    }
