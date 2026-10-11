"""ECMA-418-2 (4th ed., 2025) clause 9.1: fluctuation strength. Formula
numbers refer to the standard. Window limits, peak and minimum positions, the
candidate set and the harmonic order are piecewise constant in the input and
are chosen without gradient; every amplitude, rate and error they select is
computed in the graph.

Two printed formulas are read as the mathematics requires:
  (127) the phase term is exp(-j*pi*f_n*(n1 + n2)), the exact DFT of a
        rectangular-windowed exponential; the print has 2*pi.
  (144) the three-point centroid is taken on the 0-based index k it is written
        in, without the printed "- 1" that shifts every rate down one bin.
"""

import math

import torch
from torch.utils.checkpoint import checkpoint

from .hearing import (
    FC,
    LTQ,
    N_BANDS,
    band_features,
    calibrate,
    finish,
    loudness_nonlinearity,
    moving_median,
    pchip,
    prepare,
    safe_pow,
    safe_sqrt,
    smooth,
)

SB, SH = 65536, 16384
NK = 2048  # envelope block length at 1500 Hz
RS = 1500.0
DF = RS / NK
KMAX = 49  # k = 0..48, clause 9.1.4
C_F = 0.003840572
SKIP = 36  # l50 <= 35 discarded, clauses 9.1.12 and 9.1.14
EPS = torch.finfo(torch.float64).eps
F_GRID = 0.25 * 2 ** ((torch.arange(1, 17, dtype=torch.float64) - 2) / 3)  # clause 9.1.5


def _round(x):
    """Nearest integer, ties away from zero (note 42)."""
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


def analysis_windows(env, chunk=1024):
    """Clause 9.1.3 for envelopes (M, 2048): returns the first and last index
    of the rectangular analysis window and whether the block carries a
    usable envelope (False = quieter period, window all zero)."""
    if env.shape[0] > chunk:
        parts = [analysis_windows(env[i:i + chunk], chunk) for i in range(0, env.shape[0], chunk)]
        return tuple(torch.cat(p) for p in zip(*parts))
    with torch.no_grad():
        m, n = env.shape
        idx = torch.arange(n, device=env.device)
        ps = torch.round(moving_median(env, 33) * 1e8) / 1e8
        pw = ps * ((idx >= 64) & (idx <= n - 1 - 64))
        pmax = pw.amax(-1)
        thr = (torch.round(0.01 * pmax * 1e8) / 1e8)[:, None]
        above = pw >= thr
        n1 = torch.where(above, idx, n).amin(-1)
        n2 = torch.where(above, idx, -1).amax(-1)
        quiet = ~above & (idx >= n1[:, None]) & (idx <= n2[:, None])
        cs = quiet.long().cumsum(-1)
        run = cs - torch.where(~quiet, cs, 0).cummax(-1).values
        qlen, qe = run.max(-1)  # (9.1.3.4) longest quieter period, first on ties
        qb = qe - qlen + 1
        found = qlen > 320
        keep_left = (qb - n1) > (n2 - qe)  # (9.1.3.5), the +-64 terms cancel
        n2 = torch.where(found & keep_left, qb - 64, n2)
        n1 = torch.where(found & ~keep_left, qe + 64, n1)
        ok = (pmax > 5e-6) & (n2 - n1 + 1 >= 320) & (n2 >= 511)  # (9.1.3.2), (9.1.3.6)
        # (9.1.3.6) relative standard deviation of the linear-regression residuals
        sel = (idx >= n1[:, None] + 10) & (idx <= n2[:, None] - 10)
        c = sel.sum(-1).clamp(min=3).to(env.dtype)
        t = idx.to(env.dtype)
        tm = (t * sel).sum(-1) / c
        ym = (pw * sel).sum(-1) / c
        dt = (t - tm[:, None]) * sel
        slope = (dt * (pw - ym[:, None])).sum(-1) / (dt**2).sum(-1).clamp(min=1)
        resid = (pw - ym[:, None] - slope[:, None] * (t - tm[:, None])) * sel
        rstd = torch.sqrt((resid**2).sum(-1) / (c - 1))
        ok &= (ym != 0) & (rstd >= 1e-3 * ym.abs())
    return n1, n2, ok


def _window_response(f, s, length):
    """Formula (127): DFT bins k = 0..48 of exp(j 2 pi f n / 1500) over the
    window, for rates f (M, J). The Dirichlet kernel is written with sinc,
    which is exact at f_n = 0 and needs no epsilon."""
    fn = torch.arange(KMAX, dtype=f.dtype, device=f.device) / NK - f[..., None] / RS
    lng = length[:, None, None]
    return torch.exp(-1j * math.pi * fn * s[:, None, None]) * (lng * torch.sinc(fn * lng) / torch.sinc(fn))


def hsa(f, on, spec, phi, s, length):
    """Formulae (123)-(135): least-squares fit of a constant plus one line pair
    per rate in f (M, J) where `on`, to the spectrum bins k < K_L. Returns the
    constant, the two-sided complex line amplitudes (M, J), and the error."""
    m, j = f.shape
    fz = torch.where(on, f, 0.0)
    k_l = torch.clamp(_round(fz.detach().amax(-1) / DF) + 8, min=17, max=KMAX)  # (125)
    kmask = (torch.arange(KMAX, device=f.device) < k_l[:, None]).to(f.dtype)
    w0 = _window_response(torch.zeros_like(f[:, :1]), s, length)
    wp, wm = _window_response(fz, s, length), _window_response(-fz, s, length)
    onb = on[..., None]
    basis = torch.cat([w0, torch.where(onb, wp + wm, 0), torch.where(onb, 1j * (wp - wm), 0)], 1) * kmask[:, None]
    br, bi = basis.real, basis.imag
    a = br @ br.transpose(1, 2) + bi @ bi.transpose(1, 2)  # (131)
    b = (br @ spec.real[..., None] + bi @ spec.imag[..., None])[..., 0]  # (134)
    off = torch.cat([torch.zeros_like(on[:, :1]), ~on, ~on], 1)
    x = torch.linalg.solve(a + torch.diag_embed(off.to(a.dtype)), b)
    err = (phi * kmask).sum(-1) - (b * x).sum(-1)  # (135) at the solution
    return x[:, 0], torch.complex(x[:, 1:1 + j], x[:, 1 + j:]), err


def _hsa1(f, spec, phi, s, length):
    """One line pair at each of the rates f (M, R): returns (const, line, error), each (M, R)."""
    m, r = f.shape
    spec, phi, s, length = (v.repeat_interleave(r, 0) for v in (spec, phi, s, length))
    c, p, e = hsa(f.reshape(-1, 1), torch.ones(m * r, 1, dtype=torch.bool, device=f.device), spec, phi, s, length)
    return c.reshape(m, r), p.reshape(m, r), e.reshape(m, r)


def weight_lh(f, fc):
    """Formula (148)."""
    fm = 4.8659
    fs = torch.where(f > 0, f, 1.0)
    u = fs / fm - fm / fs
    band = 1 / (1 + 0.092623 * torch.log2(fc / 1000).abs() ** 1.24)
    g = torch.where(fs <= fm, 1 / (1 + (u * 0.33048) ** 2) ** 0.85902, band / (1 + (u * 0.21792) ** 2) ** 4.6728)
    return torch.where(f > 0, g, 0.0)


def _local_maxima(phi, cap):
    """Strict local maxima of phi (M, 49) at k = 1..47 at or above `cap` (M,),
    ascending, as (indices (M, 24), valid)."""
    with torch.no_grad():
        pk = torch.zeros_like(phi, dtype=torch.bool)
        pk[:, 1:-1] = (phi[:, 1:-1] > phi[:, :-2]) & (phi[:, 1:-1] > phi[:, 2:])
        pk &= phi >= cap[:, None]
        k = torch.where(pk, torch.arange(KMAX, device=phi.device), KMAX).sort(-1).values[:, :24]
        return k.clamp(max=KMAX - 2), k < KMAX


def block_fluctuation(spec, s, length, band, newton_cap, require_fundamental):
    """Clauses 9.1.5-9.1.10 for M usable blocks: spectrum bins (M, 49), window
    sum n1+n2 and length, band index (M,). Returns A-hat (157), the
    harmonic-complex power (159) and N'_HSA (161), each (M,)."""
    phi = spec.real**2 + spec.imag**2  # (122)
    fc = FC.to(spec.device)[band]
    args = (spec, phi, s, length)

    # 9.1.5: prominent spectral line pairs.
    km, on_m = _local_maxima(phi, torch.clamp(0.001 * phi[:, 0].detach(), min=0.15))  # (143)
    nb = torch.stack([phi.gather(-1, km + d) for d in (-1, 0, 1)], -1)
    den = torch.where(on_m, nb.sum(-1), 1.0)
    fp = ((km[..., None] + torch.tensor([-1, 0, 1], device=km.device)) * nb).sum(-1) / den * DF  # (144)
    fp = torch.where(on_m, fp, 0.0)
    _, _, e_grid = _hsa1(F_GRID.to(spec.device).expand(spec.shape[0], -1), *args)
    with torch.no_grad():
        is_min = torch.zeros_like(e_grid, dtype=torch.bool)
        is_min[:, 1:-1] = (e_grid[:, 1:-1] < e_grid[:, :-2]) & (e_grid[:, 1:-1] < e_grid[:, 2:])
        has_min = is_min.any(-1)
        f_min = F_GRID.to(spec.device)[torch.where(is_min, e_grid, torch.inf).argmin(-1)]
        dup = on_m & ((f_min[:, None] - fp).abs() < 1.25 * DF) & has_min[:, None]  # (145)
    f1 = torch.cat([fp, f_min[:, None].to(fp)], 1)
    on1 = torch.cat([on_m & ~dup, has_min[:, None]], 1)
    on2 = torch.cat([on_m, torch.zeros_like(has_min[:, None])], 1)
    c1, p1, e1 = hsa(f1, on1, *args)
    c2, p2, e2 = hsa(f1, on2, *args)
    use2 = (~has_min | (dup.any(-1) & (e2 < e1)))[:, None]
    on = torch.where(use2, on2, on1)
    p = torch.where(use2, p2, p1)
    amp = p.real**2 + p.imag**2
    with torch.no_grad():
        on = on & (amp > 0.05 * torch.where(on, amp, 0.0).amax(-1, keepdim=True))  # (146)
    live = on.any(-1)

    # 9.1.6, 9.1.7: weighted power spectrum and fine tuning of its maximum.
    fcb = fc[:, None]
    at = amp * weight_lh(f1, fcb)  # (147)
    i_max = torch.where(on, at, -torch.inf).argmax(-1, keepdim=True)
    x0 = torch.where(live, f1.gather(-1, i_max)[:, 0], 1.0)
    x, done = x0, torch.zeros_like(live)
    for _ in range(40):
        _, _, e = _hsa1(torch.stack([x - 1e-5, x, x + 1e-5], -1), *args)
        d1 = (e[:, 2] - e[:, 0]) / 2e-5  # (149)
        d2 = (e[:, 2] - 2 * e[:, 1] + e[:, 0]) / 1e-10  # (150)
        dx = 0.25 * torch.sign(d1) * torch.minimum(d1.abs() / (d2.abs() + EPS), torch.full_like(d1, newton_cap))
        x = torch.where(done, x, x - dx)  # (151), (152)
        done = done | (dx.abs() <= 1e-7)
    tuned = (x - x0).abs() <= 1.25 * DF
    f_opt = torch.where(tuned, x, x0)
    _, p_opt, _ = _hsa1(f_opt[:, None], *args)
    first = torch.arange(f1.shape[1], device=f1.device) == i_max
    f1 = torch.where(first & tuned[:, None], f_opt[:, None], f1)
    p = torch.where(first & tuned[:, None], p_opt, p)
    live = live & (f_opt >= 0.125)
    f_opt = torch.where(live, f_opt, 1.0)
    at = (p.real**2 + p.imag**2) * weight_lh(f1, fcb)

    # 9.1.8: harmonic analysis over assumed orders 1..3 of f_opt.
    with torch.no_grad():
        orders = torch.arange(1, 4, device=f1.device, dtype=f1.dtype)
        fund = f_opt[:, None, None] / orders[:, None]  # (M, 3, 1)
        r = _round(f1[:, None, :].detach() / fund)
        r = torch.where(r > 5, 0.0, r)  # (153)
        dev = (f1[:, None, :].detach() / (torch.where(r > 0, r, 1) * fund) - 1).abs()
        mem = on[:, None, :] & (r > 0) & (dev < 0.04)  # (154)
        if require_fundamental:
            mem &= (mem & (r == 1)).any(-1, keepdim=True)
        energy = torch.where(mem, at.detach()[:, None, :], 0.0).sum(-1)  # (155)
        o_max = energy.argmax(-1)
        rows = torch.arange(len(o_max), device=o_max.device)
        r_max = r[rows, o_max]
        m_max = mem[rows, o_max]
        present = torch.stack([(m_max & (r_max == h)).any(-1) for h in range(1, 6)], -1)
    f_fund = f_opt / (o_max + 1).to(f_opt.dtype)  # (156)
    rates = f_fund[:, None] * torch.arange(1, 6, device=f1.device, dtype=f1.dtype)
    c_h, p_h, _ = _hsa1(rates, *args)
    n_present = present.sum(-1).clamp(min=1)
    const = torch.where(present, c_h, 0.0).sum(-1) / n_present
    a_h = torch.where(present, p_h.real**2 + p_h.imag**2, 0.0)
    at_h = a_h * weight_lh(rates, fcb)

    # 9.1.9, 9.1.10.
    sum_at = at_h.sum(-1)
    cog = (rates * at_h).sum(-1) / (sum_at + EPS)
    a_hat = (1 + 0.79577 * safe_pow((cog - f_opt).abs(), 0.43461)) * sum_at  # (157), (158)
    power = const**2 + 2 * a_h.sum(-1)
    n_hsa = torch.relu(loudness_nonlinearity(safe_sqrt(0.5 * power)) - LTQ.to(fc.device)[band])  # (160), (161)
    zero = torch.zeros_like(a_hat)
    return torch.where(live, a_hat, zero), torch.where(live, power, zero), torch.where(live, n_hsa, zero)


def fluctuation_strength(x, fs, soundfield="free", binaural=True, pair_chunk=2048,
                         newton_cap=2e-4, require_fundamental=False):
    """ECMA-418-2 fluctuation strength of `x`, sound pressure in Pa, shape (n,)
    or (C, n) with C = 1 or 2, sampled at `fs`. Returns a dict with the
    specific fluctuation strength F'(l50, z) ("specific", vacil/Bark_HMS), its
    time average ("specific_avg"), F(l50) ("time_dependent", vacil), the
    90th-percentile single value F ("single") and the 50 Hz time axis.

    `newton_cap` is the step limit of Formula (152), printed as 2e-4 Hz.
    `require_fundamental` admits an assumed order in clause 9.1.8 only if a
    component lies at its fundamental, which the printed text does not state.
    The GPL reference implementation (mlotinga/sottek-hearing-model) uses
    newton_cap=0.2 and require_fundamental=True."""
    xp, n = prepare(x, fs, SB)
    (env, _, t), = band_features(xp, soundfield, [(SB, SH, False)], SB)
    c, nl = env.shape[:2]
    flat = env.reshape(-1, NK)
    n1, n2, ok = analysis_windows(flat.detach())
    use = ok.nonzero()[:, 0]
    idx = torch.arange(NK, device=flat.device)
    fl = flat[use]
    win = ((idx >= n1[use, None]) & (idx <= n2[use, None])).to(fl.dtype)
    spec = torch.fft.rfft(fl * win)[:, :KMAX]  # (121)
    s = (n1 + n2)[use].to(fl.dtype)
    length = (n2 - n1 + 1)[use].to(fl.dtype)
    band = torch.arange(N_BANDS, device=fl.device).repeat(c * nl)[use]

    def run(spec, s, length, band):
        return torch.stack(block_fluctuation(spec, s, length, band, newton_cap, require_fundamental))

    parts = []
    for i in range(0, len(use), pair_chunk):
        a = (spec[i:i + pair_chunk], s[i:i + pair_chunk], length[i:i + pair_chunk], band[i:i + pair_chunk])
        parts.append(checkpoint(run, *a, use_reentrant=False) if spec.requires_grad else run(*a))
    out = torch.cat(parts, 1) if parts else spec.real.new_zeros(3, 0)
    full = out.new_zeros(3, c * nl * N_BANDS).index_copy(1, use, out).reshape(3, c, nl, N_BANDS)
    a_hat, power, n_hsa = full
    n_max = n_hsa.amax(-1, keepdim=True)
    a = a_hat * NK * n_hsa**2 / ((n_max + EPS) * (power + EPS))  # (159)
    a = torch.where(a >= 5.2519, a, 0.0)
    t50 = torch.arange(n * 50 // 48000 + 1, dtype=torch.float64) / 50
    est = torch.relu(pchip(t, a.transpose(-1, -2), t50)).transpose(-1, -2)
    spec_f = smooth(calibrate(est, C_F, median_len=71), 0.75, 0.75)  # (163)-(168)
    return finish(spec_f, SKIP, binaural)
