# ecma418-torch

Roughness (clause 7) and fluctuation strength (clause 9) of
[ECMA-418-2, 4th edition, June 2025](https://ecma-international.org/wp-content/uploads/ECMA-418-2_4th_edition_june_2025.pdf)
(Sottek Hearing Model) in PyTorch, differentiable end to end.

```python
from ecma418_torch import roughness, fluctuation_strength

r = roughness(pressure_pa, 44100)             # (n,) or (2, n), pascals, any rate
f = fluctuation_strength(pressure_pa, 44100)
(r["single"] + f["single"]).backward()
```

Each call returns `specific` (per 50 Hz frame and band), `specific_avg`,
`time_dependent`, `single` (the 90th-percentile value), `time` and
`band_centre_freqs`. Two channels are combined binaurally (Formulae 112, 170).

## Agreement

| | this package | reference |
|---|---|---|
| roughness, 1 kHz 70 Hz AM 60 dB, 10 s | 0.9999987 asper | 1 asper by definition |
| specific roughness average, band 17 | 0.37348 | 0.374 (ArtemiS v17) |
| fluctuation strength, 1 kHz 4 Hz AM 60 dB, 10 s | 0.99579 vacil | 1 vacil by definition, 1.00 (ArtemiS v17) |
| specific fluctuation average, band 17 | 0.38876 | 0.395 (ArtemiS v17) |

The ArtemiS v17 values are the ones quoted in the tests of
[sottek-hearing-model](https://github.com/mlotinga/sottek-hearing-model).
Against that implementation run as an oracle (`tests/fixtures`), roughness
agrees to 2e-5 asper; the residual is its use of unrounded outer-ear filter
coefficients where this package uses Table 1 as printed. Fluctuation strength
agrees block by block to 2e-5 once the same three choices are made (below and
`test_fluctuation_matches_reference`).

## Differentiability

Every stage is a torch operation in float64. Peak positions, window limits,
candidate sets and harmonic orders are piecewise constant in the input, so
their derivative is zero wherever it exists, and the values they select stay in
the graph. The gradient is the derivative of the metric wherever the metric is
differentiable; the standard's own thresholds make it piecewise smooth.
`tests/test_gradients.py` checks this with `torch.autograd.gradcheck`, with the
second-order decay of the Taylor remainder, and with exact zeros on silence and
unmodulated noise.

## Printed formulas

- (127) The phase term is `exp(-j π f_n (n1 + n2))`, the exact DFT of a
  windowed exponential; the print has `2π`.
- (144) The centroid uses the 0-based index `k` as written, without the printed
  `- 1`.
- `newton_cap` defaults to the printed 2e-4 Hz of Formula (152), and
  `require_fundamental=False` follows the printed clause 9.1.8. The reference
  implementation uses 0.2 and True, and 1e-12 for ε0 where this package uses
  machine epsilon as the standard defines it.

The optional entropy weighting (7.1.6) needs a rotational-speed signal and is
not implemented.

## Cost

240 s of stereo at 44.1 kHz, roughness plus fluctuation strength, RTX 3090 Ti:
forward 22 s, backward 32 s, 10.6 GiB peak.
