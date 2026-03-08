# Hybrid Runtime Wall-Clock Benchmark

Selection-stage latency benchmark with precomputed candidate scores/features (candidate generation cost excluded).

- Generated at: 2026-03-07T17:14:38Z
- Device: cuda
- Warmup runs: 1, measure runs: 6

## GSO-300

| Method | Median ms / object | Median ms / candidate | Mean K | N objects |
|---|---:|---:|---:|---:|
| Zero-shot | 0.0007 | 0.0002 | 4.00 | 300 |
| Learned-only | 0.3233 | 0.0808 | 4.00 | 300 |
| Hybrid | 2.4524 | 0.6131 | 4.00 | 300 |

## OmniObject3D-100

| Method | Median ms / object | Median ms / candidate | Mean K | N objects |
|---|---:|---:|---:|---:|
| Zero-shot | 0.0011 | 0.0001 | 8.00 | 100 |
| Learned-only | 0.3333 | 0.0417 | 8.00 | 100 |
| Hybrid | 2.4462 | 0.3058 | 8.00 | 100 |

