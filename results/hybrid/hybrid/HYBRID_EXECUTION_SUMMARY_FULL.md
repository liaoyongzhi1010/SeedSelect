# Hybrid Verification Execution Summary

## GSO (in-domain, test split)

| Method | alpha | lambda | Gap Closed | Improvement | Win Rate | Est. ms | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot | - | - | 22.14% | 1.46% | 44.7% | 40.0 | 300 |
| Learned | - | - | 68.86% | 4.53% | 58.0% | 444.0 | 300 |
| Hybrid#1 | 0.4 | 0.3 | 44.89% | 2.95% | 52.7% | 201.6 | 300 |
| Hybrid#2 | 0.3 | 0.1 | 44.78% | 2.95% | 52.3% | 161.2 | 300 |
| Hybrid#3 | 0.3 | 0.3 | 44.43% | 2.92% | 53.3% | 161.2 | 300 |

## Omni (out-of-domain)

| Method | alpha | lambda | Gap Closed | Improvement | Win Rate | Est. ms | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot | - | - | 43.68% | 6.31% | 62.0% | 80.0 | 100 |
| Learned | - | - | 41.96% | 6.06% | 60.0% | 888.0 | 100 |
| Hybrid#1 | 0.3 | 0.5 | 53.44% | 7.72% | 64.0% | 322.4 | 100 |
| Hybrid#2 | 0.3 | 0.1 | 50.57% | 7.30% | 65.0% | 322.4 | 100 |
| Hybrid#3 | 0.5 | 0.1 | 49.05% | 7.08% | 67.0% | 484.0 | 100 |

## Recommended Config

- alpha=0.3, lambda=0.5 (avg gap: 48.53%)
