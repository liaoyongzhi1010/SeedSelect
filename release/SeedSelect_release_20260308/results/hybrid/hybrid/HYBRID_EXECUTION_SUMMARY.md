# Hybrid Verification Execution Summary

## GSO (in-domain, test split)

| Method | alpha | lambda | Gap Closed | Improvement | Win Rate | Est. ms | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot | - | - | 21.64% | 1.05% | 40.0% | 40.0 | 30 |
| Learned | - | - | 15.11% | 0.74% | 43.3% | 444.0 | 30 |
| Hybrid#1 | 0.3 | 0.1 | 28.80% | 1.40% | 43.3% | 161.2 | 30 |
| Hybrid#2 | 0.3 | 0.3 | 28.80% | 1.40% | 43.3% | 161.2 | 30 |
| Hybrid#3 | 0.3 | 0.5 | 28.80% | 1.40% | 43.3% | 161.2 | 30 |

## Omni (out-of-domain)

| Method | alpha | lambda | Gap Closed | Improvement | Win Rate | Est. ms | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot | - | - | 43.68% | 6.31% | 62.0% | 80.0 | 100 |
| Learned | - | - | 41.96% | 6.06% | 60.0% | 888.0 | 100 |
| Hybrid#1 | 0.4 | 0.1 | 51.14% | 7.39% | 65.0% | 403.2 | 100 |
| Hybrid#2 | 0.4 | 0.3 | 51.14% | 7.39% | 65.0% | 403.2 | 100 |
| Hybrid#3 | 0.4 | 0.5 | 51.14% | 7.39% | 65.0% | 403.2 | 100 |

## Recommended Config

- alpha=0.4, lambda=0.1 (avg gap: 39.97%)
