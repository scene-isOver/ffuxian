# PPO-CIS two-stage cascade — results

Stage 1 (throughput): `tiny_bert`, `distil_roberta`. Stage 2 (accuracy): `bert`, `roberta_plain`, `tke_bert`.

## Classifiers

| Classifier | Stage | Acc | F1 | Latency (ms) | Throughput |
|---|---:|---:|---:|---:|---:|

## Test-set comparison

| Method | Acc | Prec | Rec | F1 | Calls | Latency (ms) | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny_bert | 0.7867 | 0.6945 | 0.8176 | 0.7511 | 1.00 | 0.23 | 4267.6 |
| distil_roberta | 0.7995 | 0.7017 | 0.8528 | 0.7699 | 1.00 | 0.17 | 5733.5 |
| bert | 0.8175 | 0.7268 | 0.8591 | 0.7874 | 1.00 | 0.96 | 1037.8 |
| roberta_plain | 0.8192 | 0.7280 | 0.8629 | 0.7898 | 1.00 | 0.92 | 1087.9 |
| tke_bert | 0.8171 | 0.7273 | 0.8562 | 0.7865 | 1.00 | 0.93 | 1077.7 |
| static_sequential | 0.8171 | 0.7273 | 0.8562 | 0.7865 | 5.00 | 3.22 | 310.6 |
| static_cascade | 0.8190 | 0.7501 | 0.8098 | 0.7788 | 1.61 | 0.74 | 1348.8 |
| majority_vote | 0.8202 | 0.7286 | 0.8653 | 0.7911 | 5.00 | 3.22 | 310.6 |
| soft_vote | 0.8192 | 0.7284 | 0.8620 | 0.7896 | 5.00 | 3.22 | 310.6 |
| PPO-CIS | 0.7940 | 0.6873 | 0.8740 | 0.7695 | 1.50 | 0.63 | 1576.3 |
