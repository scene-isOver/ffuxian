# PPO-CIS two-stage cascade — results

Stage 1 (throughput): `tiny_bert`, `distil_roberta`. Stage 2 (accuracy): `bert`, `roberta_plain`, `tke_bert`.

## Classifiers

| Classifier | Stage | Acc | F1 | Latency (ms) | Throughput |
|---|---:|---:|---:|---:|---:|
| tiny_bert | 1 | 0.7833 | 0.7543 | 0.22 | 4582.7 |
| distil_roberta | 1 | 0.7966 | 0.7686 | 0.13 | 7799.2 |
| bert | 2 | 0.8112 | 0.7874 | 0.91 | 1095.2 |
| roberta_plain | 2 | 0.8194 | 0.7919 | 0.93 | 1076.1 |
| tke_bert | 2 | 0.8198 | 0.7912 | 0.91 | 1094.1 |

## Test-set comparison (cold)

| Method | Acc | Prec | Rec | F1 | Calls | Latency (ms) | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny_bert | 0.7833 | 0.6811 | 0.8451 | 0.7543 | 1.00 | 0.22 | 4582.7 |
| distil_roberta | 0.7966 | 0.6957 | 0.8586 | 0.7686 | 1.00 | 0.13 | 7799.2 |
| bert | 0.8114 | 0.7073 | 0.8885 | 0.7876 | 1.00 | 0.91 | 1095.2 |
| roberta_plain | 0.8194 | 0.7245 | 0.8731 | 0.7919 | 1.00 | 0.93 | 1076.1 |
| tke_bert | 0.8198 | 0.7271 | 0.8678 | 0.7912 | 1.00 | 0.91 | 1094.1 |
| static_sequential | 0.8198 | 0.7271 | 0.8678 | 0.7912 | 5.00 | 3.10 | 322.3 |
| static_cascade | 0.8166 | 0.7375 | 0.8287 | 0.7805 | 1.62 | 0.72 | 1384.9 |
| majority_vote | 0.8183 | 0.7196 | 0.8818 | 0.7925 | 5.00 | 3.10 | 322.3 |
| soft_vote | 0.8188 | 0.7213 | 0.8793 | 0.7925 | 5.00 | 3.10 | 322.3 |
| PPO-CIS | 0.7921 | 0.6847 | 0.8740 | 0.7679 | 1.51 | 0.60 | 1675.3 |
