# PPO-CIS two-stage cascade — results

Stage 1 (throughput): `tiny_bert`, `distil_roberta`. Stage 2 (accuracy): `bert`, `roberta_plain`, `tke_bert`.

## Classifiers

| Classifier | Stage | Acc | F1 | Latency (ms) | Throughput |
|---|---:|---:|---:|---:|---:|
| tiny_bert | 1 | 0.7869 | 0.7512 | 0.23 | 4267.6 |
| distil_roberta | 1 | 0.7995 | 0.7699 | 0.17 | 5733.5 |
| bert | 2 | 0.8177 | 0.7876 | 0.96 | 1037.8 |
| roberta_plain | 2 | 0.8192 | 0.7898 | 0.92 | 1087.9 |
| tke_bert | 2 | 0.8171 | 0.7865 | 0.93 | 1077.7 |

## Test-set comparison

| Method | Acc | Prec | Rec | F1 | Calls | Latency (ms) | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny_bert | 0.6956 | 0.6675 | 0.8446 | 0.7457 | 1.00 | 0.23 | 4341.8 |
| distil_roberta | 0.7362 | 0.7231 | 0.8116 | 0.7648 | 1.00 | 0.12 | 8222.9 |
| bert | 0.8113 | 0.8279 | 0.8116 | 0.8197 | 1.00 | 0.98 | 1024.2 |
| roberta_plain | 0.8229 | 0.8014 | 0.8838 | 0.8406 | 1.00 | 0.92 | 1083.7 |
| tke_bert | 0.7922 | 0.8022 | 0.8053 | 0.8038 | 1.00 | 0.92 | 1092.6 |
| static_sequential | 0.7922 | 0.8022 | 0.8053 | 0.8038 | 5.00 | 3.17 | 315.8 |
| static_cascade | 0.8001 | 0.8322 | 0.7786 | 0.8045 | 2.08 | 1.02 | 977.4 |
| majority_vote | 0.8038 | 0.7921 | 0.8524 | 0.8212 | 5.00 | 3.17 | 315.8 |
| soft_vote | 0.8092 | 0.7984 | 0.8548 | 0.8256 | 5.00 | 3.17 | 315.8 |
| PPO-CIS | 0.8088 | 0.8159 | 0.8242 | 0.8200 | 1.80 | 0.86 | 1165.5 |
