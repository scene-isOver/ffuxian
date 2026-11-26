# PPO-CIS two-stage cascade — results

Stage 1 (throughput): `tiny_bert`, `distil_roberta`. Stage 2 (accuracy): `bert`, `roberta_plain`, `tke_bert`.

## Classifiers

| Classifier | Stage | Acc | F1 | Latency (ms) | Throughput |
|---|---:|---:|---:|---:|---:|
| tiny_bert | 1 | 0.7681 | 0.7444 | 0.21 | 4741.5 |
| distil_roberta | 1 | 0.7978 | 0.7681 | 0.07 | 14462.2 |
| bert | 2 | 0.8223 | 0.7914 | 0.90 | 1107.5 |
| roberta_plain | 2 | 0.8240 | 0.7912 | 0.91 | 1104.5 |
| tke_bert | 2 | 0.8215 | 0.7895 | 0.91 | 1095.7 |

## Test-set comparison

| Method | Acc | Prec | Rec | F1 | Calls | Latency (ms) | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny_bert | 0.6988 | 0.7152 | 0.7141 | 0.7146 | 1.00 | 0.21 | 4741.5 |
| distil_roberta | 0.7286 | 0.7518 | 0.7258 | 0.7386 | 1.00 | 0.07 | 14462.2 |
| bert | 0.7813 | 0.8119 | 0.7628 | 0.7866 | 1.00 | 0.90 | 1107.5 |
| roberta_plain | 0.7959 | 0.8268 | 0.7761 | 0.8006 | 1.00 | 0.91 | 1104.5 |
| tke_bert | 0.7763 | 0.8110 | 0.7518 | 0.7803 | 1.00 | 0.91 | 1095.7 |
| static_sequential | 0.7763 | 0.8110 | 0.7518 | 0.7803 | 5.00 | 3.00 | 333.2 |
| static_cascade | 0.7647 | 0.8401 | 0.6850 | 0.7547 | 1.97 | 0.86 | 1159.8 |
| majority_vote | 0.7813 | 0.8124 | 0.7620 | 0.7864 | 5.00 | 3.00 | 333.2 |
| soft_vote | 0.7813 | 0.8114 | 0.7636 | 0.7867 | 5.00 | 3.00 | 333.2 |
| PPO-CIS | 0.7751 | 0.8394 | 0.7101 | 0.7694 | 1.70 | 0.67 | 1488.9 |
