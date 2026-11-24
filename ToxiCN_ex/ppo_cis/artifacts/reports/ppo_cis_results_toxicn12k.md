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

## Test-set comparison (toxicn12k)

| Method | Acc | Prec | Rec | F1 | Calls | Latency (ms) | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny_bert | 0.7112 | 0.7321 | 0.7148 | 0.7234 | 1.00 | 0.22 | 4582.7 |
| distil_roberta | 0.7373 | 0.7556 | 0.7431 | 0.7493 | 1.00 | 0.13 | 7799.2 |
| bert | 0.7900 | 0.7966 | 0.8091 | 0.8028 | 1.00 | 0.91 | 1095.2 |
| roberta_plain | 0.7917 | 0.8086 | 0.7934 | 0.8010 | 1.00 | 0.93 | 1076.1 |
| tke_bert | 0.7701 | 0.7902 | 0.7690 | 0.7795 | 1.00 | 0.91 | 1094.1 |
| static_sequential | 0.7701 | 0.7902 | 0.7690 | 0.7795 | 5.00 | 3.10 | 322.3 |
| static_cascade | 0.7535 | 0.8182 | 0.6858 | 0.7462 | 1.80 | 0.82 | 1221.2 |
| majority_vote | 0.7813 | 0.7965 | 0.7871 | 0.7918 | 5.00 | 3.10 | 322.3 |
| soft_vote | 0.7780 | 0.7952 | 0.7808 | 0.7880 | 5.00 | 3.10 | 322.3 |
| PPO-CIS | 0.7407 | 0.7407 | 0.7832 | 0.7614 | 1.58 | 0.66 | 1518.3 |
