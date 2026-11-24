# PPO-CIS two-stage cascade — results

Stage 1 (throughput): `tiny_bert`, `distil_roberta`. Stage 2 (accuracy): `bert`, `roberta_plain`, `tke_bert`.

## Classifiers

| Classifier | Stage | Acc | F1 | Latency (ms) | Throughput |
|---|---:|---:|---:|---:|---:|
| tiny_bert | 1 | 0.7814 | 0.7416 | 0.23 | 4378.2 |
| distil_roberta | 1 | 0.8006 | 0.7692 | 0.10 | 10054.0 |
| bert | 2 | 0.8190 | 0.7838 | 1.40 | 712.0 |
| roberta_plain | 2 | 0.8204 | 0.7904 | 1.46 | 682.7 |
| tke_bert | 2 | 0.8209 | 0.7924 | 0.92 | 1087.2 |

## Test-set comparison

| Method | Acc | Prec | Rec | F1 | Calls | Latency (ms) | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny_bert | 0.7814 | 0.6932 | 0.7973 | 0.7416 | 1.00 | 0.23 | 4378.2 |
| distil_roberta | 0.8004 | 0.7059 | 0.8446 | 0.7691 | 1.00 | 0.10 | 10054.0 |
| bert | 0.8190 | 0.7396 | 0.8335 | 0.7838 | 1.00 | 1.40 | 712.0 |
| roberta_plain | 0.8204 | 0.7305 | 0.8610 | 0.7904 | 1.00 | 1.46 | 682.7 |
| tke_bert | 0.8209 | 0.7285 | 0.8687 | 0.7924 | 1.00 | 0.92 | 1087.2 |
| static_sequential | 0.8209 | 0.7285 | 0.8687 | 0.7924 | 5.00 | 4.12 | 242.9 |
| static_cascade | 0.8137 | 0.7490 | 0.7920 | 0.7699 | 1.66 | 1.02 | 977.3 |
| majority_vote | 0.8223 | 0.7339 | 0.8600 | 0.7920 | 5.00 | 4.12 | 242.9 |
| soft_vote | 0.8211 | 0.7314 | 0.8620 | 0.7913 | 5.00 | 4.12 | 242.9 |
| PPO-CIS | 0.7959 | 0.6912 | 0.8697 | 0.7703 | 1.53 | 0.56 | 1774.8 |
