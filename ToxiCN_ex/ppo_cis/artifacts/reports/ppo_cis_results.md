# PPO-CIS two-stage cascade — results

Stage 1 (throughput): `tiny_bert`, `distil_roberta`. Stage 2 (accuracy): `bert`, `roberta_plain`, `tke_bert`.

## Classifiers

| Classifier | Stage | Acc | F1 | Latency (ms) | Throughput |
|---|---:|---:|---:|---:|---:|
| tiny_bert | 1 | 0.7911 | 0.7576 | 0.21 | 4829.1 |
| distil_roberta | 1 | 0.8069 | 0.7742 | 0.14 | 7351.4 |
| bert | 2 | 0.8137 | 0.7815 | 0.94 | 1058.8 |
| roberta_plain | 2 | 0.8280 | 0.7977 | 0.93 | 1074.9 |
| tke_bert | 2 | 0.8209 | 0.7924 | 0.92 | 1087.2 |

## Test-set comparison

| Method | Acc | Prec | Rec | F1 | Calls | Latency (ms) | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny_bert | 0.7911 | 0.6971 | 0.8296 | 0.7576 | 1.00 | 0.21 | 4829.2 |
| distil_roberta | 0.8069 | 0.7170 | 0.8412 | 0.7742 | 1.00 | 0.14 | 7351.4 |
| bert | 0.8137 | 0.7257 | 0.8465 | 0.7815 | 1.00 | 0.94 | 1058.8 |
| roberta_plain | 0.8280 | 0.7423 | 0.8620 | 0.7977 | 1.00 | 0.93 | 1074.9 |
| tke_bert | 0.8209 | 0.7285 | 0.8687 | 0.7924 | 1.00 | 0.92 | 1087.2 |
| static_sequential | 0.8209 | 0.7285 | 0.8687 | 0.7924 | 5.00 | 3.14 | 318.7 |
| static_cascade | 0.8154 | 0.7473 | 0.8021 | 0.7737 | 1.56 | 0.69 | 1452.7 |
| majority_vote | 0.8223 | 0.7332 | 0.8620 | 0.7924 | 5.00 | 3.14 | 318.7 |
| soft_vote | 0.8215 | 0.7320 | 0.8620 | 0.7917 | 5.00 | 3.14 | 318.7 |
| PPO-CIS | 0.7866 | 0.6837 | 0.8514 | 0.7584 | 1.50 | 0.66 | 1518.2 |
