# PPO-CIS two-stage cascade — results

Stage 1 (throughput): `tiny_bert`, `distil_roberta`. Stage 2 (accuracy): `bert`, `roberta_fusion`, `tke_bert`.

## Classifiers

| Classifier | Stage | Acc | F1 | Latency (ms) | Throughput |
|---|---:|---:|---:|---:|---:|
| tiny_bert | 1 | 0.6956 | 0.7457 | 0.23 | 4341.8 |
| distil_roberta | 1 | 0.7362 | 0.7648 | 0.12 | 8222.9 |
| bert | 2 | 0.8113 | 0.8197 | 0.98 | 1024.2 |
| roberta_fusion | 2 | 0.7404 | 0.7447 | 1.23 | 809.7 |
| tke_bert | 2 | 0.7922 | 0.8038 | 0.92 | 1092.6 |

## Test-set comparison

| Method | Acc | Prec | Rec | F1 | Calls | Latency (ms) | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny_bert | 0.6956 | 0.6675 | 0.8446 | 0.7457 | 1.00 | 0.23 | 4341.8 |
| distil_roberta | 0.7362 | 0.7231 | 0.8116 | 0.7648 | 1.00 | 0.12 | 8222.9 |
| bert | 0.8113 | 0.8279 | 0.8116 | 0.8197 | 1.00 | 0.98 | 1024.2 |
| roberta_fusion | 0.7404 | 0.7750 | 0.7166 | 0.7447 | 1.00 | 1.23 | 809.7 |
| tke_bert | 0.7922 | 0.8022 | 0.8053 | 0.8038 | 1.00 | 0.92 | 1092.6 |
| static_sequential | 0.7922 | 0.8022 | 0.8053 | 0.8038 | 5.00 | 3.48 | 287.5 |
| static_cascade | 0.7947 | 0.8315 | 0.7669 | 0.7979 | 2.13 | 1.09 | 920.8 |
| majority_vote | 0.7860 | 0.7795 | 0.8297 | 0.8038 | 5.00 | 3.48 | 287.5 |
| soft_vote | 0.7959 | 0.7905 | 0.8352 | 0.8122 | 5.00 | 3.48 | 287.5 |
| PPO-CIS | 0.7876 | 0.7822 | 0.8289 | 0.8049 | 1.70 | 0.76 | 1311.0 |
