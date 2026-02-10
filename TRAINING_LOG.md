# Training Log

| Timestamp | Script | Model | Model Name | Device | State Dim | Action Dim | Max Epochs | Num Envs | Parallel | Batch Inference | Async Training | Episodes/Epoch | Train Every N | Training Iterations | Batch Size | Save Every | Load Model |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-02-09 18:03:24 | rl_train.py | CNNTD3 | CNNTD3 | cuda | 185 | 2 | 50 |  | 1 | No | No | No | 50 | 2 | 50 | 64 |  | No |
| 2026-02-09 18:05:21 | rl_train.py | CNNTD3 | CNNTD3_base | cuda | 185 | 2 | 50 |  | 1 | No | No | No | 50 | 2 | 50 | 64 |  | No |
| 2026-02-10 09:06:01 | rl_train_parallel.py | CNNTD3 | CNNTD3_parallel_base | cuda | 185 | 2 | 50 |  | 20 | Yes | No | Yes | 50 | 2 | 50 | 256 | 50 | No |
| 2026-02-10 09:57:49 | rl_train_parallel.py | CNNTD3 | CNNTD3_parallel_base | cuda | 185 | 2 | 50 |  | 20 | Yes | No | Yes | 50 | 2 | 50 | 256 | 50 | No |
| 2026-02-10 09:58:35 | rl_train_parallel.py | CNNTD3 | CNNTD3_parallel_base | cuda | 185 | 2 | 50 |  | 20 | Yes | No | Yes | 50 | 2 | 50 | 256 | 50 | YES |
| 2026-02-10 12:06:56 | rl_train_parallel_twin.py | CNNTD3 | CNNTD3_parallel8_sim2 | cuda | 185 | 2 | 80 | 50 | 8 | Yes | No | Yes | 50 | 2 | 50 | 256 | 50 | No |
