# Training Log

| Timestamp | Script | Model | Model Name | Device | State Dim | Action Dim | Max Epochs | Num Envs | Parallel | Batch Inference | Async Training | Episodes/Epoch | Train Every N | Training Iterations | Batch Size | Save Every | Load Model |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-02-09 18:03:24 | rl_train.py | CNNTD3 | CNNTD3 | cuda | 185 | 2 | 50 |  | 1 | No | No | No | 50 | 2 | 50 | 64 |  | No |
| 2026-02-09 18:05:21 | rl_train.py | CNNTD3 | CNNTD3_base | cuda | 185 | 2 | 50 |  | 1 | No | No | No | 50 | 2 | 50 | 64 |  | No |
| 2026-02-10 09:06:01 | rl_train_parallel.py | CNNTD3 | CNNTD3_parallel_base | cuda | 185 | 2 | 50 |  | 20 | Yes | No | Yes | 50 | 2 | 50 | 256 | 50 | No |
| 2026-02-10 09:57:49 | rl_train_parallel.py | CNNTD3 | CNNTD3_parallel_base | cuda | 185 | 2 | 50 |  | 20 | Yes | No | Yes | 50 | 2 | 50 | 256 | 50 | No |
| 2026-02-10 09:58:35 | rl_train_parallel.py | CNNTD3 | CNNTD3_parallel_base | cuda | 185 | 2 | 50 |  | 20 | Yes | No | Yes | 50 | 2 | 50 | 256 | 50 | YES |
| 2026-02-10 12:06:56 | rl_train_parallel_twin.py | CNNTD3 | CNNTD3_parallel8_sim2 | cuda | 185 | 2 | 80 | 50 | 8 | Yes | No | Yes | 50 | 2 | 50 | 256 | 50 | No |
| 2026-02-10 12:19:17 | rl_train_parallel_twin.py | CNNTD3 | CNNTD3_parallel8_sim2 | cuda | 185 | 2 | 80 | 50 | 8 | Yes | No | Yes | 50 | 2 | 50 | 256 | 50 | No |
| 2026-02-10 12:21:35 | rl_train_parallel_twin.py | CNNTD3 | CNNTD3_parallel8_sim2 | cuda | 185 | 2 | 80 | 50 | 8 | Yes | No | Yes | 50 | 2 | 50 | 256 | 50 | No |
| 2026-02-10 13:13:49 | rl_train_parallel.py | CNNTD3 | CNNTD3_parallel3 | cuda | 185 | 2 | 80 | 50 | 20 | Yes | No | Yes | 50 | 2 | 50 | 256 | 50 | No |
| 2026-02-10 13:18:09 | rl_train_parallel.py | CNNTD3 | CNNTD3_parallel3 | cuda | 185 | 2 | 80 | 50 | 20 | Yes | No | Yes | 50 | 2 | 50 | 256 | 50 | No |
| 2026-02-10 14:02:18 | rl_train_parallel.py | CNNTD3 | CNNTD3_parallel_world2 | cuda | 185 | 2 | 80 | 50 | 20 | Yes | No | Yes | 50 | 2 | 50 | 256 | 50 | No |
| 2026-02-10 16:05:41 | rl_train_parallel.py | CNNTD3 | CNNTD3_parallel_world0_base | cuda | 185 | 2 | 80 | 50 | 20 | Yes | No | Yes | 50 | 2 | 50 | 256 | 50 | No |
| 2026-02-10 16:06:05 | rl_train_parallel.py | CNNTD3 | CNNTD3_parallel_world0_base | cuda | 185 | 2 | 80 | 50 | 20 | Yes | No | Yes | 50 | 2 | 50 | 256 | 50 | No |
| 2026-02-10 16:06:27 | rl_train_parallel.py | CNNTD3 | CNNTD3_parallel_world0_base | cuda | 185 | 2 | 80 | 50 | 20 | Yes | No | Yes | 50 | 2 | 50 | 256 | 50 | No |
| 2026-02-10 16:08:28 | rl_train_parallel.py | CNNTD3 | CNNTD3_parallel_world1_base | cuda | 185 | 2 | 30 |  | 20 | Yes | No | Yes | 50 | 2 | 50 | 256 | 50 | No |
| 2026-02-10 17:20:19 | rl_train_parallel.py | CNNTD3 | CNNTD3_parallel_world1_base | cuda | 185 | 2 | 50 |  | 20 | Yes | No | Yes | 50 | 2 | 50 | 256 | 50 | No |
| 2026-02-10 18:05:40 | rl_train_parallel.py | CNNTD3 | CNNTD3_parallel_world1 | cuda | 185 | 2 | 70 | 40 | 20 | Yes | No | Yes | 50 | 2 | 50 | 256 | 50 | No |
| 2026-02-10 19:04:51 | rl_train_parallel.py | CNNTD3 | CNNTD3_parallel_world2 | cuda | 185 | 2 | 70 | 40 | 20 | Yes | No | Yes | 50 | 2 | 50 | 256 | 50 | No |
| 2026-02-10 19:06:18 | rl_train_parallel.py | CNNTD3 | CNNTD3_parallel_world2 | cuda | 185 | 2 | 70 | 40 | 20 | Yes | No | Yes | 50 | 2 | 50 | 256 | 50 | No |
| 2026-02-10 20:01:28 | rl_train_parallel.py | CNNTD3 | CNNTD3_parallel_world3 | cuda | 185 | 2 | 70 | 40 | 20 | Yes | No | Yes | 50 | 2 | 50 | 256 | 50 | No |
