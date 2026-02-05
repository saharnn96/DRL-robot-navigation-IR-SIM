# Simulation Timing Analysis

## Benchmark Results

| Configuration   | Time (100 steps) | Per Step | Steps/Second |
| --------------- | ---------------- | -------- | ------------ |
| GPU + No Render | 4.61s            | 46 ms    | ~22 Hz       |
| GPU + Render    | 7.73s            | 77 ms    | ~13 Hz       |
| CPU + No Render | 1.5s\*           | 15 ms    | ~67 Hz       |
| CPU + Render    | 3.9s\*           | 39 ms    | ~26 Hz       |

\*Earlier benchmark results

---

## Simulation Time vs Real Time

- **Simulated time per step**: `0.3s` (configured in `step_time`)
- **Real wall-clock time per step**: `~46ms` (without render)
- **Simulation speed**: ~6.5x faster than real-time

---

## What Takes Time (Per Step)

| Component                     | Runs On | Approximate Time |
| ----------------------------- | ------- | ---------------- |
| LIDAR ray casting             | CPU     | ~20-25 ms        |
| Collision detection           | CPU     | ~5-10 ms         |
| RVO obstacle AI (4 obstacles) | CPU     | ~10-15 ms        |
| Physics integration           | CPU     | ~1-2 ms          |
| Model inference               | GPU     | ~1-2 ms          |
| **Rendering (if enabled)**    | CPU/GPU | **~31 ms**       |

> Note: ~95% of simulation time is CPU-bound. GPU mainly helps during batch training updates.

---

## How to Make It Faster

### 1. Already Implemented

- ✅ `disable_plotting=True` - Saves ~40% time

### 2. YAML Configuration Changes

```yaml
# Reduce LIDAR rays (currently 180)
sensors:
  - type: "lidar2d"
    number: 90 # Half the rays = faster

    # Disable LIDAR noise
    noise: False # Removes noise computation

# Fewer moving obstacles
obstacle:
  - number: 2 # Instead of 4

# Simpler collision mode
collision_mode: "stop" # Instead of 'reactive'
```

### 3. Hardware Solutions

| Solution                        | Expected Speedup | Notes                   |
| ------------------------------- | ---------------- | ----------------------- |
| Faster single-core CPU (5+ GHz) | 1.3-1.5x         | Direct improvement      |
| Multiple parallel simulations   | 2-4x             | Uses more CPU cores     |
| More RAM                        | Minimal          | Not a bottleneck        |
| Better GPU                      | Minimal          | Simulation is CPU-bound |

---

## Multi-threading Considerations

### ❌ Single Simulation Cannot Be Parallelized

Each step depends on the previous state:

```
Step 1 → Step 2 → Step 3 → ...
```

### ✅ Multiple Parallel Environments Work

```
Sim 1: Step 1 → Step 2 → Step 3 → ...   (Core 1)
Sim 2: Step 1 → Step 2 → Step 3 → ...   (Core 2)
Sim 3: Step 1 → Step 2 → Step 3 → ...   (Core 3)
```

This is called "vectorized environments" and can provide 2-4x speedup.

---

## `step_time` and `sample_time` Explained

### `step_time` (Physics timestep)

- Time interval for each physics calculation
- Current: `0.3s`
- **Lower value** = More accurate physics, more steps needed
- **Higher value** = Faster training, risk of collision tunneling

### `sample_time` (Sensor/Render rate)

- How often sensor data is sampled
- Usually set equal to `step_time`

### Safe Values Based on Robot Speed

| step_time | Distance per step (at 1 m/s) | Risk Level       |
| --------- | ---------------------------- | ---------------- |
| 0.1s      | 0.1m                         | Very Safe        |
| 0.3s      | 0.3m                         | Safe ✓ (current) |
| 0.5s      | 0.5m                         | Moderate         |
| 1.0s      | 1.0m                         | Risky            |

---

## Recommendations

For **maximum training speed** without breaking physics:

1. Keep `step_time: 0.3` (current, safe)
2. Use `disable_plotting=True` (already done)
3. Consider reducing LIDAR to 90 rays (requires retraining)
4. Run 2-4 parallel simulations if CPU has multiple cores

---

## System Info

- **GPU**: NVIDIA RTX 2000 Ada Generation Laptop GPU
- **PyTorch**: 2.5.1+cu121 (CUDA enabled)
- **Date**: February 2, 2026
