"""
Debug version of Parallel RL Training

This script allows testing each optimization separately to find the issue:
- ENABLE_PARALLEL: Use multiple simulation environments
- ENABLE_BATCH_INFERENCE: Use get_action_batch() vs individual get_action()
- ENABLE_ASYNC_TRAINING: Train in background thread vs blocking

Set these flags to True/False to isolate the problem.
"""

from robot_nav.models.SAC.SAC import SAC
import torch
import numpy as np
from robot_nav.training_log import log_training_run
from multiprocessing import Process, Queue, cpu_count
from robot_nav.SIM_ENV.sim import SIM
from utils import get_buffer
import time
import threading
from queue import Queue as ThreadQueue

# ============= DEBUG FLAGS =============
ENABLE_PARALLEL = True        # Use multiple environments (vs single)
ENABLE_BATCH_INFERENCE = False # Use batch inference (vs loop)
ENABLE_ASYNC_TRAINING = True  # Use async training (vs blocking)
NUM_ENVS = 20                   # Number of parallel envs (if enabled)
# =======================================


def worker_collect_experience(worker_id, action_queue, experience_queue, num_steps):
    """Worker process that runs a simulation and collects experiences."""
    sim = SIM(world_file="worlds/eval_world.yaml", disable_plotting=True)
    latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset(
        robot_state=None,
        robot_goal=None,
        random_obstacles=True,
        random_obstacle_ids=[i + 1 for i in range(6)],
    )
    while True:
        cmd = action_queue.get()
        
        if cmd == "stop":
            break
        elif cmd == "reset":
            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()
            experience_queue.put({
                "worker_id": worker_id,
                "type": "reset",
                "obs": (latest_scan, distance, cos, sin, collision, goal, a, reward)
            })
        else:
            lin_vel, ang_vel = cmd
            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
                lin_velocity=lin_vel, ang_velocity=ang_vel
            )
            experience_queue.put({
                "worker_id": worker_id,
                "type": "step",
                "obs": (latest_scan, distance, cos, sin, collision, goal, a, reward)
            })


class ParallelEnvs:
    """Manages multiple parallel simulation environments."""
    
    def __init__(self, num_envs=4):
        self.num_envs = num_envs
        self.action_queues = [Queue() for _ in range(num_envs)]
        self.experience_queue = Queue()
        self.workers = []
        
        for i in range(num_envs):
            p = Process(
                target=worker_collect_experience,
                args=(i, self.action_queues[i], self.experience_queue, 100)
            )
            p.start()
            self.workers.append(p)
    
    def reset_all(self):
        for q in self.action_queues:
            q.put("reset")
        
        observations = [None] * self.num_envs
        for _ in range(self.num_envs):
            result = self.experience_queue.get()
            observations[result["worker_id"]] = result["obs"]
        return observations
    
    def step_all(self, actions):
        for i, action in enumerate(actions):
            self.action_queues[i].put(action)
        
        observations = [None] * self.num_envs
        for _ in range(self.num_envs):
            result = self.experience_queue.get()
            observations[result["worker_id"]] = result["obs"]
        return observations
    
    def close(self):
        for q in self.action_queues:
            q.put("stop")
        for p in self.workers:
            p.join()


class AsyncTrainer:
    """Handles training in a background thread."""
    
    def __init__(self, model, replay_buffer, training_iterations, batch_size):
        self.model = model
        self.replay_buffer = replay_buffer
        self.training_iterations = training_iterations
        self.batch_size = batch_size
        self.train_queue = ThreadQueue()
        self.is_training = False
        self.stop_flag = False
        self.train_count = 0
        self.model_lock = threading.Lock()
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
    
    def _training_loop(self):
        while not self.stop_flag:
            try:
                request = self.train_queue.get(timeout=0.1)
                if request == "train":
                    self.is_training = True
                    with self.model_lock:
                        self.model.train(
                            replay_buffer=self.replay_buffer,
                            iterations=self.training_iterations,
                            batch_size=self.batch_size,
                        )
                    self.train_count += 1
                    self.is_training = False
            except:
                pass
    
    def get_actions_safe(self, states_batch, add_noise):
        with self.model_lock:
            return self.model.get_action_batch(states_batch, add_noise)
    
    def request_training(self):
        if not self.is_training and self.train_queue.empty():
            self.train_queue.put("train")
    
    def stop(self):
        self.stop_flag = True
        self.training_thread.join(timeout=5.0)


def main():
    """Main training function with configurable optimizations."""
    print("=" * 60)
    print("DEBUG MODE - Testing optimizations separately")
    print(f"  ENABLE_PARALLEL:       {ENABLE_PARALLEL}")
    print(f"  ENABLE_BATCH_INFERENCE: {ENABLE_BATCH_INFERENCE}")
    print(f"  ENABLE_ASYNC_TRAINING:  {ENABLE_ASYNC_TRAINING}")
    print(f"  NUM_ENVS:              {NUM_ENVS if ENABLE_PARALLEL else 1}")
    print("=" * 60)
    
    # Configuration
    action_dim = 2
    max_action = 1
    state_dim = 185
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training parameters (same as rl_train.py)
    num_envs = NUM_ENVS if ENABLE_PARALLEL else 1
    nr_eval_episodes = 10
    max_epochs = 30
    epoch = 0
    episodes_per_epoch = 50
    train_every_n = 4
    training_iterations = 50
    batch_size = 64
    max_steps = 200
    save_every = 50
    
    # Initialize model
    model = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        save_every=save_every,
        load_model=False,
        model_name="SAC_static_env",
    )

    # Log training run parameters
    try:
        log_training_run({
            "script": "rl_train_parallel_debug.py",
            "model": type(model).__name__,
            "model_name": getattr(model, "model_name", ""),
            "device": str(device),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_epochs": max_epochs,
            "epoch": epoch,
            "num_envs": num_envs,
            "parallel": ENABLE_PARALLEL,
            "batch_inference": ENABLE_BATCH_INFERENCE,
            "async_training": ENABLE_ASYNC_TRAINING,
            "episodes_per_epoch": episodes_per_epoch,
            "train_every_n": train_every_n,
            "training_iterations": training_iterations,
            "batch_size": batch_size,
            "save_every": save_every,
            "load_model": False,
        })
    except Exception:
        pass
    
    # Initialize environment(s)
    if ENABLE_PARALLEL:
        envs = ParallelEnvs(num_envs=num_envs)
        observations = envs.reset_all()
    else:
        sim = SIM(world_file="worlds/eval_world.yaml", disable_plotting=True)
        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset(
            robot_state=None,
            robot_goal=None,
            random_obstacles=True,
            random_obstacle_ids=[i + 1 for i in range(6)],
        )
        observations = [(latest_scan, distance, cos, sin, collision, goal, a, reward)]
    
    # Initialize eval sim and replay buffer
    eval_sim = SIM(world_file="worlds/eval_world.yaml", disable_plotting=True)
    temp_sim = SIM(world_file="worlds/eval_world.yaml", disable_plotting=True)
    replay_buffer = get_buffer(model, temp_sim, False, False, 10, training_iterations, batch_size)
    
    # Initialize async trainer if enabled
    if ENABLE_ASYNC_TRAINING:
        async_trainer = AsyncTrainer(model, replay_buffer, training_iterations, batch_size)
    
    steps = [0] * num_envs
    episode_count = 0
    start_time = time.time()
    
    try:
        while epoch < max_epochs:
            # === GET STATES ===
            states = []
            for obs in observations:
                latest_scan, distance, cos, sin, collision, goal, a, reward = obs
                state, _ = model.prepare_state(latest_scan, distance, cos, sin, collision, goal, a)
                states.append(state)
            
            # === GET ACTIONS ===
            if ENABLE_BATCH_INFERENCE and ENABLE_ASYNC_TRAINING:
                # Batch + Async: use thread-safe method
                states_batch = np.array(states)
                actions_batch = async_trainer.get_actions_safe(states_batch, add_noise=True)
                actions = [((a[0] + 1) / 4, a[1]) for a in actions_batch]
            elif ENABLE_BATCH_INFERENCE:
                # Batch only: use batch method directly
                states_batch = np.array(states)
                actions_batch = model.get_action_batch(states_batch, add_noise=True)
                actions = [((a[0] + 1) / 4, a[1]) for a in actions_batch]
            else:
                # No batch: loop through states (LIKE ORIGINAL rl_train.py)
                actions = []
                for state in states:
                    action = model.get_action(np.array(state), True)
                    a_in = ((action[0] + 1) / 4, action[1])
                    actions.append(a_in)
            
            # === STEP ENVIRONMENTS ===
            if ENABLE_PARALLEL:
                next_observations = envs.step_all(actions)
            else:
                a_in = actions[0]
                latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
                    lin_velocity=a_in[0], ang_velocity=a_in[1]
                )
                next_observations = [(latest_scan, distance, cos, sin, collision, goal, a, reward)]
            
            # === PROCESS RESULTS ===
            for i, (obs, next_obs, action) in enumerate(zip(observations, next_observations, actions)):
                latest_scan, distance, cos, sin, collision, goal, a, reward = obs
                state, _ = model.prepare_state(latest_scan, distance, cos, sin, collision, goal, a)
                
                n_scan, n_dist, n_cos, n_sin, n_col, n_goal, n_a, n_reward = next_obs
                next_state, terminal = model.prepare_state(n_scan, n_dist, n_cos, n_sin, n_col, n_goal, n_a)
                
                replay_buffer.add(state, list(action), n_reward, terminal, next_state)
                steps[i] += 1
                
                if terminal or steps[i] >= max_steps:
                    episode_count += 1
                    steps[i] = 0
                    
                    # Reset this environment
                    if ENABLE_PARALLEL:
                        envs.action_queues[i].put("reset")
                        reset_result = envs.experience_queue.get()
                        next_observations[reset_result["worker_id"]] = reset_result["obs"]
                    else:
                        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset(
                            robot_state=None,
                            robot_goal=None,
                            random_obstacles=True,
                            random_obstacle_ids=[i + 1 for i in range(6)],
                        )
                        next_observations[0] = (latest_scan, distance, cos, sin, collision, goal, a, reward)
                    
                    # Train
                    if episode_count % train_every_n == 0:
                        if ENABLE_ASYNC_TRAINING:
                            async_trainer.request_training()
                        else:
                            # BLOCKING training (like original)
                            model.train(
                                replay_buffer=replay_buffer,
                                iterations=training_iterations,
                                batch_size=batch_size,
                            )
            
            observations = next_observations
            
            # Epoch completed
            if episode_count >= episodes_per_epoch:
                epoch += 1
                elapsed = time.time() - start_time
                if ENABLE_ASYNC_TRAINING:
                    print(f"Epoch {epoch} completed in {elapsed:.1f}s ({episode_count} episodes, {async_trainer.train_count} train cycles)")
                else:
                    print(f"Epoch {epoch} completed in {elapsed:.1f}s ({episode_count} episodes)")
                evaluate(model, epoch, eval_sim, eval_episodes=nr_eval_episodes)
                episode_count = 0
                start_time = time.time()
    
    finally:
        if ENABLE_ASYNC_TRAINING:
            async_trainer.stop()
        if ENABLE_PARALLEL:
            envs.close()


def evaluate(model, epoch, sim, eval_episodes=10):
    """Evaluate the model."""
    print("..............................................")
    print(f"Epoch {epoch}. Evaluating scenarios")
    avg_reward = 0.0
    col = 0
    goals = 0
    
    for _ in range(eval_episodes):
        count = 0
        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset(
            robot_state=None,
            robot_goal=None,
            random_obstacles=True,
            random_obstacle_ids=[i + 1 for i in range(6)],
        )
        done = False
        while not done and count < 501:
            state, terminal = model.prepare_state(latest_scan, distance, cos, sin, collision, goal, a)
            action = model.get_action(np.array(state), False)
            a_in = [(action[0] + 1) / 4, action[1]]
            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
                lin_velocity=a_in[0], ang_velocity=a_in[1]
            )
            avg_reward += reward
            count += 1
            if collision:
                col += 1
            if goal:
                goals += 1
            done = collision or goal
    
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    avg_goal = goals / eval_episodes
    print(f"Average Reward: {avg_reward}")
    print(f"Average Collision rate: {avg_col}")
    print(f"Average Goal rate: {avg_goal}")
    print("..............................................")
    model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    model.writer.add_scalar("eval/avg_col", avg_col, epoch)
    model.writer.add_scalar("eval/avg_goal", avg_goal, epoch)


if __name__ == "__main__":
    main()
