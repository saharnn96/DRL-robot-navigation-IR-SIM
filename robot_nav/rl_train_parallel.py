"""
Parallelized RL Training Script with Async Training

Uses multiple simulation environments running in parallel processes
to collect experience faster, utilizing all CPU cores.
Training runs in a separate thread to avoid blocking collection.
"""

from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
import torch
import numpy as np
from multiprocessing import Process, Queue, cpu_count
from robot_nav.SIM_ENV.sim import SIM
from utils import get_buffer
from robot_nav.training_log import log_training_run
import time
import threading
from queue import Queue as ThreadQueue


def worker_collect_experience(worker_id, action_queue, experience_queue, num_steps):
    """
    Worker process that runs a simulation and collects experiences.
    
    Args:
        worker_id: Unique ID for this worker
        action_queue: Queue to receive actions from main process
        experience_queue: Queue to send experiences back
        num_steps: Number of steps to collect before sending batch
    """
    # Each worker has its own simulation instance
    sim = SIM(world_file="worlds/robot_world_1.yaml", disable_plotting=True)
    latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()
    
    while True:
        # Wait for action from main process (or "reset" or "stop" command)
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
            # cmd is an action
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
        
        # Start worker processes
        for i in range(num_envs):
            p = Process(
                target=worker_collect_experience,
                args=(i, self.action_queues[i], self.experience_queue, 100)
            )
            p.start()
            self.workers.append(p)
    
    def reset_all(self):
        """Reset all environments and get initial observations."""
        for q in self.action_queues:
            q.put("reset")
        
        observations = [None] * self.num_envs
        for _ in range(self.num_envs):
            result = self.experience_queue.get()
            observations[result["worker_id"]] = result["obs"]
        return observations
    
    def step_all(self, actions):
        """
        Send actions to all environments and collect results.
        
        Args:
            actions: List of (lin_vel, ang_vel) tuples, one per environment
        """
        for i, action in enumerate(actions):
            self.action_queues[i].put(action)
        
        observations = [None] * self.num_envs
        for _ in range(self.num_envs):
            result = self.experience_queue.get()
            observations[result["worker_id"]] = result["obs"]
        return observations
    
    def close(self):
        """Stop all workers."""
        for q in self.action_queues:
            q.put("stop")
        for p in self.workers:
            p.join()


def worker_eval_episode(worker_id, world_file, model_state_dict, model_class_name, device_str, result_queue, max_steps=501):
    """
    Worker process that runs a single evaluation episode.
    
    Args:
        worker_id: Unique ID for this worker
        world_file: Path to world file for simulation
        model_state_dict: Dictionary containing actor/critic state dicts
        model_class_name: Name of the model class ("CNNTD3", "SAC", etc.)
        device_str: Device string ("cuda" or "cpu")
        result_queue: Queue to send episode results back
        max_steps: Maximum steps per episode
    """
    # Import here to avoid issues with multiprocessing
    import torch
    import numpy as np
    from robot_nav.SIM_ENV.sim import SIM
    
    # Dynamically import the actor class ONLY (not full model to avoid TensorBoard writer)
    from robot_nav.models.CNNTD3.CNNTD3 import Actor

    
    # Create simulation environment
    sim = SIM(world_file=world_file, disable_plotting=True)
    
    # Reconstruct ONLY actor network (no TensorBoard writer!)
    device = torch.device(device_str)
    state_dim = model_state_dict["state_dim"]
    action_dim = model_state_dict["action_dim"]
    max_action = model_state_dict["max_action"]
    
    actor = Actor(action_dim).to(device)

    # Load the model weights
    actor.load_state_dict(model_state_dict["actor"])
    actor.eval()
    
    # Run one episode
    latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()
    episode_reward = 0.0
    collisions = 0
    goals_reached = 0
    steps = 0
    done = False
    
    while not done and steps < max_steps:
        # Reproduce model.prepare_state normalization
        scan = np.array(latest_scan, dtype=np.float32)
        scan[np.isinf(scan)] = 7.0
        scan /= 7.0
        norm_dist = distance / 10.0
        lin_vel = a[0] * 2
        ang_vel_norm = (a[1] + 1) / 2
        state = np.concatenate((scan, [norm_dist, cos, sin, lin_vel, ang_vel_norm]))
        
        # Get action directly from actor
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_tensor = actor(state_tensor)
            action = action_tensor.cpu().numpy()[0]
        
        a_in = [(action[0] + 1) / 4, action[1]]
        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1]
        )
        
        episode_reward += reward
        steps += 1
        
        if collision:
            collisions = 1
        if goal:
            goals_reached = 1
        done = collision or goal
    
    # Send results back
    result_queue.put({
        "worker_id": worker_id,
        "reward": episode_reward,
        "collisions": collisions,
        "goals": goals_reached,
        "steps": steps,
    })


class ParallelEvaluator:
    """
    Manages parallel evaluation of multiple episodes.
    Each episode runs in its own process for true parallelism.
    """
    
    @staticmethod
    def evaluate_parallel(model, world_file, num_episodes=10, max_steps=501):
        """
        Run evaluation episodes in parallel.
        
        Args:
            model: The RL model to evaluate
            world_file: Path to world configuration file
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            
        Returns:
            dict with avg_reward, avg_collisions, avg_goals
        """
        result_queue = Queue()
        processes = []
        
        # Prepare model state for multiprocessing
        model_state = {
            "state_dim": model.state_dim,
            "action_dim": model.action_dim,
            "max_action": model.max_action,
            "actor": model.actor.state_dict(),
        }
        
        model_class_name = type(model).__name__
        device_str = str(model.device)
        
        # Start worker processes
        for i in range(num_episodes):
            p = Process(
                target=worker_eval_episode,
                args=(i, world_file, model_state, model_class_name, device_str, result_queue, max_steps)
            )
            p.start()
            processes.append(p)
        
        # Collect results
        results = []
        for _ in range(num_episodes):
            results.append(result_queue.get())
        
        # Wait for all processes to finish
        for p in processes:
            p.join()
        
        # Aggregate results
        total_reward = sum(r["reward"] for r in results)
        total_collisions = sum(r["collisions"] for r in results)
        total_goals = sum(r["goals"] for r in results)
        
        return {
            "avg_reward": total_reward / num_episodes,
            "avg_collisions": total_collisions / num_episodes,
            "avg_goals": total_goals / num_episodes,
        }


class AsyncTrainer:
    """
    Handles training in a background thread so collection can continue.
    Uses a lock to prevent race conditions between inference and training.
    """
    
    def __init__(self, model, replay_buffer, training_iterations, batch_size):
        self.model = model
        self.replay_buffer = replay_buffer
        self.training_iterations = training_iterations
        self.batch_size = batch_size
        
        self.train_queue = ThreadQueue()  # Queue training requests
        self.is_training = False
        self.training_thread = None
        self.stop_flag = False
        self.train_count = 0
        
        # Lock to prevent simultaneous model access
        self.model_lock = threading.Lock()
        
        # Start the training thread
        self._start_training_thread()
    
    def _start_training_thread(self):
        """Start the background training thread."""
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
    
    def _training_loop(self):
        """Background thread that processes training requests."""
        while not self.stop_flag:
            try:
                # Wait for training request (with timeout to check stop flag)
                request = self.train_queue.get(timeout=0.1)
                if request == "train":
                    self.is_training = True
                    # Acquire lock during training
                    with self.model_lock:
                        self.model.train(
                            replay_buffer=self.replay_buffer,
                            iterations=self.training_iterations,
                            batch_size=self.batch_size,
                        )
                    self.train_count += 1
                    self.is_training = False
            except:
                # Timeout, check stop flag and continue
                pass
    
    def get_actions_safe(self, states, add_noise):
        """Thread-safe per-state action inference (no batch)."""
        actions = []
        with self.model_lock:
            # Call get_action for each state while holding the model lock to avoid
            # race conditions between inference and background training.
            for s in states:
                actions.append(self.model.get_action(np.array(s), add_noise))
        return actions
    
    def request_training(self):
        """Request a training session (non-blocking)."""
        # Only queue if not already training
        if not self.is_training and self.train_queue.empty():
            self.train_queue.put("train")
    
    def stop(self):
        """Stop the training thread."""
        self.stop_flag = True
        if self.training_thread:
            self.training_thread.join(timeout=5.0)


def main():
    """Main training function with parallel environments."""
    # Configuration
    action_dim = 2
    max_action = 1
    state_dim = 185
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training parameters
    num_envs = min(20, cpu_count() - 4)  # Use more parallel envs, leave 4 cores for main + training
    print(f"Using {num_envs} parallel environments")
    
    nr_eval_episodes = 10
    max_epochs = 70
    epoch = 40
    episodes_per_epoch = 50
    train_every_n = 2 # Train less often to prevent overfitting
    training_iterations = 50  # Reduced iterations to prevent overfitting
    batch_size = 256  # Fixed smaller batch size for more gradient noise
    max_steps = 200
    save_every = 50
    

    
    # Initialize model
    model = CNNTD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        save_every=save_every,
        load_model=True,
        model_name="CNNTD3_parallel_world1",
    )
        # Log training run parameters (before model init)
    # Log training run parameters
    try:
        log_training_run({
            "script": "rl_train_parallel.py",
            "model": type(model).__name__,
            "model_name": getattr(model, "model_name", ""),
            "device": str(device),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_epochs": max_epochs,
            "epoch": epoch,
            "num_envs": num_envs,
            "parallel": True,
            "batch_inference": False,
            "async_training": True,
            "episodes_per_epoch": episodes_per_epoch,
            "train_every_n": train_every_n,
            "training_iterations": training_iterations,
            "batch_size": batch_size,
            "save_every": save_every,
            "load_model": model.load_model if hasattr(model, "load_model") else False,
        })
    except Exception:
        pass
    # Initialize parallel environments
    envs = ParallelEnvs(num_envs=num_envs)
    
    # Initialize single env for evaluation
    eval_sim = SIM(world_file="worlds/robot_world_1.yaml", disable_plotting=True)
    
    # Get replay buffer (use single sim for buffer initialization)
    temp_sim = SIM(world_file="worlds/robot_world_1.yaml", disable_plotting=True)
    replay_buffer = get_buffer(model, temp_sim, False, False, 10, training_iterations, batch_size)
    
    # Initialize async trainer (runs in background thread)
    async_trainer = AsyncTrainer(model, replay_buffer, training_iterations, batch_size)
    
    # Reset all environments
    observations = envs.reset_all()
    steps = [0] * num_envs
    episode_count = 0
    
    start_time = time.time()
    
    try:
        while epoch < max_epochs:
            # Get states for all environments
            states = []
            for obs in observations:
                latest_scan, distance, cos, sin, collision, goal, a, reward = obs
                state, _ = model.prepare_state(latest_scan, distance, cos, sin, collision, goal, a)
                states.append(state)
            
            # Get actions using thread-safe per-environment inference (no batch)
            actions_batch = async_trainer.get_actions_safe(states, add_noise=True)
            actions = [((a[0] + 1) / 4, a[1]) for a in actions_batch]
            
            # Step all environments in parallel
            next_observations = envs.step_all(actions)
            
            # Process results and add to replay buffer
            for i, (obs, next_obs, action) in enumerate(zip(observations, next_observations, actions)):
                latest_scan, distance, cos, sin, collision, goal, a, reward = obs
                state, _ = model.prepare_state(latest_scan, distance, cos, sin, collision, goal, a)
                
                n_scan, n_dist, n_cos, n_sin, n_col, n_goal, n_a, n_reward = next_obs
                next_state, terminal = model.prepare_state(n_scan, n_dist, n_cos, n_sin, n_col, n_goal, n_a)
                
                replay_buffer.add(state, list(action), n_reward, terminal, next_state)
                steps[i] += 1
                
                # Check for episode end
                if terminal or steps[i] >= max_steps:
                    episode_count += 1
                    steps[i] = 0
                    
                    # Request reset for this env
                    envs.action_queues[i].put("reset")
                    reset_result = envs.experience_queue.get()
                    next_observations[reset_result["worker_id"]] = reset_result["obs"]
                    
                    # Request async training (non-blocking!)
                    if episode_count % train_every_n == 0:
                        async_trainer.request_training()
            
            observations = next_observations
            
            # Epoch completed
            if episode_count >= episodes_per_epoch:
                epoch += 1
                elapsed = time.time() - start_time
                train_status = "training" if async_trainer.is_training else "idle"
                print(f"Epoch {epoch} completed in {elapsed:.1f}s ({episode_count} episodes, {async_trainer.train_count} train cycles, trainer: {train_status})")
                evaluate(model, epoch, eval_sim, eval_episodes=nr_eval_episodes)
                episode_count = 0
                start_time = time.time()
    
    finally:
        async_trainer.stop()
        envs.close()


def evaluate(model, epoch, sim, eval_episodes=10):
    """Evaluate the model using parallel episodes."""
    print("..............................................")
    print(f"Epoch {epoch}. Evaluating {eval_episodes} scenarios in parallel")
    
    eval_start = time.time()
    
    # Run evaluation in parallel
    results = ParallelEvaluator.evaluate_parallel(
        model=model,
        world_file="worlds/robot_world_1.yaml",
        num_episodes=eval_episodes,
        max_steps=501
    )
    
    eval_time = time.time() - eval_start
    
    avg_reward = results["avg_reward"]
    avg_col = results["avg_collisions"]
    avg_goal = results["avg_goals"]
    
    print(f"Evaluation completed in {eval_time:.2f}s")
    print(f"Average Reward: {avg_reward:.4f}")
    print(f"Average Collision rate: {avg_col:.4f}")
    print(f"Average Goal rate: {avg_goal:.4f}")
    print("..............................................")
    
    model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    model.writer.add_scalar("eval/avg_col", avg_col, epoch)
    model.writer.add_scalar("eval/avg_goal", avg_goal, epoch)
    model.writer.add_scalar("eval/eval_time", eval_time, epoch)


if __name__ == "__main__":
    main()
