"""
Parallelized RL Training Script

Uses multiple simulation environments running in parallel processes
to collect experience faster, utilizing all CPU cores.
"""

from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
import torch
import numpy as np
from multiprocessing import Process, Queue, cpu_count
from robot_nav.SIM_ENV.sim import SIM
from utils import get_buffer
import time


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
    sim = SIM(world_file="worlds/robot_world.yaml", disable_plotting=True)
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


def main():
    """Main training function with parallel environments."""
    # Configuration
    action_dim = 2
    max_action = 1
    state_dim = 185
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training parameters
    num_envs = min(8, cpu_count() - 2)  # Use 8 parallel envs or leave 2 cores free
    print(f"Using {num_envs} parallel environments")
    
    nr_eval_episodes = 10
    max_epochs = 30
    epoch = 0
    episodes_per_epoch = 50
    train_every_n = 2
    training_iterations = 50
    batch_size = 64 * num_envs  # Scale batch size with envs
    max_steps = 200
    save_every = 5
    
    # Initialize model
    model = CNNTD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        save_every=save_every,
        load_model=False,
        model_name="CNNTD3_parallel",
    )
    
    # Initialize parallel environments
    envs = ParallelEnvs(num_envs=num_envs)
    
    # Initialize single env for evaluation
    eval_sim = SIM(world_file="worlds/robot_world.yaml", disable_plotting=True)
    
    # Get replay buffer (use single sim for buffer initialization)
    temp_sim = SIM(world_file="worlds/robot_world.yaml", disable_plotting=True)
    replay_buffer = get_buffer(model, temp_sim, False, False, 10, training_iterations, batch_size)
    
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
            
            # Get actions for all environments (batch inference)
            actions = []
            for state in states:
                action = model.get_action(np.array(state), True)
                a_in = ((action[0] + 1) / 4, action[1])
                actions.append(a_in)
            
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
                    
                    # Train periodically
                    if episode_count % train_every_n == 0:
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
                print(f"Epoch {epoch} completed in {elapsed:.1f}s ({episode_count} episodes)")
                evaluate(model, epoch, eval_sim, eval_episodes=nr_eval_episodes)
                episode_count = 0
                start_time = time.time()
    
    finally:
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
        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()
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
