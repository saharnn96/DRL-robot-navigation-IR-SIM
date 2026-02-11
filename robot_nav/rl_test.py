from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3

import torch
import numpy as np
from robot_nav.SIM_ENV.sim import SIM
import yaml
import matplotlib.pyplot as plt

WORLD_NAME = "robot_world_3"
MODEL_NAME = "CNNTD3_parallel_world1_base"  # name of the model to load and test, should match the name used during training

def main(args=None):
    """Main testing function"""
    action_dim = 2  # number of actions produced by the model
    max_action = 1  # maximum absolute value of output actions
    state_dim = 185  # number of input values in the neural network (vector length of state input)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # using cuda if it is available, cpu otherwise
    epoch = 0  # epoch number
    max_steps = 300  # maximum number of steps in single episode

    model = CNNTD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        load_model=True,
        model_name=MODEL_NAME,
    )  # instantiate a model

    sim = SIM(world_file="worlds/"+ WORLD_NAME + ".yaml")  # instantiate environment
    with open("robot_nav/eval_points.yaml") as file:
        points = yaml.safe_load(file)
    robot_poses = points["robot"]["poses"]
    robot_goals = points["robot"]["goals"]

    assert len(robot_poses) == len(
        robot_goals
    ), "Number of robot poses do not equal the robot goals"

    print("..............................................")
    print(f"Testing {len(robot_poses)} scenarios")
    total_reward = 0.0
    total_steps = 0
    col = 0
    goals = 0
    rewards_per_ep = []
    lin_actions = []
    ang_actions = []
    steps_per_ep = []
    for idx in range(len(robot_poses)):
        count = 0
        ep_reward = 0.0
        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset(
            robot_state=robot_poses[idx],
            robot_goal=robot_goals[idx],
            random_obstacles=False,
        )
            
        done = False
        while not done and count < max_steps:
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )
            action = model.get_action(np.array(state), False)
            a_in = [(action[0] + 1) / 4, action[1]]
            lin_actions.append(a_in[0])
            ang_actions.append(a_in[1])
            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
                lin_velocity=a_in[0], ang_velocity=a_in[1]
            )
            total_reward += reward
            ep_reward += reward
            total_steps += 1
            count += 1
            if collision:
                col += 1
            if goal:
                goals += 1
            done = collision or goal
        rewards_per_ep.append(ep_reward)
        steps_per_ep.append(count)
    # Convert to numpy arrays
    rewards_per_ep = np.array(rewards_per_ep)
    lin_actions = np.array(lin_actions)
    ang_actions = np.array(ang_actions)
    steps_per_ep = np.array(steps_per_ep)

    avg_step_reward = total_reward / total_steps
    avg_reward = total_reward / len(robot_poses)
    avg_col = col / len(robot_poses)
    avg_goal = goals / len(robot_poses)
    print(f"Total Reward: {total_reward}")
    print(f"Average Reward: {avg_reward}")
    print(f"Average Step Reward: {avg_step_reward}")
    print(f"Average Collision rate: {avg_col}")
    print(f"Average Goal rate: {avg_goal}")
    print(f"Average Steps per Episode: {np.mean(steps_per_ep):.2f}")
    print(f"Mean Linear Action: {np.mean(lin_actions):.4f}")
    print(f"Mean Angular Action: {np.mean(ang_actions):.4f}")
    print("..............................................")
    model.writer.add_scalar("test/total_reward", total_reward, epoch)
    model.writer.add_scalar("test/avg_reward", avg_reward, epoch)
    model.writer.add_scalar("test/avg_step_reward", avg_step_reward, epoch)
    model.writer.add_scalar("test/avg_col", avg_col, epoch)
    model.writer.add_scalar("test/avg_goal", avg_goal, epoch)

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Reward per episode bar chart
    axes[0, 0].bar(range(len(rewards_per_ep)), rewards_per_ep, color='steelblue')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Reward per Episode')
    axes[0, 0].axhline(y=np.mean(rewards_per_ep), color='r', linestyle='--', label=f'Mean: {np.mean(rewards_per_ep):.2f}')
    axes[0, 0].legend()

    # Linear actions histogram
    axes[0, 1].hist(lin_actions, bins=30, color='green', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Linear Velocity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Linear Actions Distribution (mean: {np.mean(lin_actions):.3f})')

    # Angular actions histogram
    axes[1, 0].hist(ang_actions, bins=30, color='orange', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Angular Velocity')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Angular Actions Distribution (mean: {np.mean(ang_actions):.3f})')

    # Steps per episode bar chart
    colors = ['green' if goals else 'red' for goals in [g == 1 for g in range(len(steps_per_ep))]]
    axes[1, 1].bar(range(len(steps_per_ep)), steps_per_ep, color='purple', alpha=0.7)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Steps')
    axes[1, 1].set_title(f'Steps per Episode (mean: {np.mean(steps_per_ep):.1f})')
    axes[1, 1].axhline(y=np.mean(steps_per_ep), color='r', linestyle='--', label=f'Mean: {np.mean(steps_per_ep):.1f}')
    axes[1, 1].legend()


    plt.suptitle(f"Test Results\nModel: {MODEL_NAME} | World: {WORLD_NAME}", fontsize=16, y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    # Sanitize file name

    fig_filename = f"robot_nav/models/CNNTD3/checkpoint/test_results_{MODEL_NAME}_{WORLD_NAME}.png"
    plt.savefig(fig_filename, dpi=150)
    plt.show()

    # Add histograms to tensorboard
    model.writer.add_histogram("test/lin_actions", lin_actions, epoch)
    model.writer.add_histogram("test/ang_actions", ang_actions, epoch)
    model.writer.add_figure("test/results_figure", fig, epoch)


if __name__ == "__main__":
    main()