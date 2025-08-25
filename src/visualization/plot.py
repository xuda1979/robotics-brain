import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ..environments._2d_env import Environment2D

def plot_plan(env: Environment2D, trajectory: torch.Tensor, filename: str):
    """
    Generates and saves a plot of the environment and the planned trajectory.

    Args:
        env (Environment2D): The 2D environment.
        trajectory (torch.Tensor): The resulting trajectory, shape (plan_horizon + 1, 2).
                                  This should include the starting position.
        filename (str): The path to save the plot image.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot obstacles
    for i in range(env.obstacle_centers.shape[0]):
        center = env.obstacle_centers[i].cpu().numpy()
        radius = env.obstacle_radii[i].cpu().item()
        # Only add label once to avoid duplicates in the legend
        label = 'Obstacle' if i == 0 else ""
        circle = patches.Circle(center, radius, color='red', alpha=0.5, label=label)
        ax.add_patch(circle)

    # Plot start and goal
    ax.plot(env.robot_pos[0].cpu(), env.robot_pos[1].cpu(), 'go', markersize=10, label='Start')
    ax.plot(env.goal_pos[0].cpu(), env.goal_pos[1].cpu(), 'b*', markersize=15, label='Goal')

    # Plot trajectory
    traj_np = trajectory.cpu().numpy()
    ax.plot(traj_np[:, 0], traj_np[:, 1], 'c-', label='Trajectory')
    ax.plot(traj_np[:, 0], traj_np[:, 1], 'c.')

    # Set plot limits and labels
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Robot Plan Visualization")
    ax.legend()
    ax.grid(True)

    # Save the plot
    plt.savefig(filename)
    plt.close(fig)
    print(f"Plot saved to {filename}")
