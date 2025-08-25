import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ..environments._2d_env import Environment2D

def plot_plan(env: Environment2D, trajectory: torch.Tensor, filename: str):
    """
    生成并保存环境和规划轨迹的图。

    Args:
        env (Environment2D): 2D环境。
        trajectory (torch.Tensor): 生成的轨迹，形状为 (plan_horizon + 1, 2)。
                                  这应包括起始位置。
        filename (str): 保存绘图图像的路径。
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制障碍物
    for i, poly_verts in enumerate(env.obstacles):
        # 只添加一次标签以避免图例中出现重复
        label = '障碍物' if i == 0 else ""
        polygon = patches.Polygon(poly_verts.cpu().numpy(), closed=True, color='red', alpha=0.5, label=label)
        ax.add_patch(polygon)

    # 绘制起点和终点
    ax.plot(env.robot_pos[0].cpu(), env.robot_pos[1].cpu(), 'go', markersize=10, label='起点')
    ax.plot(env.goal_pos[0].cpu(), env.goal_pos[1].cpu(), 'b*', markersize=15, label='终点')

    # 绘制轨迹
    traj_np = trajectory.cpu().numpy()
    ax.plot(traj_np[:, 0], traj_np[:, 1], 'c-', label='轨迹')
    ax.plot(traj_np[:, 0], traj_np[:, 1], 'c.')

    # 设置绘图范围和标签
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X 位置")
    ax.set_ylabel("Y 位置")
    ax.set_title("机器人计划可视化")
    ax.legend()
    ax.grid(True)

    # 保存绘图
    plt.savefig(filename)
    plt.close(fig)
    print(f"图已保存至 {filename}")
