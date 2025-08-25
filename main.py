import argparse
import torch
from src.environments._2d_env import Environment2D
from src.planning.dynamics import DifferentiableDynamicsModel
from src.planning.planner import GPUParallelPlanner
from src.visualization.plot import plot_plan

class RobotBrain:
    """
    一个机器人大脑，使用基于GPU的并行规划器在2D环境中导航。
    """
    def __init__(self, planner: GPUParallelPlanner, dynamics_model: DifferentiableDynamicsModel) -> None:
        self.planner = planner
        self.dynamics_model = dynamics_model

    def plan(self, start_pos: torch.Tensor, goal_pos: torch.Tensor):
        """
        根据观察结果生成一个高层次的行动计划。
        """
        return self.planner.plan(start_pos, goal_pos)

    def get_trajectory(self, start_pos: torch.Tensor, plan: torch.Tensor):
        """
        根据起始位置和计划计算轨迹。
        """
        # rollout函数需要一批计划，所以我们对计划进行unsqueeze操作。
        trajectory = self.dynamics_model.rollout_plans(start_pos, plan.unsqueeze(0)).squeeze(0)
        # 在轨迹前添加起始位置以便绘图
        full_trajectory = torch.cat([start_pos.unsqueeze(0), trajectory], dim=0)
        return full_trajectory


    def act(self, plan: torch.Tensor) -> None:
        """
        通过打印来“执行”一个给定的行动计划。
        """
        print("\n正在执行最终计划（动作序列）:")
        print(plan)


def main() -> None:
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="运行基于GPU的2D规划器。")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='运行规划器的设备（cuda或cpu）。')
    parser.add_argument('--safety-weight', type=float, default=0.5,
                        help='规划器中安全成本的权重。')
    args = parser.parse_args()
    device = args.device

    # --- 设置 ---
    # 如果请求了GPU，则检查GPU是否可用，如果不可用则回退到CPU。
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，将回退到CPU。为获得更好性能，请在启用CUDA的GPU上运行。")
        device = 'cpu'

    print(f"正在使用设备: {device}")


    # --- 环境 ---
    print("正在设置2D环境...")
    start_pos = torch.tensor([0.0, 0.0], device=device)
    goal_pos = torch.tensor([1.0, 1.0], device=device)
    # 以逆时针（CCW）顺序定义多边形顶点。
    obstacles = [
        # 一个方形障碍物
        torch.tensor([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]),
        # 一个三角形障碍物
        torch.tensor([[0.1, 0.7], [0.3, 0.7], [0.2, 0.9]]),
        # 另一个矩形
        torch.tensor([[0.7, 0.1], [0.9, 0.1], [0.9, 0.3], [0.7, 0.3]]),
    ]
    env = Environment2D(start_pos, goal_pos, obstacles, device=device)
    print("环境已创建。")

    # --- 模型和规划器 ---
    print("正在初始化模型...")
    dynamics_model = DifferentiableDynamicsModel(env, safety_weight=args.safety_weight)

    # 为CPU与GPU使用不同的参数以便及时执行
    if device == 'cpu':
        print("正在使用CPU友好参数（较少的计划和迭代次数）。")
        num_plans = 512
        iterations = 30
    else:
        num_plans = 4096
        iterations = 100

    planner = GPUParallelPlanner(
        dynamics_model,
        num_plans=num_plans,
        plan_horizon=20,
        iterations=iterations
    )
    print("规划器已初始化。")

    # --- 机器人大脑 ---
    brain = RobotBrain(planner, dynamics_model)

    # --- 运行 ---
    print("\n--- 开始规划 ---")
    plan = brain.plan(env.robot_pos, env.goal_pos)
    trajectory = brain.get_trajectory(env.robot_pos, plan)
    brain.act(plan)

    # --- 可视化 ---
    print("\n--- 生成可视化 ---")
    plot_plan(env, trajectory, "plan_visualization.png")
    print("\n--- 规划结束 ---")


if __name__ == "__main__":
    main()
