import argparse
import torch
import importlib
from src.environments.base import BaseEnv, DynamicEnv
from src.planning.dynamics import DifferentiableDynamicsModel
from src.planning.planner import GPUParallelPlanner
from src.visualization.plot import plot_plan

class RobotBrain:
    """
    一个机器人大脑，使用基于GPU的并行规划器在2D环境中导航。
    """
    def __init__(self, planner: GPUParallelPlanner, dynamics_model: DifferentiableDynamicsModel, env: BaseEnv) -> None:
        self.planner = planner
        self.dynamics_model = dynamics_model
        self.env = env

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
        通过向环境发送命令来“执行”一个给定的行动计划。
        """
        print("\n正在执行最终计划（动作序列）:")
        print(plan)
        if isinstance(self.env, DynamicEnv):
            # 在动态环境中，我们可以真地移动机器人
            # 这是一个非常简单的实现，它只取计划的第一个动作
            # 并设置一个恒定的速度。
            # 一个更复杂的实现会根据计划动态调整速度。
            action = plan[0]
            # 基于行动设置速度的简单逻辑
            # action[0]控制前进速度，action[1]控制转向
            forward_speed = action[0] * 5.0  # 缩放因子
            turn_speed = action[1] * 2.0  # 缩放因子

            left_speed = forward_speed - turn_speed
            right_speed = forward_speed + turn_speed

            self.env.set_velocity(left_speed, right_speed)


def main() -> None:
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="运行基于GPU的规划器。")
    parser.add_argument('--environment', type=str, default='2d', choices=['2d', 'webots'],
                        help='要使用的环境（2d或webots）。')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='运行规划器的设备（cuda或cpu）。')
    parser.add_argument('--safety-weight', type=float, default=0.5,
                        help='规划器中安全成本的权重。')
    parser.add_argument('--random-env', action='store_true',
                        help='如果设置，则为2D环境使用随机生成的障碍物。')
    parser.add_argument('--headless', action='store_true',
                        help='在无头模式下运行Webots（无GUI）。')
    args = parser.parse_args()
    device = args.device

    # --- 设置 ---
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，将回退到CPU。为获得更好性能，请在启用CUDA的GPU上运行。")
        device = 'cpu'
    print(f"正在使用设备: {device}")

    # --- 环境工厂 ---
    def get_env_class(env_name):
        if env_name == '2d':
            module = importlib.import_module('src.environments._2d_env')
            return getattr(module, 'Environment2D')
        elif env_name == 'webots':
            try:
                module = importlib.import_module('src.environments._webots_env')
                return getattr(module, 'WebotsEnvironment')
            except ModuleNotFoundError:
                print("错误: 'webots' 环境需要安装 Webots 'controller' 模块。")
                print("请访问 https://cyberbotics.com/ 进行安装。")
                exit(1)
        else:
            raise ValueError(f"未知的环境: {env_name}")

    env_class = get_env_class(args.environment)

    print(f"正在设置 {args.environment} 环境...")
    if args.environment == '2d':
        if args.random_env:
            print("正在生成随机2D环境...")
            env = env_class.create_random(device=device)
        else:
            print("正在使用默认的2D环境...")
            start_pos = torch.tensor([0.0, 0.0], device=device)
            goal_pos = torch.tensor([1.0, 1.0], device=device)
            obstacles = [
                torch.tensor([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]),
                torch.tensor([[0.1, 0.7], [0.3, 0.7], [0.2, 0.9]]),
                torch.tensor([[0.7, 0.1], [0.9, 0.1], [0.9, 0.3], [0.7, 0.3]]),
            ]
            env = env_class(start_pos, goal_pos, obstacles, device=device)
    else:
        # Webots环境
        env = env_class("worlds/default.wbt", device=device, headless=args.headless)
    print("环境已创建。")


    # --- 模型和规划器 ---
    print("正在初始化模型...")
    dynamics_model = DifferentiableDynamicsModel(env, safety_weight=args.safety_weight)

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
    brain = RobotBrain(planner, dynamics_model, env)

    # --- 运行 ---
    print("\n--- 开始规划 ---")
    plan = brain.plan(env.robot_pos, env.goal_pos)
    trajectory = brain.get_trajectory(env.robot_pos, plan)
    brain.act(plan)

    # --- 可视化与仿真 ---
    if args.environment == '2d':
        print("\n--- 生成可视化 ---")
        plot_plan(env, trajectory, "plan_visualization.png")
        print("\n--- 规划结束 ---")

    if isinstance(env, DynamicEnv):
        print("\n--- 仿真循环 ---")
        for _ in range(100):
            if not env.step():
                break
        env.set_velocity(0, 0) # 停止机器人
        env.close()
        print("\n--- 规划结束 ---")


if __name__ == "__main__":
    main()
