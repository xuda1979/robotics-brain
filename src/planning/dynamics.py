import torch
from src.environments.base import BaseEnv

class DifferentiableDynamicsModel:
    """
    一个用于2D环境的可微动力学模型。
    该模型预测行动计划的结果并对其进行评分。
    """
    def __init__(self, env: BaseEnv, safety_weight: float = 0.1):
        self.env = env
        self.safety_weight = safety_weight

    def rollout_plans(self, start_pos: torch.Tensor, plans: torch.Tensor):
        """
        为一批计划模拟轨迹。

        Args:
            start_pos (torch.Tensor): 机器人的起始位置，形状为 (2,)。
            plans (torch.Tensor): 一批行动计划，形状为 (num_plans, plan_horizon, 2)。
                                每个动作是一个2D速度向量。

        Returns:
            torch.Tensor: 每个计划的轨迹，形状为 (num_plans, plan_horizon, 2)。
        """
        # 注意：`torch.cumsum`是可微的。
        # 我们将起始位置添加到动作的累积和中以获得轨迹。
        return start_pos.unsqueeze(0).unsqueeze(0) + torch.cumsum(plans, dim=1)


    def _point_to_segment_dist(self, points: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor):
        """
        计算一批点到一条线段的最小距离。
        此函数是可微的。
        """
        edge_vec = v2 - v1
        point_vec = points - v1

        # 将point_vec投影到edge_vec上
        # t是归一化的投影长度。
        l2 = torch.sum(edge_vec**2)
        # 添加epsilon以避免零长度边的除零错误
        t = torch.sum(point_vec * edge_vec, dim=1) / (l2 + 1e-6)

        # 将t限制在[0, 1]范围内，以找到线段上最近的点
        t_clamped = torch.clamp(t, 0, 1)

        # 线段上的最近点
        closest_point = v1 + t_clamped.unsqueeze(1) * edge_vec

        # 返回原始点到线段上最近点的距离
        return torch.linalg.norm(points - closest_point, dim=1)


    def score_plans(self, start_pos: torch.Tensor, goal_pos: torch.Tensor, plans: torch.Tensor):
        """
        根据目标接近度和避障情况为一批计划评分。
        分数应相对于计划是可微的。
        """
        trajectories = self.rollout_plans(start_pos, plans)
        num_plans, plan_horizon, _ = trajectories.shape
        flat_trajectories = trajectories.reshape(num_plans * plan_horizon, 2)

        # --- 目标接近度分数 ---
        final_positions = trajectories[:, -1, :]
        dist_to_goal = torch.linalg.norm(final_positions - goal_pos.unsqueeze(0), dim=1)
        goal_score = -dist_to_goal

        # --- 碰撞分数 ---
        collisions = self.env.check_collision(flat_trajectories)
        collisions = collisions.view(num_plans, plan_horizon)
        collision_penalty = -1000.0 * torch.any(collisions, dim=1).float()

        # --- 安全分数（与多边形边的接近度） ---
        safety_penalty = torch.tensor(0.0, device=self.env.device)
        if self.env.obstacles and self.safety_weight > 0:
            # 这部分计算量大且未完全向量化。
            # 为获得最佳性能，应将其实现为批处理操作。
            min_dist_to_obstacle = torch.full((num_plans * plan_horizon,), float('inf'), device=self.env.device)

            for poly in self.env.obstacles:
                v_prev = poly[-1]
                for v_curr in poly:
                    # 计算所有轨迹点到当前边的距离
                    edge_dists = self._point_to_segment_dist(flat_trajectories, v_prev, v_curr)
                    # 更新每个点的最小距离
                    min_dist_to_obstacle = torch.min(min_dist_to_obstacle, edge_dists)
                    v_prev = v_curr

            # 我们只应考虑点在障碍物外部时的距离。
            # 然而，由于我们对碰撞有很大的惩罚，我们可以通过
            # 在任何地方都应用安全惩罚来近似这一点。更精细的方法会使用掩码。
            # 使用relu只在外部时（距离为正）惩罚接近度。
            # 碰撞检查处理内部的点。
            safe_dist = torch.relu(min_dist_to_obstacle)

            # 反比距离惩罚
            safety_cost = 1.0 / (safe_dist + 1e-6)

            # 重塑并在轨迹视界上求和
            total_safety_cost = torch.sum(safety_cost.view(num_plans, plan_horizon), dim=1)
            safety_penalty = -self.safety_weight * total_safety_cost

        return goal_score + collision_penalty + safety_penalty
