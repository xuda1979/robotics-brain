import torch

class GPUParallelPlanner:
    """
    一个在GPU上采样、评分并优化候选行动计划的规划器。
    """
    def __init__(self, dynamics_model, num_plans: int = 2048, plan_horizon: int = 12, iterations: int = 10):
        """
        初始化规划器。

        Args:
            dynamics_model: 一个可微动力学模型的实例。
            num_plans (int): 并行采样的候选计划数量。
            plan_horizon (int): 每个计划中的步数。
            iterations (int): 优化迭代的次数。
        """
        self.dynamics_model = dynamics_model
        self.num_plans = num_plans
        self.plan_horizon = plan_horizon
        self.iterations = iterations
        self.device = self.dynamics_model.env.device

    def sample_and_evaluate_plans(self, start_pos, goal_pos):
        """在GPU上并行采样随机计划并进行评估。"""
        # 采样随机动作序列（高斯噪声）
        # 噪声的规模（此处为0.1）是一个可以调整的超参数。
        plans = torch.randn(self.num_plans, self.plan_horizon, 2, device=self.device) * 0.1

        # 使用动力学模型评估每个计划。分数越高越好。
        scores = self.dynamics_model.score_plans(start_pos, goal_pos, plans)

        best_idx = torch.argmax(scores)
        return plans[best_idx], scores[best_idx]

    def refine_plan(self, start_pos, goal_pos, plan: torch.Tensor, learning_rate: float = 0.1):
        """通过对计划分数进行梯度提升来优化所选计划。"""
        refined_plan = plan.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([refined_plan], lr=learning_rate)

        for _ in range(self.iterations):
            optimizer.zero_grad()
            # 我们需要对一个大小为1的批次进行评分
            score = self.dynamics_model.score_plans(start_pos, goal_pos, refined_plan.unsqueeze(0))
            # 我们想要最大化分数，所以我们使用-score进行最小化
            loss = -score.mean()
            loss.backward()
            optimizer.step()

        return refined_plan.detach()

    def plan(self, start_pos, goal_pos):
        """
        为给定的起点和终点生成并优化一个计划。

        Args:
            start_pos (torch.Tensor): 机器人的起始位置，形状为 (2,)。
            goal_pos (torch.Tensor): 目标位置，形状为 (2,)。

        Returns:
            torch.Tensor: 最终优化后的计划，形状为 (plan_horizon, 2)。
        """
        # 1. 采样一批计划并找到最好的一个
        print(f"正在采样 {self.num_plans} 个计划...")
        best_plan, best_score = self.sample_and_evaluate_plans(start_pos, goal_pos)
        print(f"最佳初始计划得分: {best_score:.2f}")

        # 2. 使用基于梯度的优化方法优化最佳计划
        print(f"正在优化计划，迭代 {self.iterations} 次...")
        refined_plan = self.refine_plan(start_pos, goal_pos, best_plan)

        # 可选：对最终计划进行评分
        final_score = self.dynamics_model.score_plans(start_pos, goal_pos, refined_plan.unsqueeze(0)).squeeze()
        print(f"最终优化计划得分: {final_score:.2f}")

        return refined_plan
