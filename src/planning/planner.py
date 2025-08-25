import torch

class GPUParallelPlanner:
    """
    A planner that samples and scores candidate action plans on the GPU and refines them.
    """
    def __init__(self, dynamics_model, num_plans: int = 2048, plan_horizon: int = 12, iterations: int = 10):
        """
        Initializes the planner.

        Args:
            dynamics_model: An instance of a differentiable dynamics model.
            num_plans (int): The number of candidate plans to sample in parallel.
            plan_horizon (int): The number of steps in each plan.
            iterations (int): The number of refinement iterations.
        """
        self.dynamics_model = dynamics_model
        self.num_plans = num_plans
        self.plan_horizon = plan_horizon
        self.iterations = iterations
        self.device = self.dynamics_model.env.device

    def sample_and_evaluate_plans(self, start_pos, goal_pos):
        """Sample random plans and evaluate them in parallel on the GPU."""
        # Sample random action sequences (Gaussian noise)
        # The scale of the noise (0.1 here) is a hyperparameter that can be tuned.
        plans = torch.randn(self.num_plans, self.plan_horizon, 2, device=self.device) * 0.1

        # Evaluate each plan using the dynamics model. Higher score is better.
        scores = self.dynamics_model.score_plans(start_pos, goal_pos, plans)

        best_idx = torch.argmax(scores)
        return plans[best_idx], scores[best_idx]

    def refine_plan(self, start_pos, goal_pos, plan: torch.Tensor, learning_rate: float = 0.1):
        """Refine the selected plan via gradient ascent on the plan score."""
        refined_plan = plan.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([refined_plan], lr=learning_rate)

        for _ in range(self.iterations):
            optimizer.zero_grad()
            # We need to score a batch of size 1
            score = self.dynamics_model.score_plans(start_pos, goal_pos, refined_plan.unsqueeze(0))
            # We want to maximize the score, so we use -score for minimization
            loss = -score.mean()
            loss.backward()
            optimizer.step()

        return refined_plan.detach()

    def plan(self, start_pos, goal_pos):
        """
        Generate and refine a plan for a given start and goal.

        Args:
            start_pos (torch.Tensor): The starting position of the robot, shape (2,).
            goal_pos (torch.Tensor): The goal position, shape (2,).

        Returns:
            torch.Tensor: The final, refined plan, shape (plan_horizon, 2).
        """
        # 1. Sample a batch of plans and find the best one
        print(f"Sampling {self.num_plans} plans...")
        best_plan, best_score = self.sample_and_evaluate_plans(start_pos, goal_pos)
        print(f"Best initial plan score: {best_score:.2f}")

        # 2. Refine the best plan using gradient-based optimization
        print(f"Refining plan for {self.iterations} iterations...")
        refined_plan = self.refine_plan(start_pos, goal_pos, best_plan)

        # Optional: score the final plan
        final_score = self.dynamics_model.score_plans(start_pos, goal_pos, refined_plan.unsqueeze(0)).squeeze()
        print(f"Final refined plan score: {final_score:.2f}")

        return refined_plan
