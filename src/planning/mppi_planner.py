import torch

class MPPIPlanner:
    """
    A planner that uses Model Predictive Path Integral (MPPI) control to find an optimal action plan.
    MPPI is a sampling-based algorithm that updates the control distribution using an importance-weighted average
    of sampled trajectories, which is better at handling non-smooth costs than CEM or gradient-based methods.
    """
    def __init__(self, dynamics_model, num_plans: int = 2048, plan_horizon: int = 12, iterations: int = 10,
                 temperature: float = 0.5, noise_std: float = 0.3):
        """
        Initializes the MPPI planner.

        Args:
            dynamics_model: An instance of a differentiable dynamics model.
            num_plans (int): The number of candidate plans to sample in each iteration.
            plan_horizon (int): The number of steps in each plan.
            iterations (int): The number of optimization iterations.
            temperature (float): The temperature parameter (lambda) for the softmax weighting.
                                 Controls the selectivity of the weighting (lower = more greedy).
            noise_std (float): The standard deviation of the exploration noise.
        """
        self.dynamics_model = dynamics_model
        self.num_plans = num_plans
        self.plan_horizon = plan_horizon
        self.iterations = iterations
        self.temperature = temperature
        self.noise_std = noise_std
        self.device = self.dynamics_model.env.device

    def plan(self, start_pos, goal_pos):
        """
        Generates an optimized plan using MPPI.

        Args:
            start_pos (torch.Tensor): The robot's starting position, shape (2,).
            goal_pos (torch.Tensor): The target position, shape (2,).

        Returns:
            torch.Tensor: The final optimized plan, shape (plan_horizon, 2).
        """
        # Initialize the mean action sequence with zeros
        mean = torch.zeros(self.plan_horizon, 2, device=self.device)

        # In a full MPC loop, we would warm-start with the previous solution shifted.
        # Since this plan() method seems to be called once for the whole trajectory in this repo,
        # we start from scratch.

        print(f"Running MPPI planner for {self.iterations} iterations...")

        for i in range(self.iterations):
            # 1. Sample noise and generate candidate plans
            # shape: (num_plans, plan_horizon, 2)
            noise = torch.randn(self.num_plans, self.plan_horizon, 2, device=self.device) * self.noise_std

            # Broadcast mean to all samples: (plan_horizon, 2) -> (num_plans, plan_horizon, 2)
            sampled_plans = mean.unsqueeze(0) + noise

            # 2. Score the sampled plans
            # The dynamics model returns scores (higher is better).
            # MPPI works with costs (lower is better).
            # So cost = -score.
            scores = self.dynamics_model.score_plans(start_pos, goal_pos, sampled_plans)
            costs = -scores

            # 3. Compute weights using the softmax function
            # Subtract minimum cost for numerical stability
            min_cost = torch.min(costs)
            exp_costs = torch.exp(-(costs - min_cost) / self.temperature)

            # Normalize weights
            weights = exp_costs / (torch.sum(exp_costs) + 1e-10)

            # 4. Update the mean action sequence
            # Weighted average of the sampled plans
            # weights shape: (num_plans,) -> (num_plans, 1, 1) for broadcasting
            weighted_plans = weights.view(-1, 1, 1) * sampled_plans
            mean = torch.sum(weighted_plans, dim=0)

            print(f"Iteration {i+1}/{self.iterations}, Best Score: {scores.max().item():.2f}")

        # Final evaluation
        final_score = self.dynamics_model.score_plans(start_pos, goal_pos, mean.unsqueeze(0)).item()
        print(f"MPPI planning complete. Final plan score: {final_score:.2f}")

        return mean
