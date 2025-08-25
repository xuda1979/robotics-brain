import torch

class CEMPlanner:
    """
    A planner that uses the Cross-Entropy Method (CEM) to find an optimal action plan.
    """
    def __init__(self, dynamics_model, num_plans: int = 2048, plan_horizon: int = 12, iterations: int = 10, num_elites: int = 64):
        """
        Initializes the CEM planner.

        Args:
            dynamics_model: An instance of a differentiable dynamics model.
            num_plans (int): The number of candidate plans to sample in each iteration.
            plan_horizon (int): The number of steps in each plan.
            iterations (int): The number of optimization iterations.
            num_elites (int): The number of top plans to use for updating the distribution.
        """
        self.dynamics_model = dynamics_model
        self.num_plans = num_plans
        self.plan_horizon = plan_horizon
        self.iterations = iterations
        self.num_elites = num_elites
        self.device = self.dynamics_model.env.device

    def plan(self, start_pos, goal_pos):
        """
        Generates an optimized plan using the Cross-Entropy Method.

        Args:
            start_pos (torch.Tensor): The robot's starting position, shape (2,).
            goal_pos (torch.Tensor): The target position, shape (2,).

        Returns:
            torch.Tensor: The final optimized plan, shape (plan_horizon, 2).
        """
        # Initialize the distribution (mean and standard deviation) for the actions
        mean = torch.zeros(self.plan_horizon, 2, device=self.device)
        std = torch.ones(self.plan_horizon, 2, device=self.device)

        print(f"Running CEM planner for {self.iterations} iterations...")

        for i in range(self.iterations):
            # 1. Sample action sequences from the current distribution
            # We add a small amount of noise to the standard deviation for exploration
            # and to prevent it from collapsing to zero.
            # Using torch.distributions.Normal for independent sampling is more stable.
            std_reg = std + 1e-6 # Add regularization
            distribution = torch.distributions.Normal(mean, std_reg)
            sampled_plans = distribution.sample((self.num_plans,))

            # 2. Score the sampled plans
            scores = self.dynamics_model.score_plans(start_pos, goal_pos, sampled_plans)

            # 3. Select the elite plans (the ones with the highest scores)
            _, elite_indices = torch.topk(scores, self.num_elites)
            elite_plans = sampled_plans[elite_indices]

            # 4. Update the distribution based on the elite plans
            mean = torch.mean(elite_plans, dim=0)
            std = torch.std(elite_plans, dim=0)

            print(f"Iteration {i+1}/{self.iterations}, Best Score: {scores.max().item():.2f}")


        # The final plan is the mean of the distribution after the last iteration
        print(f"CEM planning complete. Final plan score: {scores.max().item():.2f}")
        return mean
