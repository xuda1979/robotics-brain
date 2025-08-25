import torch
from ..environments._2d_env import Environment2D

class DifferentiableDynamicsModel:
    """
    A differentiable dynamics model for the 2D environment.
    This model predicts the outcome of action plans and scores them.
    """
    def __init__(self, env: Environment2D):
        self.env = env

    def rollout_plans(self, start_pos: torch.Tensor, plans: torch.Tensor):
        """
        Simulates the trajectories for a batch of plans.

        Args:
            start_pos (torch.Tensor): The starting position of the robot, shape (2,).
            plans (torch.Tensor): A batch of action plans, shape (num_plans, plan_horizon, 2).
                                Each action is a 2D velocity vector.

        Returns:
            torch.Tensor: The trajectory for each plan, shape (num_plans, plan_horizon, 2).
        """
        # Note: `torch.cumsum` is differentiable.
        # We add the start position to the cumulative sum of actions to get the trajectory.
        # This is more efficient than a for-loop.
        return start_pos.unsqueeze(0).unsqueeze(0) + torch.cumsum(plans, dim=1)


    def score_plans(self, start_pos: torch.Tensor, goal_pos: torch.Tensor, plans: torch.Tensor):
        """
        Scores a batch of plans based on goal proximity and obstacle avoidance.
        The score should be differentiable with respect to the plans.

        Args:
            start_pos (torch.Tensor): The starting position of the robot, shape (2,).
            goal_pos (torch.Tensor): The goal position, shape (2,).
            plans (torch.Tensor): A batch of action plans, shape (num_plans, plan_horizon, 2).

        Returns:
            torch.Tensor: A score for each plan, shape (num_plans,).
        """
        trajectories = self.rollout_plans(start_pos, plans)
        num_plans, plan_horizon, _ = trajectories.shape

        # --- Goal Proximity Score ---
        # Distance from the final position to the goal
        final_positions = trajectories[:, -1, :]
        dist_to_goal = torch.linalg.norm(final_positions - goal_pos.unsqueeze(0), dim=1)
        # Higher score for being closer to the goal. We use a negative sign because we want to maximize the score.
        goal_score = -dist_to_goal

        # --- Collision Score ---
        # Flatten trajectories to check all points for collision at once
        flat_trajectories = trajectories.reshape(num_plans * plan_horizon, 2)
        collisions = self.env.check_collision(flat_trajectories)
        # Reshape back to (num_plans, plan_horizon)
        collisions = collisions.view(num_plans, plan_horizon)
        # Penalize plans that have any collision. The large penalty ensures collision avoidance.
        collision_penalty = -1000.0 * torch.any(collisions, dim=1).float()

        return goal_score + collision_penalty
