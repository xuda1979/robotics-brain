import torch
from ..environments._2d_env import Environment2D

class DifferentiableDynamicsModel:
    """
    A differentiable dynamics model for the 2D environment.
    This model predicts the outcome of action plans and scores them.
    """
    def __init__(self, env: Environment2D, safety_weight: float = 0.1):
        self.env = env
        self.safety_weight = safety_weight

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

        # --- Safety Score ---
        # Penalize proximity to obstacles to encourage safer paths.
        safety_penalty = torch.tensor(0.0, device=self.env.device)
        if self.env.obstacle_centers.shape[0] > 0 and self.safety_weight > 0:
            # Reshape for broadcasting:
            # flat_trajectories: (num_plans * plan_horizon, 1, 2)
            # obstacle_centers: (1, num_obstacles, 2)
            expanded_trajectories = flat_trajectories.unsqueeze(1)
            expanded_centers = self.env.obstacle_centers.unsqueeze(0)

            # Calculate distance from each point to each obstacle center
            # dist_to_centers shape: (num_plans * plan_horizon, num_obstacles)
            dist_to_centers = torch.linalg.norm(expanded_trajectories - expanded_centers, dim=2)

            # Subtract radius to get distance from obstacle surface
            dist_to_surface = dist_to_centers - self.env.obstacle_radii.unsqueeze(0)

            # Use a clipped inverse distance penalty: 1 / (dist + epsilon)
            # We clip the distance at a minimum value to avoid huge penalties and division by zero.
            # The penalty is only applied if the distance is positive and less than a certain threshold.
            # Let's use a simple inverse square distance for a stronger repulsive force.
            # We use `torch.relu` to only consider positive distances (i.e., outside the obstacle).
            safe_dist = torch.relu(dist_to_surface)
            # Add a small epsilon to avoid division by zero
            safety_cost = 1.0 / (safe_dist + 1e-6)

            # We sum the costs from all obstacles for each point in the trajectory.
            # Then, we sum the costs over the entire trajectory for each plan.
            # Reshape back to (num_plans, plan_horizon, num_obstacles)
            safety_cost = safety_cost.view(num_plans, plan_horizon, -1)
            # Sum over obstacles and trajectory horizon
            total_safety_cost = torch.sum(safety_cost, dim=[1, 2])
            safety_penalty = -self.safety_weight * total_safety_cost


        return goal_score + collision_penalty + safety_penalty
