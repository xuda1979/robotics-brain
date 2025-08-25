import torch
from ..environments._2d_env_vectorized import Environment2DVectorized

class DifferentiableDynamicsModelVectorized:
    """
    A differentiable dynamics model for the 2D environment, optimized for vectorized operations.
    This model predicts the outcomes of action plans and scores them.
    """
    def __init__(self, env: Environment2DVectorized, safety_weight: float = 0.1, action_weight: float = 0.01):
        self.env = env
        self.safety_weight = safety_weight
        self.action_weight = action_weight

    def rollout_plans(self, start_pos: torch.Tensor, plans: torch.Tensor):
        """
        Simulates trajectories for a batch of plans.

        Args:
            start_pos (torch.Tensor): The robot's starting position, shape (2,).
            plans (torch.Tensor): A batch of action plans, shape (num_plans, plan_horizon, 2).
                                Each action is a 2D velocity vector.

        Returns:
            torch.Tensor: The trajectories for each plan, shape (num_plans, plan_horizon, 2).
        """
        return start_pos.unsqueeze(0).unsqueeze(0) + torch.cumsum(plans, dim=1)

    def _point_to_segment_dist_vectorized(self, points: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor):
        """
        Calculates the minimum distance from a batch of points to a batch of line segments.
        This function is fully vectorized and differentiable.

        Args:
            points (torch.Tensor): Batch of points, shape (num_points, 2).
            v1 (torch.Tensor): Batch of first vertices of segments, shape (num_segments, 2).
            v2 (torch.Tensor): Batch of second vertices of segments, shape (num_segments, 2).

        Returns:
            torch.Tensor: Minimum distance from each point to each segment, shape (num_points, num_segments).
        """
        num_points, _ = points.shape
        num_segments, _ = v1.shape

        # Expand dimensions to enable broadcasting
        # points -> (num_points, 1, 2)
        # v1, v2 -> (1, num_segments, 2)
        points = points.unsqueeze(1)
        v1 = v1.unsqueeze(0)
        v2 = v2.unsqueeze(0)

        edge_vec = v2 - v1
        point_vec = points - v1

        # Project point_vec onto edge_vec
        # t is the normalized projection length.
        l2 = torch.sum(edge_vec**2, dim=2)
        t = torch.sum(point_vec * edge_vec, dim=2) / (l2 + 1e-6)

        # Clamp t to the [0, 1] range to find the closest point on the segment
        t_clamped = torch.clamp(t, 0, 1)

        # Closest point on the segment
        closest_point = v1 + t_clamped.unsqueeze(2) * edge_vec

        # Return the distance from the original point to the closest point on the segment
        return torch.linalg.norm(points - closest_point, dim=2)

    def score_plans(self, start_pos: torch.Tensor, goal_pos: torch.Tensor, plans: torch.Tensor):
        """
        Scores a batch of plans based on goal proximity and obstacle avoidance.
        The score is differentiable with respect to the plans.
        """
        trajectories = self.rollout_plans(start_pos, plans)
        num_plans, plan_horizon, _ = trajectories.shape
        flat_trajectories = trajectories.reshape(num_plans * plan_horizon, 2)

        # --- Goal Proximity Score ---
        final_positions = trajectories[:, -1, :]
        dist_to_goal = torch.linalg.norm(final_positions - goal_pos.unsqueeze(0), dim=1)
        goal_score = -dist_to_goal

        # --- Collision Score ---
        # Uses the vectorized collision check from the new environment
        collisions = self.env.check_collision(flat_trajectories)
        collisions = collisions.view(num_plans, plan_horizon)
        collision_penalty = -1000.0 * torch.any(collisions, dim=1).float()

        # --- Safety Score (Proximity to Polygon Edges) ---
        safety_penalty = torch.tensor(0.0, device=self.env.device)
        if self.env.obstacles and self.safety_weight > 0:
            # This part is now fully vectorized.

            # Calculate distances from all trajectory points to all obstacle edges
            # all_dists shape: (num_plans * plan_horizon, total_edges)
            all_dists = self._point_to_segment_dist_vectorized(flat_trajectories, self.env.v1, self.env.v2)

            # Find the minimum distance from each point to any edge
            # min_dist_to_obstacle shape: (num_plans * plan_horizon,)
            min_dist_to_obstacle, _ = torch.min(all_dists, dim=1)

            # Invert the distance to get a cost (higher cost for smaller distance)
            safety_cost = 1.0 / (min_dist_to_obstacle + 1e-6)

            # Reshape and sum over the plan horizon
            total_safety_cost = torch.sum(safety_cost.view(num_plans, plan_horizon), dim=1)
            safety_penalty = -self.safety_weight * total_safety_cost

        # --- Action Cost ---
        # Penalize large actions to encourage smoother paths.
        action_cost = torch.sum(torch.linalg.norm(plans, dim=2), dim=1)
        action_penalty = -self.action_weight * action_cost

        return goal_score + collision_penalty + safety_penalty + action_penalty
