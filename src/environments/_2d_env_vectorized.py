import torch
from typing import List

class Environment2DVectorized:
    """
    Represents a 2D environment with a robot, a goal, and polygonal obstacles.
    This version is optimized for vectorized operations on the GPU.
    """
    def __init__(self, robot_pos: torch.Tensor, goal_pos: torch.Tensor, obstacles: List[torch.Tensor], device="cpu"):
        """
        Initializes the 2D environment.

        Args:
            robot_pos (torch.Tensor): The robot's starting position (x, y).
            goal_pos (torch.Tensor): The goal position (x, y).
            obstacles (List[torch.Tensor]): A list of obstacles, where each obstacle is a
                tensor of vertices defining a convex polygon in counter-clockwise order.
            device (str): The device to store tensors on ('cpu' or 'cuda').
        """
        self.robot_pos = robot_pos.to(device)
        self.goal_pos = goal_pos.to(device)
        self.device = device
        self.obstacles = [obs.to(device) for obs in obstacles]

        if self.obstacles:
            # --- Vectorized Obstacle Pre-computation ---
            # To enable efficient batch processing, we concatenate all obstacle edges
            # into a single tensor.

            # self.v1 will be a tensor of the first vertex of each edge, shape: (total_edges, 2)
            # self.v2 will be a tensor of the second vertex of each edge, shape: (total_edges, 2)
            # self.edge_to_obstacle_map will map each edge to an obstacle index.
            # shape: (total_edges,)

            all_v1 = []
            all_v2 = []
            edge_to_obstacle_map = []

            for i, poly in enumerate(self.obstacles):
                v1 = poly
                v2 = torch.roll(poly, -1, dims=0)
                all_v1.append(v1)
                all_v2.append(v2)
                edge_to_obstacle_map.append(torch.full((len(v1),), i, device=self.device))

            self.v1 = torch.cat(all_v1, dim=0)
            self.v2 = torch.cat(all_v2, dim=0)
            self.edge_to_obstacle_map = torch.cat(edge_to_obstacle_map, dim=0)
            self.num_obstacles = len(self.obstacles)

    def check_collision(self, pos_batch: torch.Tensor) -> torch.Tensor:
        """
        Checks if positions in a batch collide with any convex polygon obstacle.
        A point is inside a convex polygon if it is on the same side of all its edges.
        This implementation is fully vectorized.

        Args:
            pos_batch (torch.Tensor): A batch of positions to check, shape (batch_size, 2).

        Returns:
            torch.Tensor: A boolean tensor of shape (batch_size,) indicating collisions.
        """
        if not self.obstacles:
            return torch.zeros(pos_batch.shape[0], dtype=torch.bool, device=self.device)

        batch_size = pos_batch.shape[0]

        # Expand points and edges for batch comparison
        # p: (batch_size, 1, 2)
        # v1, v2: (1, total_edges, 2)
        p = pos_batch.unsqueeze(1)
        v1 = self.v1.unsqueeze(0)
        v2 = self.v2.unsqueeze(0)

        # Vectorized cross-product calculation to determine "sidedness"
        # (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        # cross_product shape: (batch_size, total_edges)
        cross_product = (v2[..., 0] - v1[..., 0]) * (p[..., 1] - v1[..., 1]) - \
                        (v2[..., 1] - v1[..., 1]) * (p[..., 0] - v1[..., 0])

        # For a CCW polygon, a point is inside if all cross-products are non-negative.
        # A small epsilon handles floating-point inaccuracies for points on an edge.
        # is_on_correct_side shape: (batch_size, total_edges)
        is_on_correct_side = cross_product >= -1e-6

        # Now we need to check if any point is inside *any* polygon.
        # This is equivalent to checking if for any obstacle, a point is on the correct
        # side of ALL of its edges.

        # We use scatter_add to count how many edges of each obstacle a point is "inside".
        # We need to reshape is_on_correct_side to be a long tensor for scatter_add
        counts = torch.zeros(batch_size, self.num_obstacles, device=self.device, dtype=torch.long)

        # Expand edge_to_obstacle_map to match the batch size
        map_expanded = self.edge_to_obstacle_map.repeat(batch_size, 1)

        # scatter_add_ requires the index tensor to be of the same size
        counts.scatter_add_(1, map_expanded, is_on_correct_side.long())

        # Get the number of edges for each obstacle
        obstacle_edge_counts = torch.tensor([len(poly) for poly in self.obstacles], device=self.device, dtype=torch.long)

        # If the count for any obstacle equals its number of edges, the point is inside.
        # is_inside_any_obstacle shape: (batch_size, num_obstacles)
        is_inside_any_obstacle = counts == obstacle_edge_counts.unsqueeze(0)

        # The final result is true if a point is inside *any* of the obstacles.
        # total_collisions shape: (batch_size,)
        total_collisions = torch.any(is_inside_any_obstacle, dim=1)

        return total_collisions
