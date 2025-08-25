import torch

class Environment2D:
    """
    Represents a simple 2D environment with a robot, a goal, and obstacles.
    """
    def __init__(self, robot_pos, goal_pos, obstacles, device="cpu"):
        """
        Initializes the 2D environment.

        Args:
            robot_pos (torch.Tensor): The starting position of the robot (x, y).
            goal_pos (torch.Tensor): The goal position (x, y).
            obstacles (list[tuple[torch.Tensor, float]]): A list of obstacles, where each
                obstacle is a tuple of (center_position, radius).
            device (str): The device to store tensors on ('cpu' or 'cuda').
        """
        self.robot_pos = robot_pos.to(device)
        self.goal_pos = goal_pos.to(device)
        # Store obstacles as a tensor for efficient computation
        if obstacles:
            self.obstacle_centers = torch.stack([obs[0] for obs in obstacles]).to(device)
            self.obstacle_radii = torch.tensor([obs[1] for obs in obstacles]).to(device)
        else:
            # Handle case with no obstacles
            self.obstacle_centers = torch.empty((0, 2), device=device)
            self.obstacle_radii = torch.empty((0,), device=device)
        self.device = device

    def check_collision(self, pos_batch):
        """
        Checks if a given position collides with any obstacle.

        Args:
            pos_batch (torch.Tensor): A batch of positions to check, shape (batch_size, 2).

        Returns:
            torch.Tensor: A boolean tensor of shape (batch_size,) indicating collision.
        """
        if self.obstacle_centers.shape[0] == 0:
            return torch.zeros(pos_batch.shape[0], dtype=torch.bool, device=self.device)

        # Expand dims for broadcasting
        # pos_batch shape: (batch_size, 1, 2)
        # self.obstacle_centers shape: (1, num_obstacles, 2)
        expanded_pos = pos_batch.unsqueeze(1)
        expanded_centers = self.obstacle_centers.unsqueeze(0)

        # Calculate distances from each point to each obstacle center
        # distances shape: (batch_size, num_obstacles)
        distances = torch.linalg.norm(expanded_pos - expanded_centers, dim=2)

        # Check if any distance is less than the corresponding radius
        # collisions shape: (batch_size, num_obstacles)
        collisions = distances < self.obstacle_radii.unsqueeze(0)

        # Return true if any collision occurs for a given position
        return torch.any(collisions, dim=1)
