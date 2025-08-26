import torch
from controller import Supervisor, DistanceSensor, Motor

class WebotsEnvironment:
    """
    Represents the Webots simulation environment for the robot.
    """
    def __init__(self, world_path: str, device: str = "cpu"):
        """
        Initializes the Webots environment.

        Args:
            world_path (str): Path to the Webots world file (.wbt).
            device (str): The device to use for torch tensors ('cpu' or 'cuda').
        """
        self.device = device

        # Initialize the Supervisor instance
        self.robot = Supervisor()

        # Get the time step of the current world.
        self.timestep = int(self.robot.getBasicTimeStep())

        # Get device handles
        self.ds_front = self.robot.getDevice("ds_front")
        self.ds_front.enable(self.timestep)

        # Get motor handles
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Get the target node
        self.target_node = self.robot.getFromDef("TARGET")

        # Get obstacle nodes and their properties
        self.obstacles = []
        for i in range(1, 5):
            node = self.robot.getFromDef(f"WALL{i}")
            if node:
                pos = node.getPosition()
                size = node.getField("geometry").getSFNode().getField("size").getSFVec3f()
                self.obstacles.append({'pos': torch.tensor([pos[0], pos[1]], device=self.device),
                                       'size': torch.tensor([size[0], size[1]], device=self.device)})

    @property
    def robot_pos(self) -> torch.Tensor:
        """
        Returns the current position of the robot.
        """
        pos = self.robot.getSelf().getPosition()
        return torch.tensor(pos[:2], device=self.device, dtype=torch.float32)

    @property
    def goal_pos(self) -> torch.Tensor:
        """
        Returns the position of the target.
        """
        if self.target_node:
            pos = self.target_node.getPosition()
            return torch.tensor(pos[:2], device=self.device, dtype=torch.float32)
        return torch.zeros(2, device=self.device, dtype=torch.float32)

    def check_collision(self, pos_batch: torch.Tensor) -> torch.Tensor:
        """
        Checks for collisions between a batch of positions and the obstacles.
        """
        collisions = torch.zeros(pos_batch.shape[0], dtype=torch.bool, device=self.device)
        for obs in self.obstacles:
            # AABB collision check
            half_size = obs['size'] / 2
            min_corner = obs['pos'] - half_size
            max_corner = obs['pos'] + half_size

            in_x = (pos_batch[:, 0] > min_corner[0]) & (pos_batch[:, 0] < max_corner[0])
            in_y = (pos_batch[:, 1] > min_corner[1]) & (pos_batch[:, 1] < max_corner[1])

            collisions = collisions | (in_x & in_y)

        return collisions

    def set_velocity(self, left: float, right: float):
        """
        Sets the velocity of the robot's motors.
        """
        self.left_motor.setVelocity(left)
        self.right_motor.setVelocity(right)

    def step(self):
        """
        Performs one simulation step.
        """
        if self.robot.step(self.timestep) == -1:
            return False
        return True

    def close(self):
        """
        Cleans up the Webots simulation.
        """
        self.robot.simulationQuit(0)
