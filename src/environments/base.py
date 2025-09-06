from abc import ABC, abstractmethod
import torch

class BaseEnv(ABC):
    """
    Abstract base class for all environments.
    Defines the essential properties and methods required for planning.
    """

    @property
    @abstractmethod
    def robot_pos(self) -> torch.Tensor:
        """Returns the current position of the robot."""
        pass

    @property
    @abstractmethod
    def goal_pos(self) -> torch.Tensor:
        """Returns the position of the target."""
        pass

    @abstractmethod
    def check_collision(self, pos_batch: torch.Tensor) -> torch.Tensor:
        """
        Checks for collisions between a batch of positions and obstacles.
        """
        pass

class DynamicEnv(BaseEnv):
    """
    Abstract base class for dynamic environments that can be stepped through time.
    Inherits from BaseEnv and adds methods for simulation control.
    """

    @abstractmethod
    def set_velocity(self, left: float, right: float) -> None:
        """Sets the velocity of the robot's motors."""
        pass

    @abstractmethod
    def step(self) -> bool:
        """Performs one simulation step."""
        pass

    def close(self) -> None:
        """Cleans up the environment's resources."""
        # Provide a default implementation as not all dynamic envs may need cleanup.
        pass
