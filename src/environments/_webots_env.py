import torch
import subprocess
import time
from controller import Supervisor, DistanceSensor, Motor
from .base import DynamicEnv

class WebotsEnvironment(DynamicEnv):
    """
    Represents the Webots simulation environment for the robot.
    This class can now automatically launch and manage the Webots simulator process.
    """
    def __init__(self, world_path: str, device: str = "cpu", headless=False):
        """
        Initializes the Webots environment.

        Args:
            world_path (str): Path to the Webots world file (.wbt).
            device (str): The device to use for torch tensors ('cpu' or 'cuda').
            headless (bool): If True, run Webots without rendering the GUI.
        """
        self.device = device
        self.webots_process = None

        # --- Launch Webots Process ---
        try:
            webots_command = ["webots", "--batch", f"--mode=fast", "--minimize", world_path]
            if headless:
                webots_command.append("--no-rendering")

            print(f"正在启动Webots: {' '.join(webots_command)}")
            self.webots_process = subprocess.Popen(webots_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("已启动Webots进程。等待初始化...")
            time.sleep(10) # 等待Webots完全加载
        except FileNotFoundError:
            print("错误: 'webots' 命令未找到。")
            print("请确保Webots已安装并且其可执行文件在您的系统PATH中。")
            exit(1)
        except Exception as e:
            print(f"启动Webots时出错: {e}")
            exit(1)

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
        Cleans up the Webots simulation and terminates the simulator process.
        """
        print("正在关闭Webots仿真...")
        self.robot.simulationQuit(0)

        if self.webots_process:
            print("正在终止Webots进程...")
            self.webots_process.terminate()
            try:
                # 等待进程终止
                self.webots_process.wait(timeout=5)
                print("Webots进程已终止。")
            except subprocess.TimeoutExpired:
                print("Webots进程未在5秒内终止。强制终止...")
                self.webots_process.kill()
                print("Webots进程已被强制终止。")
