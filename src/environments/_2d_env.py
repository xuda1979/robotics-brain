import torch
from typing import List
from .base import BaseEnv

class Environment2D(BaseEnv):
    """
    表示一个带有机器人、目标和多边形障碍物的2D环境。
    """
    def __init__(self, robot_pos: torch.Tensor, goal_pos: torch.Tensor, obstacles: List[torch.Tensor], device="cpu"):
        """
        初始化2D环境。

        Args:
            robot_pos (torch.Tensor): 机器人的起始位置 (x, y)。
            goal_pos (torch.Tensor): 目标位置 (x, y)。
            obstacles (List[torch.Tensor]): 障碍物列表，每个障碍物是一个定义了
                凸多边形（顶点按逆时针顺序）的顶点张量 (num_vertices, 2)。
            device (str): 存储张量的设备 ('cpu' 或 'cuda')。
        """
        self._robot_pos = robot_pos.to(device)
        self._goal_pos = goal_pos.to(device)
        self.device = device
        self.obstacles = [obs.to(device) for obs in obstacles]
        self._vectorize_obstacles()

    def _vectorize_obstacles(self):
        """
        一个辅助函数，用于为高效的、向量化的碰撞检测准备障碍物数据。
        """
        self.v1 = None
        self.v2 = None
        self.obstacles_mask = None

        if self.obstacles:
            max_vertices = max(obs.shape[0] for obs in self.obstacles) if self.obstacles else 0
            if max_vertices == 0:
                return

            padded_v1_list = []
            padded_v2_list = []
            masks = []

            for obs in self.obstacles:
                num_vertices = obs.shape[0]
                padding_size = max_vertices - num_vertices

                v1 = obs
                v2 = torch.roll(obs, -1, dims=0)

                v1_padding = torch.zeros((padding_size, 2), device=self.device)
                v2_padding = torch.zeros((padding_size, 2), device=self.device)

                padded_v1 = torch.cat([v1, v1_padding], dim=0)
                padded_v2 = torch.cat([v2, v2_padding], dim=0)

                padded_v1_list.append(padded_v1)
                padded_v2_list.append(padded_v2)

                mask = torch.ones(max_vertices, device=self.device, dtype=torch.bool)
                if padding_size > 0:
                    mask[-padding_size:] = False
                masks.append(mask)

            self.v1 = torch.stack(padded_v1_list)
            self.v2 = torch.stack(padded_v2_list)
            self.obstacles_mask = torch.stack(masks)

    @classmethod
    def create_random(cls, device="cpu", num_obstacles_range=(3, 10),
                      vertex_range=(3, 6), size_range=(0.05, 0.15)):
        """
        一个工厂方法，用于创建一个带有随机障碍物、起始点和目标点的环境。
        """
        # 创建一个临时的空环境以使用其方法
        env = cls(torch.zeros(2, device=device), torch.zeros(2, device=device), [], device=device)

        # 生成随机障碍物
        num_obstacles = torch.randint(num_obstacles_range[0], num_obstacles_range[1] + 1, (1,)).item()
        new_obstacles = []
        for _ in range(num_obstacles):
            # 通过在一个圆中生成点并按角度排序来创建随机凸多边形
            center = torch.rand(2, device=device) * 0.8 + 0.1
            num_vertices = torch.randint(vertex_range[0], vertex_range[1] + 1, (1,)).item()

            radii = torch.rand(num_vertices, device=device) * (size_range[1] - size_range[0]) + size_range[0]
            angles = torch.sort(torch.rand(num_vertices, device=device) * 2 * torch.pi)[0]

            x = center[0] + radii * torch.cos(angles)
            y = center[1] + radii * torch.sin(angles)

            obstacle = torch.stack([x, y], dim=1)
            new_obstacles.append(obstacle)

        env.obstacles = new_obstacles
        env._vectorize_obstacles() # 必须在检查碰撞之前向量化

        # 生成随机的起始点和目标点，确保它们不在障碍物内部
        while True:
            pos = torch.rand(2, device=device)
            if not env.check_collision(pos.unsqueeze(0))[0]:
                env._robot_pos = pos
                break

        while True:
            pos = torch.rand(2, device=device)
            if not env.check_collision(pos.unsqueeze(0))[0]:
                if torch.linalg.norm(env.robot_pos - pos) > 0.5:
                    env._goal_pos = pos
                    break

        return env

    @property
    def robot_pos(self) -> torch.Tensor:
        return self._robot_pos

    @property
    def goal_pos(self) -> torch.Tensor:
        return self._goal_pos

    def check_collision(self, pos_batch: torch.Tensor) -> torch.Tensor:
        """
        检查批处理中的位置是否与任何凸多边形障碍物发生碰撞。
        此实现是完全向量化的，以在GPU上实现高性能。

        Args:
            pos_batch (torch.Tensor): 一批要检查的位置，形状为 (batch_size, 2)。

        Returns:
            torch.Tensor: 一个形状为 (batch_size,) 的布尔张量，指示是否发生碰撞。
        """
        if self.v1 is None:
            return torch.zeros(pos_batch.shape[0], dtype=torch.bool, device=self.device)

        batch_size = pos_batch.shape[0]
        num_obstacles, max_vertices, _ = self.v1.shape

        # --- 向量化碰撞检查 ---
        # 为广播调整形状
        # p: (batch_size, 1, 1, 2)
        p = pos_batch.view(batch_size, 1, 1, 2)
        # v1, v2: (1, num_obstacles, max_vertices, 2)
        v1 = self.v1.view(1, num_obstacles, max_vertices, 2)
        v2 = self.v2.view(1, num_obstacles, max_vertices, 2)

        # 叉积计算以确定“朝向”
        # (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        # cross_product 形状: (batch_size, num_obstacles, max_vertices)
        cross_product = (v2[..., 0] - v1[..., 0]) * (p[..., 1] - v1[..., 1]) - \
                        (v2[..., 1] - v1[..., 1]) * (p[..., 0] - v1[..., 0])

        # 使用掩码忽略填充值。我们希望它们 >= 0 以便 `all` 通过。
        # obstacles_mask 形状: (1, num_obstacles, max_vertices)
        mask = self.obstacles_mask.view(1, num_obstacles, max_vertices)
        # 当掩码为True时，保留cross_product。当掩码为False时，使用0。
        # 这确保了填充值不会导致 `torch.all` 失败。
        final_cross_product = torch.where(mask, cross_product, 0.)

        # 对于一个逆时针顺序的多边形，如果一个点在其内部，则所有的叉积都必须为非负数。
        # is_inside_polygon 形状: (batch_size, num_obstacles)
        is_inside_polygon = torch.all(final_cross_product >= -1e-6, dim=2)

        # 如果一个点在任何一个多边形内，则发生碰撞。
        # is_inside_any_polygon 形状: (batch_size,)
        is_inside_any_polygon = torch.any(is_inside_polygon, dim=1)

        return is_inside_any_polygon
