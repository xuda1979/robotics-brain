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

        if self.obstacles:
            # 为了高效的批处理，我们可以将多边形填充到相同的顶点数
            # 并将它们堆叠成一个单一的张量。这是一个该逻辑的占位符。
            # 目前，我们以循环方式处理它们，这在GPU上效率较低。
            # 这也为以后的距离计算预先计算了边。
            self.obstacle_edges = []
            for poly in self.obstacles:
                # 边由顶点i定义到顶点i+1（循环）
                edges = torch.roll(poly, -1, dims=0) - poly
                self.obstacle_edges.append(edges)

    @property
    def robot_pos(self) -> torch.Tensor:
        return self._robot_pos

    @property
    def goal_pos(self) -> torch.Tensor:
        return self._goal_pos


    def check_collision(self, pos_batch: torch.Tensor) -> torch.Tensor:
        """
        检查批处理中的位置是否与任何凸多边形障碍物发生碰撞。
        如果一个点位于所有边的同一侧，则该点在凸多边形内部。

        Args:
            pos_batch (torch.Tensor): 一批要检查的位置，形状为 (batch_size, 2)。

        Returns:
            torch.Tensor: 一个形状为 (batch_size,) 的布尔张量，指示是否发生碰撞。
        """
        if not self.obstacles:
            return torch.zeros(pos_batch.shape[0], dtype=torch.bool, device=self.device)

        # 将碰撞张量初始化为全False
        batch_size = pos_batch.shape[0]
        total_collisions = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # 这个循环效率低下，应为生产代码进行向量化。
        # 然而，它目前正确地实现了逻辑。
        for poly in self.obstacles:
            # 对每个多边形，检查是否有任何点在内部。
            # 多边形的顶点
            v1 = poly  # 形状: (num_vertices, 2)
            v2 = torch.roll(v1, -1, dims=0) # 形状: (num_vertices, 2)

            # 扩展点和顶点以进行批处理比较
            # pos_batch: (batch_size, 1, 2)
            # v1, v2: (1, num_vertices, 2)
            p = pos_batch.unsqueeze(1)
            v1 = v1.unsqueeze(0)
            v2 = v2.unsqueeze(0)

            # 向量化的叉积计算以确定“朝向”
            # (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
            cross_product = (v2[..., 0] - v1[..., 0]) * (p[..., 1] - v1[..., 1]) - \
                            (v2[..., 1] - v1[..., 1]) * (p[..., 0] - v1[..., 0])

            # 对于一个逆时针顺序的多边形，如果所有的叉积都是非负的，那么一个点就在其内部。
            # 一个小的epsilon可以处理边上点的浮点不精确问题。
            is_inside = torch.all(cross_product >= -1e-6, dim=1) # 形状: (batch_size,)
            total_collisions = total_collisions | is_inside

        return total_collisions
