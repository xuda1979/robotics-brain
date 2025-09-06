import torch
import pytest
from src.environments._2d_env import Environment2D

@pytest.fixture
def default_env():
    """A pytest fixture to create a default 2D environment for testing."""
    start_pos = torch.tensor([0.0, 0.0])
    goal_pos = torch.tensor([1.0, 1.0])
    obstacles = [
        torch.tensor([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]), # A square
        torch.tensor([[0.1, 0.7], [0.3, 0.7], [0.2, 0.9]]), # A triangle
    ]
    return Environment2D(start_pos, goal_pos, obstacles, device="cpu")

def test_vectorized_collision_checker_inside(default_env):
    """
    Tests that the vectorized collision checker correctly identifies points
    that are inside an obstacle.
    """
    # Points inside the square
    points_inside = torch.tensor([
        [0.5, 0.5],
        [0.45, 0.55],
    ])
    collisions = default_env.check_collision(points_inside)
    assert torch.all(collisions), "Expected all points to be in collision"

    # Points inside the triangle
    points_inside_triangle = torch.tensor([
        [0.2, 0.8],
    ])
    collisions_triangle = default_env.check_collision(points_inside_triangle)
    assert torch.all(collisions_triangle), "Expected point in triangle to be in collision"


def test_vectorized_collision_checker_outside(default_env):
    """
    Tests that the vectorized collision checker correctly identifies points
    that are outside the obstacles.
    """
    points_outside = torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0],
        [0.7, 0.7],
        [0.3, 0.3],
    ])
    collisions = default_env.check_collision(points_outside)
    assert not torch.any(collisions), "Expected all points to be outside obstacles"

def test_vectorized_collision_checker_on_edge(default_env):
    """
    Tests that points on the edge of an obstacle are correctly identified
    as being in collision.
    """
    points_on_edge = torch.tensor([
        [0.4, 0.5], # On the left edge of the square
        [0.5, 0.4], # On the bottom edge of the square
    ])
    collisions = default_env.check_collision(points_on_edge)
    assert torch.all(collisions), "Expected points on the edge to be in collision"

def test_domain_randomization_creation():
    """
    Tests the creation of a random 2D environment.
    """
    env = Environment2D.create_random(device="cpu")

    # Check that obstacles were created
    assert env.obstacles is not None
    assert len(env.obstacles) > 0

    # Check that vectorized tensors were created
    assert env.v1 is not None
    assert env.v2 is not None
    assert env.obstacles_mask is not None

    # Check that robot and goal positions are valid
    assert env.robot_pos is not None
    assert env.goal_pos is not None

    # Check that robot and goal are not inside an obstacle
    robot_in_collision = env.check_collision(env.robot_pos.unsqueeze(0))
    goal_in_collision = env.check_collision(env.goal_pos.unsqueeze(0))
    assert not robot_in_collision[0], "Randomly placed robot should not be in collision"
    assert not goal_in_collision[0], "Randomly placed goal should not be in collision"

    # Check that the distance between start and goal is reasonable
    assert torch.linalg.norm(env.robot_pos - env.goal_pos) > 0.5
