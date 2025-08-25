import argparse
import torch
import time
from src.environments._2d_env_vectorized import Environment2DVectorized
from src.planning.dynamics_vectorized import DifferentiableDynamicsModelVectorized
from src.planning.planner import GPUParallelPlanner
from src.visualization.plot import plot_plan

class RobotBrain:
    """
    A robot brain that uses a GPU-based parallel planner to navigate in a 2D environment.
    """
    def __init__(self, planner: GPUParallelPlanner, dynamics_model: DifferentiableDynamicsModelVectorized) -> None:
        self.planner = planner
        self.dynamics_model = dynamics_model

    def plan(self, start_pos: torch.Tensor, goal_pos: torch.Tensor):
        """
        Generates a high-level action plan based on observations.
        """
        return self.planner.plan(start_pos, goal_pos)

    def get_trajectory(self, start_pos: torch.Tensor, plan: torch.Tensor):
        """
        Computes the trajectory given a starting position and a plan.
        """
        # The rollout function expects a batch of plans, so we unsqueeze the plan.
        trajectory = self.dynamics_model.rollout_plans(start_pos, plan.unsqueeze(0)).squeeze(0)
        # Prepend the start position to the trajectory for plotting
        full_trajectory = torch.cat([start_pos.unsqueeze(0), trajectory], dim=0)
        return full_trajectory

    def act(self, plan: torch.Tensor) -> None:
        """
        "Executes" a given action plan by printing it.
        """
        print("\nExecuting final plan (action sequence):")
        print(plan)

def main() -> None:
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run the GPU-based 2D planner (Vectorized).")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run the planner on (cuda or cpu).')
    parser.add_argument('--safety-weight', type=float, default=0.5,
                        help='Weight for the safety cost in the planner.')
    args = parser.parse_args()
    device = args.device

    # --- Setup ---
    # Check for GPU availability if requested, fall back to CPU if not.
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU. For better performance, run on a CUDA-enabled GPU.")
        device = 'cpu'

    print(f"Using device: {device}")

    # --- Environment ---
    print("Setting up 2D vectorized environment...")
    start_pos = torch.tensor([0.0, 0.0], device=device)
    goal_pos = torch.tensor([1.0, 1.0], device=device)
    # Define polygon vertices in counter-clockwise (CCW) order.
    obstacles = [
        # A square obstacle
        torch.tensor([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]),
        # A triangular obstacle
        torch.tensor([[0.1, 0.7], [0.3, 0.7], [0.2, 0.9]]),
        # Another rectangle
        torch.tensor([[0.7, 0.1], [0.9, 0.1], [0.9, 0.3], [0.7, 0.3]]),
    ]
    env = Environment2DVectorized(start_pos, goal_pos, obstacles, device=device)
    print("Environment created.")

    # --- Models and Planner ---
    print("Initializing models...")
    dynamics_model = DifferentiableDynamicsModelVectorized(env, safety_weight=args.safety_weight)

    # Use different parameters for CPU vs GPU for timely execution
    if device == 'cpu':
        print("Using CPU-friendly parameters (fewer plans and iterations).")
        num_plans = 512
        iterations = 30
    else:
        num_plans = 4096
        iterations = 100

    planner = GPUParallelPlanner(
        dynamics_model,
        num_plans=num_plans,
        plan_horizon=20,
        iterations=iterations
    )
    print("Planner initialized.")

    # --- Robot Brain ---
    brain = RobotBrain(planner, dynamics_model)

    # --- Run ---
    print("\n--- Starting Planning ---")
    start_time = time.time()
    plan = brain.plan(env.robot_pos, env.goal_pos)
    end_time = time.time()
    print(f"\nPlanning took {end_time - start_time:.4f} seconds.")

    trajectory = brain.get_trajectory(env.robot_pos, plan)
    brain.act(plan)

    # --- Visualization ---
    print("\n--- Generating Visualization ---")
    plot_plan(env, trajectory, "plan_visualization_vectorized.png")
    print("\n--- Planning Complete ---")

if __name__ == "__main__":
    main()
