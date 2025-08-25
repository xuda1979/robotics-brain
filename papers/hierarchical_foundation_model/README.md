# A Hierarchical Foundation Model for Generalist Robots

**Abstract**

The creation of a single, general-purpose robotic intelligence that can operate a wide variety of morphologies remains a grand challenge. We propose a Hierarchical Omni-Bodied Foundation Model (HOF-Model), an architecture designed to decouple high-level task planning from low-level motor control. The model consists of two main components: (1) a high-level, morphology-agnostic policy that interprets natural language commands and environmental context to produce a sequence of abstract skill embeddings, and (2) a low-level, morphology-specific controller that translates these skill embeddings into concrete joint torques for a given robot body. This hierarchical design allows the high-level policy to be trained on data from a vast range of different robots, learning a general understanding of tasks, while the lightweight low-level controller can be quickly fine-tuned for a new, unseen robot morphology. We believe this approach offers a scalable solution for developing generalist robotic agents.

## 1. Introduction

Foundation models have revolutionized fields like natural language processing and computer vision by learning powerful, general-purpose representations from vast datasets. A similar paradigm is emerging in robotics, with the goal of creating a single model that can solve a wide array of tasks. However, a key challenge in robotics is the diversity of robot morphologies (or "embodiments"). A policy trained to control a quadruped is not directly applicable to a robotic arm.

To overcome this, we propose a hierarchical architecture that explicitly separates task-level reasoning from body-specific control. This is inspired by how humans and animals are thought to learn: we possess a general understanding of a task (e.g., "pick up the cup") that is independent of the specific muscles we use to execute it.

Our proposed HOF-Model consists of:
*   **A High-Level Policy:** This is a transformer-based model that takes as input a natural language command (e.g., "go to the red ball"), a high-level summary of the environment (e.g., from a vision model), and learns to output a sequence of abstract "skill embeddings." These embeddings represent sub-goals or behaviors (e.g., "turn left," "move forward," "grasp"). This policy is designed to be independent of the robot's body.
*   **A Low-Level Controller:** This is a smaller, simpler neural network that takes a skill embedding from the high-level policy and the robot's current proprioceptive state (joint angles, velocities) as input. Its sole job is to output the correct motor commands (e.g., joint torques) to execute that skill on its specific body.

By training the high-level policy on data from hundreds of different simulated robots, it can learn a general, abstract "language of skills." To deploy the system on a new robot, we only need to train or fine-tune the much smaller low-level controller, a significantly more data-efficient process.

## 2. Related Work

*   **Foundation Models in Robotics:** Several recent works have explored training large, transformer-based models for robotics (e.g., Gato, RT-1, RT-2). These models typically produce actions directly from raw inputs and are often tied to a specific robot embodiment.
*   **Hierarchical Reinforcement Learning (HRL):** HRL has long explored the idea of using a hierarchy of policies, where a high-level policy sets goals for a low-level policy. Our work adapts this paradigm to the context of large-scale, multi-embodiment training. The key difference is that our hierarchy is designed to explicitly separate morphology-agnostic and morphology-specific knowledge.
*   **Skill-based Learning:** Many approaches have focused on learning a library of reusable "skills" or "motor primitives." Our work uses a continuous skill embedding space, which allows for more flexible and compositional behaviors than a discrete set of skills.

## 3. Proposed Method: HOF-Model

The HOF-Model is composed of a `HighLevelPolicy` and a `LowLevelController`, which work in tandem.

### 3.1. High-Level Policy

The high-level policy is a transformer model that processes a combination of state and language inputs.
*   **Inputs:** A tokenized natural language command and a state vector (which could be a compressed representation of visual input or other high-level sensory data).
*   **Architecture:** The state vector and token embeddings are projected into a common dimension and fed as a sequence into a transformer encoder.
*   **Output:** The transformer's output corresponding to the initial state token is passed through a linear layer to produce a `skill_embedding`. This embedding represents the next sub-task for the low-level controller to execute.

### 3.2. Low-Level Controller

The low-level controller is a simple Multi-Layer Perceptron (MLP).
*   **Inputs:** The `skill_embedding` from the high-level policy and the robot's current proprioceptive state vector.
*   **Architecture:** These inputs are concatenated and passed through a few fully connected layers.
*   **Output:** A vector of actions (e.g., joint torques or target positions), typically passed through a `tanh` function to constrain the output range.

### 3.3. Training Procedure

The model is trained end-to-end using a combination of behavioral cloning (BC) on expert trajectories and reinforcement learning (RL) to fine-tune. The key is that the training data comes from a diverse set of robot morphologies, each with its own low-level controller but sharing the same high-level policy.

### 3.4. Code Implementation

The following code provides a refined and runnable skeleton of the HOF-Model.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class HighLevelPolicy(nn.Module):
    """
    High-level policy that processes language and state to output a skill embedding.
    This component is morphology-agnostic.
    """
    def __init__(self, state_dim: int, embed_dim: int, vocab_size: int, hidden_size: int = 256):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, hidden_size)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.output_proj = nn.Linear(hidden_size, embed_dim)

    def forward(self, state: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state (torch.Tensor): State tensor of shape [batch_size, state_dim].
            token_ids (torch.Tensor): Language command tokens of shape [seq_len, batch_size].

        Returns:
            torch.Tensor: Skill embedding of shape [batch_size, embed_dim].
        """
        # Project state to the hidden size and add a sequence dimension
        state_feat = F.relu(self.state_proj(state)).unsqueeze(0)  # Shape: [1, batch_size, hidden_size]

        # Embed language tokens
        token_embeds = self.token_embedding(token_ids)  # Shape: [seq_len, batch_size, hidden_size]

        # Concatenate state and language to form the input sequence
        sequence = torch.cat([state_feat, token_embeds], dim=0)

        # Pass through transformer
        h = self.transformer(sequence)

        # Use the output corresponding to the state token as the basis for the skill
        skill_embed = self.output_proj(h[0]) # Shape: [batch_size, embed_dim]
        return skill_embed

class LowLevelController(nn.Module):
    """
    Low-level controller that maps a skill embedding and proprioception to actions.
    This component is morphology-specific.
    """
    def __init__(self, embed_dim: int, proprio_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim + proprio_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, skill_embed: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            skill_embed (torch.Tensor): Skill embedding from the high-level policy.
            proprio (torch.Tensor): Proprioceptive state of the robot.

        Returns:
            torch.Tensor: Predicted action for the robot.
        """
        x = torch.cat([skill_embed, proprio], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Use tanh to constrain actions to a reasonable range (e.g., [-1, 1])
        return torch.tanh(self.fc3(x))

class HierarchicalAgent(nn.Module):
    """A complete hierarchical agent combining the high and low-level policies."""
    def __init__(
        self,
        state_dim: int,
        vocab_size: int,
        embed_dim: int,
        proprio_dim: int,
        action_dim: int
    ):
        super().__init__()
        self.high_level_policy = HighLevelPolicy(state_dim, embed_dim, vocab_size)
        self.low_level_controller = LowLevelController(embed_dim, proprio_dim, action_dim)

    def forward(
        self,
        language_tokens: torch.Tensor,
        state: torch.Tensor,
        proprio: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs a full forward pass from language/state to action.
        """
        skill_embedding = self.high_level_policy(state, language_tokens)
        action = self.low_level_controller(skill_embedding, proprio)
        return action

# Example usage
if __name__ == "__main__":
    # Configuration
    batch_size = 4
    state_dim = 128      # High-level state (e.g., from camera)
    vocab_size = 1000    # Size of the language vocabulary
    embed_dim = 64       # Dimension of the skill embedding
    proprio_dim = 32     # Proprioceptive state (e.g., joint angles)
    action_dim = 8       # Action space dimension (e.g., joint torques)
    seq_len = 10         # Length of the language command

    # Instantiate the agent
    agent = HierarchicalAgent(state_dim, vocab_size, embed_dim, proprio_dim, action_dim)

    # Create dummy data
    tokens = torch.randint(0, vocab_size, (seq_len, batch_size))
    state = torch.randn(batch_size, state_dim)
    proprio = torch.randn(batch_size, proprio_dim)

    # Get action from the agent
    action = agent(tokens, state, proprio)

    print("Predicted action shape:", action.shape)
    assert action.shape == (batch_size, action_dim)
    print("Code skeleton runs successfully.")

```

## 4. Proposed Experiments

*   **Multi-Embodiment Training:** We will use a diverse suite of simulated robots (e.g., from Isaac Gym or MuJoCo Menagerie) to train a single HOF-Model. The tasks will range from simple locomotion to complex manipulation.
*   **Zero-Shot Transfer:** We will evaluate the trained model's ability to perform tasks with unseen combinations of language commands and environments.
*   **Fine-Tuning for New Morphologies:** The primary evaluation will be to take a new, held-out robot morphology. We will freeze the high-level policy and fine-tune only the low-level controller on a small number of demonstrations for that robot. We will measure the sample efficiency and final performance compared to training a policy from scratch.

## 5. Conclusion

The proposed Hierarchical Omni-Bodied Foundation Model presents a scalable and principled approach to building generalist robotic agents. By decoupling task-level understanding from embodiment-specific control, it paves the way for a single, powerful policy to be trained on vast, diverse datasets and then efficiently adapted to new robot platforms. This architecture represents a promising step towards the goal of a universal robot intelligence.
