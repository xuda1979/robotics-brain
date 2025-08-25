# Lifelong Dreamer: A World Model for Continual Reinforcement Learning

**Abstract**

Model-based reinforcement learning agents, such as Dreamer, have achieved remarkable performance by learning a world model of their environment and planning via imagination. However, these agents are typically trained on a single task and forget their knowledge when faced with a new one. In parallel, the field of lifelong learning has developed methods for knowledge accumulation and transfer, but often in simpler, stateless settings. This paper proposes **Lifelong Dreamer**, an architecture that integrates a Dreamer-style world model with a non-parametric knowledge space to enable true continual learning. The agent learns a latent world model while simultaneously building a growing library of skills. A meta-controller, guided by Bayesian non-parametric priors, decides when to reuse an existing skill and when to create a new one. When encountering a new task, the agent queries its knowledge space for the most relevant prior skill, allowing it to initialize its policy and world model for rapid adaptation. This synergy allows the agent to not only master individual tasks but also to build upon its past experiences to learn new tasks more efficiently over its entire lifetime.

## 1. Introduction

A hallmark of intelligence is the ability to learn continually from a stream of experiences, retaining and reusing knowledge to master new challenges. While deep reinforcement learning (RL) has produced agents with superhuman performance in specific domains, they suffer from "catastrophic forgetting"—when trained on a new task, they abruptly lose proficiency in previous ones.

Model-based RL agents like Dreamer offer a promising foundation. By learning a world model, they can learn efficiently from a compact, latent representation of the environment. However, this world model and the associated policy are task-specific. How can an agent learn a new world model for a new task without forgetting the old one?

We propose **Lifelong Dreamer**, a novel framework that endows a world-model-based agent with a mechanism for lifelong knowledge accumulation. Our core idea is to maintain a **knowledge space**, a non-parametric, growing library of skills and their associated world model parameters. This space is indexed by task embeddings derived from language instructions or environmental cues.

When faced with a task, a meta-controller assesses its novelty.
*   If the task is similar to a known one, the agent retrieves the corresponding skill and world model from the knowledge space, fine-tuning them for the current context.
*   If the task is novel, the agent creates a new "slot" in the knowledge space, initializing a new skill and world model, and learns it from scratch.

This process, guided by Bayesian non-parametric models, allows the agent to gracefully expand its repertoire of skills over its lifetime, leading to faster learning on new tasks and preventing catastrophic forgetting.

## 2. Related Work

*   **Model-Based Reinforcement Learning:** Algorithms like World Models, PlaNet, and Dreamer learn a predictive model of the environment and use it for planning or policy learning. Dreamer V3, the latest iteration, has demonstrated state-of-the-art performance across a wide range of domains. Our work uses Dreamer's core components as the "engine" for learning within a single task.
*   **Lifelong and Continual Learning:** This field focuses on preventing catastrophic forgetting in neural networks. Methods include regularization-based approaches (e.g., Elastic Weight Consolidation), parameter isolation methods (allocating different parameters for different tasks), and replay-based methods. Our knowledge space is a form of parameter isolation.
*   **Bayesian Non-parametrics:** Models like the Dirichlet Process or Chinese Restaurant Process provide a principled mathematical framework for deciding when data belongs to an existing cluster or a new one. We use this to drive the meta-controller's decision to reuse or create a skill.

## 3. Proposed Method: Lifelong Dreamer

The Lifelong Dreamer agent is composed of three main parts: a latent world model, a skill-conditioned policy, and a knowledge space managed by a meta-controller.

### 3.1. Latent World Model

Following Dreamer, we learn a world model in a compact latent space. The model has components to predict the next latent state, reward, and termination flags from the current state and action. This allows the agent to "dream" or imagine future trajectories for policy learning.

### 3.2. Knowledge Space and Skill Policies

The knowledge space is the agent's long-term memory. It is a dictionary-like structure that stores tuples of `(task_embedding, skill_policy, world_model_params)`.
*   **Task Embedding:** A vector representation of the current task, derived from a language instruction or a learned summary of the environment's properties.
*   **Skill Policy:** A neural network that maps a latent state from the world model and the task embedding to an action.
*   **World Model Parameters:** The specific parameters of the world model's dynamics that are tailored to this task.

### 3.3. Meta-Controller

The meta-controller decides which skill to use. Given a new task embedding, it computes its similarity to all embeddings in the knowledge space. Using a Bayesian non-parametric prior (like a Dirichlet Process), it calculates the probability that this task belongs to an existing skill cluster versus forming a new one. This determines whether the agent reuses and fine-tunes an existing skill or initializes a new one.

### 3.4. Code Implementation

This skeleton outlines the core components: the world model, the skill policy, and a simplified knowledge space.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class WorldModel(nn.Module):
    """
    A simplified latent world model inspired by Dreamer.
    It predicts the next latent state, reward, and continuation probability.
    """
    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int = 128):
        super().__init__()
        # Note: A full Dreamer model is much more complex (e.g., using RSSM).
        self.obs_encoder = nn.Linear(obs_dim, latent_dim)
        self.action_encoder = nn.Linear(action_dim, latent_dim)
        self.dynamics = nn.GRUCell(latent_dim * 2, latent_dim)
        self.reward_head = nn.Linear(latent_dim, 1)
        self.discount_head = nn.Linear(latent_dim, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([self.obs_encoder(obs), self.action_encoder(action)], dim=-1)
        hidden = self.dynamics(x, hidden)
        reward = self.reward_head(hidden)
        discount = torch.sigmoid(self.discount_head(hidden)) # Predict probability of continuation
        return hidden, reward, discount

class SkillPolicy(nn.Module):
    """A policy conditioned on the current latent state and a task embedding."""
    def __init__(self, latent_dim: int, task_dim: int, action_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + task_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, latent: torch.Tensor, task_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([latent, task_embed], dim=-1)
        return torch.tanh(self.fc(x)) # tanh for action bounds

class KnowledgeSpace:
    """
    A simplified knowledge space that stores and retrieves skills.
    A full implementation would use Bayesian non-parametrics.
    """
    def __init__(self):
        self.task_embeddings: List[torch.Tensor] = []
        self.policies: List[SkillPolicy] = []
        # In a full implementation, we'd also store world model parameters

    def query(self, task_embed: torch.Tensor, threshold: float = 0.9) -> Optional[SkillPolicy]:
        """
        Finds the closest skill policy if similarity is above a threshold.
        """
        if not self.task_embeddings:
            return None

        sims = [F.cosine_similarity(task_embed, e, dim=0) for e in self.task_embeddings]

        if not sims:
            return None

        best_sim, idx = torch.max(torch.tensor(sims), dim=0)

        if best_sim > threshold:
            return self.policies[idx]
        return None

    def add_skill(self, task_embed: torch.Tensor, policy: SkillPolicy):
        """Adds a new task embedding and its associated policy to the space."""
        self.task_embeddings.append(task_embed.detach().clone())
        self.policies.append(policy)

# Example usage
if __name__ == "__main__":
    obs_dim, action_dim, task_dim, latent_dim = 16, 4, 8, 128

    knowledge_space = KnowledgeSpace()

    # --- Task 1 ---
    print("--- Task 1: Learning 'Fetch' ---")
    task1_embed = torch.randn(1, task_dim)
    # Check if a similar skill exists (it doesn't)
    retrieved_policy = knowledge_space.query(task1_embed.squeeze(0))
    assert retrieved_policy is None
    print("No similar skill found. Creating a new one.")

    # Create and add a new skill for this task
    task1_policy = SkillPolicy(latent_dim, task_dim, action_dim)
    knowledge_space.add_skill(task1_embed.squeeze(0), task1_policy)
    print("Added 'Fetch' skill to knowledge space.")

    # --- Task 2 ---
    print("\n--- Task 2: Learning 'Push' ---")
    task2_embed = torch.randn(1, task_dim)
    retrieved_policy = knowledge_space.query(task2_embed.squeeze(0))
    assert retrieved_policy is None
    print("No similar skill found. Creating a new one.")
    task2_policy = SkillPolicy(latent_dim, task_dim, action_dim)
    knowledge_space.add_skill(task2_embed.squeeze(0), task2_policy)
    print("Added 'Push' skill to knowledge space.")

    # --- Task 3 ---
    print("\n--- Task 3: Encountering 'Fetch' again ---")
    # This embedding is very similar to task 1
    task3_embed = task1_embed + torch.randn_like(task1_embed) * 0.01
    retrieved_policy = knowledge_space.query(task3_embed.squeeze(0))
    assert retrieved_policy is not None
    print("Found a similar skill! Reusing the 'Fetch' policy.")

    # Use the retrieved policy
    latent = torch.randn(1, latent_dim)
    action = retrieved_policy(latent, task3_embed)
    print("Sampled action using retrieved policy:", action)
    print("\nCode skeleton runs successfully.")
```

## 4. Proposed Experiments

*   **Continual Learning Benchmarks:** We will evaluate Lifelong Dreamer on standard continual learning benchmarks adapted for RL, such as sequences of tasks from Meta-World or Procgen. We will measure performance on all tasks at the end of training to quantify knowledge retention and forward transfer (how quickly new tasks are learned).
*   **Comparison to Baselines:** We will compare against several baselines: (1) Fine-tuning a single Dreamer model (expected to forget), (2) Training separate Dreamer models for each task (upper bound), and (3) Standard continual learning methods like EWC applied to Dreamer.
*   **Ablation Studies:** We will analyze the contribution of the key components, such as the effect of the Bayesian non-parametric meta-controller versus a simpler similarity threshold.

## 5. Conclusion

Lifelong Dreamer offers a path toward creating RL agents that learn and adapt continuously throughout their existence. By combining the powerful intra-task learning of world models with the inter-task knowledge management of a lifelong learning framework, our approach addresses the critical challenge of catastrophic forgetting. This architecture could enable the development of truly generalist agents that accumulate a rich repertoire of skills and knowledge from a lifetime of experience.
