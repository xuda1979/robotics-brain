## Lifelong World‑Model Reinforcement Learning

**Summary**

Dreamer V3 is a general model‑based reinforcement learning algorithm that learns a world model of the environment and improves the agent’s behavior by imagining future trajectories. It uses normalized neural networks and transformation functions to stabilise training and outperforms specialised model‑based or model‑free methods across more than 150 tasks【932370549811763†L139-L149】. Separate research on lifelong reinforcement learning proposes building **knowledge spaces** using Bayesian non‑parametrics and language embeddings to allow an agent to accumulate knowledge from a continuous stream of tasks and reuse it for new tasks【906133705229545†L140-L155】.

**Proposed Idea**

We propose **Lifelong World‑Model Reinforcement Learning**, combining Dreamer‑style world models with knowledge spaces for lifelong learning. The algorithm learns a latent world model that predicts future observations, rewards and termination flags. Alongside, it maintains a growing library of skills represented as a knowledge space indexed by task embeddings derived from language instructions and environment features. A meta‑controller selects which existing skill to reuse and when to create a new skill, using Bayesian nonparametric priors such as the Dirichlet Process. During interaction with each task, the agent imagines trajectories using the world model and updates both the skill policy and the world model. When a new task arrives, the agent queries the knowledge space for similar tasks and initializes the world model and policy from the closest match, enabling fast adaptation and lifelong accumulation of knowledge.

**Code Skeleton**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class WorldModel(nn.Module):
    """
    Latent world model for predicting next state, reward, and discount.
    """
    def __init__(self, obs_dim, action_dim, latent_dim=128):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, latent_dim)
        self.action_encoder = nn.Linear(action_dim, latent_dim)
        self.dynamics = nn.GRUCell(latent_dim * 2, latent_dim)
        self.reward_head = nn.Linear(latent_dim, 1)
        self.discount_head = nn.Linear(latent_dim, 1)

    def forward(self, obs, action, hidden):
        x = torch.cat([self.obs_encoder(obs), self.action_encoder(action)], dim=-1)
        hidden = self.dynamics(x, hidden)
        reward = self.reward_head(hidden)
        discount = torch.sigmoid(self.discount_head(hidden))
        return hidden, reward, discount

class SkillPolicy(nn.Module):
    """
    Policy conditioned on latent state and task embedding.
    """
    def __init__(self, latent_dim, task_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + task_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, latent, task_embed):
        x = torch.cat([latent, task_embed], dim=-1)
        return self.fc(x)

class KnowledgeSpace:
    """
    Simple knowledge space storing task embeddings and associated skill policies.
    """
    def __init__(self):
        self.task_embeddings = []
        self.policies = []

    def query(self, task_embed, k=1):
        """
        Find the closest existing task embeddings.
        """
        if not self.task_embeddings:
            return None
        dists = [torch.norm(task_embed - e) for e in self.task_embeddings]
        idx = torch.argmin(torch.tensor(dists))
        return self.policies[idx]

    def add_skill(self, task_embed, policy):
        self.task_embeddings.append(task_embed.detach().clone())
        self.policies.append(policy)

# Example usage
if __name__ == "__main__":
    obs_dim, action_dim, task_dim = 16, 4, 8
    world_model = WorldModel(obs_dim, action_dim)
    skill_policy = SkillPolicy(latent_dim=128, task_dim=task_dim, action_dim=action_dim)
    knowledge_space = KnowledgeSpace()

    # fake task embedding and state
    task_embed = torch.randn(task_dim)
    latent = torch.randn(1, 128)
    action = skill_policy(latent, task_embed)
    knowledge_space.add_skill(task_embed, skill_policy)
    print("Sampled action:", action)
```
