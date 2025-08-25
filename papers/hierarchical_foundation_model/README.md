# Hierarchical Omni-Bodied Foundation Model

## Summary
Skild AI researchers argue that general-purpose robot intelligence requires a hierarchical architecture with separate high-level and low-level components. The high-level component reasons about goals and tasks, while the low-level component executes motor actions and adapts to the robot’s specific morphology【615241397368816†L14-L32】. Training across diverse robot morphologies enables the high-level policy to generalize and transfer skills.

## Proposed Idea
We propose a hierarchical foundation model that combines a language-conditioned high-level policy with morphology-conditioned low-level controllers. The high-level policy, implemented as a transformer, outputs skill embeddings conditioned on natural language commands and a latent representation of the robot’s current state. A low-level controller decodes these embeddings into joint torques using a proprioceptive state encoder. During training, both components are co-optimized on a suite of simulated robots via reinforcement learning and behavior cloning. At inference time, the high-level policy can be frozen and reused across new robots by fine-tuning only the low-level controller.

## Code skeleton
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HighLevelPolicy(nn.Module:
    def __init__(self, state_dim, embed_dim, vocab_size, hidden_size=256):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, hidden_size)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4), num_layers=3
        )
        self.fc = nn.Linear(hidden_size, embed_dim)

    def forward(self, state, token_ids):
        # state: (batch, state_dim), token_ids: (seq_len, batch)
        state_feat = F.relu(self.state_proj(state)).unsqueeze(0)  # seq_len=1
        token_embeds = self.token_embedding(token_ids)  # (seq_len, batch, hidden)
        sequence = torch.cat([state_feat, token_embeds], dim=0)
        h = self.transformer(sequence)
        skill_embed = self.fc(h[0])  # use the first token (state) output
        return skill_embed

class LowLevelController(nn.Module:
    def __init__(self, embed_dim, proprio_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim + proprio_dim, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, skill_embed, proprio):
        x = torch.cat([skill_embed, proprio], dim=-1)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))

class HierarchicalAgent(nn.Module:
    def __init__(self, state_dim, vocab_size, embed_dim, proprio_dim, action_dim):
        super().__init__()
        self.high = HighLevelPolicy(state_dim, embed_dim, vocab_size)
        self.low = LowLevelController(embed_dim, proprio_dim, action_dim)

    def forward(self, language_tokens, state, proprio):
        skill = self.high(state, language_tokens)
        action = self.low(skill, proprio)
        return action

# Example usage
# agent = HierarchicalAgent(state_dim=128, vocab_size=1000, embed_dim=64, proprio_dim=64, action_dim=12)
# tokens = torch.randint(0, 1000, (5, batch_size))
# state = torch.randn(batch_size, 128)
# proprio = torch.randn(batch_size, 64)
# action = agent(tokens, state, proprio)
```

This skeleton defines a high-level transformer policy conditioned on language and state, and a low-level controller conditioned on both the skill embedding and proprioception. Training across multiple robot morphologies would require additional wrappers to simulate diverse robots and to fine-tune the low-level controller.
