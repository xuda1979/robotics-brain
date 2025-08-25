## Height‑Dimension Neural Networks

**Summary**

Recent research proposes adding a "height" dimension to neural networks by incorporating intra‑layer feedback loops and vertical connections, inspired by the columnar structure of the brain's cortex. This third dimension allows information to flow up and down within a layer, enabling neurons to share context and implement recurrent processing. Experiments show that networks with a height dimension exhibit richer interactions and improvements in memory, perception and cognitive capabilities【374616379590202†L100-L129】.

**Proposed Idea**

We propose to apply height‑dimension neural networks to robotics perception and control. Each layer will be extended with vertical columns that allow recurrent information flow across temporal and spatial scales. For example, a vision model could include vertical loops within convolutional layers to integrate context across frames, improving temporal consistency and robustness. Similarly, a control network could use height connections to maintain a memory of previous actions and environmental states. We also propose gating mechanisms to modulate vertical interactions based on task demands and meta‑learning to adapt the height dimension for different robot morphologies.

**Code Skeleton**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HeightLayer(nn.Module):
    """
    Height-Dimension Layer with vertical recurrent connections.
    """
    def __init__(self, input_dim, hidden_dim, height_depth=3):
        super().__init__()
        self.height_depth = height_depth
        # horizontal feedforward weights
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # vertical recurrent weights
        self.vert_weights = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(height_depth)
        ])
        self.gate = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(height_depth)
        ])

    def forward(self, x):
        """
        x: [batch, input_dim]
        Returns: [batch, hidden_dim]
        """
        h = F.relu(self.input_proj(x))
        # propagate through height dimension
        for i in range(self.height_depth):
            v = F.relu(self.vert_weights[i](h))
            g = torch.sigmoid(self.gate[i](h))
            h = h + g * v  # gated vertical update
        return h

class HeightNetwork(nn.Module):
    """
    Simple network using height‑dimension layers.
    """
    def __init__(self, obs_dim, hidden_dim, action_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            HeightLayer(obs_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        h = obs
        for layer in self.layers:
            h = layer(h)
        return self.output_proj(h)

# Example usage
if __name__ == "__main__":
    obs_dim = 32
    hidden_dim = 64
    action_dim = 8
    net = HeightNetwork(obs_dim, hidden_dim, action_dim)
    obs = torch.randn(2, obs_dim)
    actions = net(obs)
    print("Predicted actions:", actions)
```
