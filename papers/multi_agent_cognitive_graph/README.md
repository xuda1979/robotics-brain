# Learning to Collaborate: A Multi-Agent Cognitive Graph for Embodied AI

**Abstract**

While single-agent embodied AI has made significant strides, intelligence in the real world is often a collective endeavor. From search-and-rescue teams to automated warehouses, multi-agent systems must coordinate to achieve complex goals. This paper introduces the **Multi-Agent Cognitive Graph (MAC-Graph)**, a novel framework for learning coordinated behaviors in teams of robots. We represent the multi-agent system as a dynamic graph, where each node is an agent and edges represent communication or spatial relationships. A Graph Neural Network (GNN) operates on this graph, allowing agents to share information and develop a shared understanding of the environment and task. This shared context is then used by individual agents' policies to produce coordinated actions. The entire system, from perception to communication to action, is designed to be learned end-to-end through multi-agent reinforcement learning. We propose that this approach will enable emergent, complex collaborative behaviors that are difficult to hand-engineer, paving the way for more capable and flexible robotic teams.

## 1. Introduction

Most research in embodied AI focuses on the single-agent paradigm: one robot, one brain, one task. Yet, many critical real-world applications fundamentally require multiple agents to work together. Effective collaboration requires agents to reason about each other: "What is my teammate doing?", "What do they know that I don't?", "How can I help them achieve their goal?".

Traditional approaches to multi-agent coordination rely on hand-crafted communication protocols and decision rules. These methods are often brittle and fail to scale to complex, dynamic environments. We propose a learning-based approach that allows robotic teams to *learn* how to coordinate.

Our framework, the Multi-Agent Cognitive Graph (MAC-Graph), models the team of agents and their relationships as a graph.
*   **Nodes:** Each agent is a node, containing its internal state, which is derived from its own observations (e.g., camera images, laser scans).
*   **Edges:** Edges in the graph represent potential for interaction. An edge might exist if two agents are within communication range or within a certain physical distance. The graph's structure is dynamic, changing as the agents move.

A Graph Neural Network (GNN) serves as the "collective brain" of the team. By passing messages along the edges of the graph, the GNN enriches each agent's local view with context from its teammates. The output of the GNN is a "socially-aware" state representation for each agent, which is then fed into a local policy network to select an action. This allows an agent's decision to be influenced by the states and intentions of its neighbors, leading to emergent cooperation.

## 2. Related Work

*   **Multi-Agent Reinforcement Learning (MARL):** MARL is a large field, with common paradigms including centralized training with decentralized execution (CTDE), and value decomposition methods. Our work fits within the CTDE framework, where the GNN is part of the centralized training process.
*   **Graph Neural Networks:** GNNs have become a powerful tool for processing relational data. They have been applied to multi-agent systems in various contexts, from modeling traffic to simulating particle physics. We apply them specifically to the problem of learning coordination strategies for embodied agents.
*   **Team-Based Embodied AI:** There is growing interest in multi-agent embodied AI, particularly in simulation environments like Habitat and RoboCup. Our work provides a general framework for learning coordination that could be applied in these and other settings.

## 3. Proposed Method: The MAC-Graph Framework

The MAC-Graph framework has two key components: the dynamic cognitive graph and a GNN-based communication module.

### 3.1. Dynamic Cognitive Graph Construction

At each timestep, a cognitive graph **G = (V, E)** is constructed.
*   **V:** The set of nodes, where each node **v_i** corresponds to agent **i**. The initial feature vector for each node, **h_i**, is produced by an encoder processing the agent's local observations.
*   **E:** The set of edges. An edge **(i, j)** exists in **E** if agents **i** and **j** can interact. In the simplest case, this is determined by a distance threshold: `||pos_i - pos_j|| < d_comms`. The adjacency matrix **A** represents these edges.

### 3.2. GNN for Communication and Coordination

A GNN is used to perform message passing on the cognitive graph. For each node **i**, it aggregates messages from its neighbors **N(i)** and updates its hidden state. This can be formulated as:

1.  **Message Aggregation:**
    **m_i = एGG_{j \in N(i)} M(h_i, h_j)**
    where **M** is a message function (e.g., an MLP) and **AGG** is an aggregation function (e.g., sum or mean).
2.  **State Update:**
    **h'_i = U(h_i, m_i)**
    where **U** is an update function (e.g., a GRU cell).

After one or more rounds of message passing, the final node states **h'** represent a shared understanding of the team's state. Each agent **i** then uses its updated state **h'_i** to inform its action selection policy, **π(a_i | h'_i)**.

### 3.3. Code Implementation

The following code provides a practical PyTorch implementation of the `CognitiveGraph` GNN module. It operates on tensors representing the states and adjacency matrix of the agent team.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class CognitiveGraphGNN(nn.Module):
    """
    A Graph Neural Network for multi-agent coordination.
    It takes all agent states and an adjacency matrix, and performs message passing.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        # Message function: learns to create a message from two agent states
        self.message_fn = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU()
        )
        # Update function: takes the aggregated message and updates the agent's state
        self.update_fn = nn.GRUCell(input_size=hidden_dim, hidden_size=state_dim)

    def forward(self, agent_states: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Performs one round of message passing.

        Args:
            agent_states (torch.Tensor): Tensor of agent states, shape [N, state_dim].
            adjacency (torch.Tensor): Adjacency matrix, shape [N, N].

        Returns:
            torch.Tensor: Updated agent states, shape [N, state_dim].
        """
        num_agents = agent_states.shape[0]

        # Expand states for message creation: [N, N, state_dim]
        # state_i_expanded(i,j) is state of agent i, state_j_expanded(i,j) is state of agent j
        state_i_expanded = agent_states.unsqueeze(1).expand(num_agents, num_agents, -1)
        state_j_expanded = agent_states.unsqueeze(0).expand(num_agents, num_agents, -1)

        # Create all possible messages
        all_pairs = torch.cat([state_i_expanded, state_j_expanded], dim=-1)
        all_messages = self.message_fn(all_pairs) # Shape: [N, N, hidden_dim]

        # Mask messages by adjacency matrix (zero out messages between non-neighbors)
        masked_messages = all_messages * adjacency.unsqueeze(-1)

        # Aggregate messages for each agent (sum over incoming messages)
        aggregated_messages = torch.sum(masked_messages, dim=1) # Shape: [N, hidden_dim]

        # Update agent states using the GRU cell
        new_agent_states = self.update_fn(aggregated_messages, agent_states)

        return new_agent_states

class MultiAgentSystem:
    """
    A simple wrapper for a multi-agent system using the Cognitive Graph.
    """
    def __init__(self, num_agents: int, state_dim: int):
        self.num_agents = num_agents
        self.state_dim = state_dim
        # Agent states could be initialized from encoders processing observations
        self.agent_states = torch.randn(num_agents, state_dim)
        self.agent_positions = torch.randn(num_agents, 2)
        self.cog_graph_gnn = CognitiveGraphGNN(state_dim, hidden_dim=64)

    def get_adjacency(self, comms_range: float) -> torch.Tensor:
        """Builds adjacency matrix based on agent proximity."""
        dist_matrix = torch.cdist(self.agent_positions, self.agent_positions)
        adjacency = (dist_matrix < comms_range).float()
        # Agents don't message themselves
        adjacency.fill_diagonal_(0)
        return adjacency

    def step(self):
        """Simulates one step of coordination."""
        adjacency = self.get_adjacency(comms_range=1.5)
        print("Adjacency Matrix:\n", adjacency)

        updated_states = self.cog_graph_gnn(self.agent_states, adjacency)
        self.agent_states = updated_states
        print("\nUpdated agent states (first 5 features):\n", self.agent_states[:, :5])

# Example usage
if __name__ == "__main__":
    num_agents = 4
    state_dim = 16

    system = MultiAgentSystem(num_agents, state_dim)
    system.step()
    print("\nCode skeleton runs successfully.")

```

## 4. Proposed Experiments

*   **Cooperative Navigation:** We will evaluate the MAC-Graph framework on cooperative navigation and exploration tasks in a simulated environment like Habitat. A team of agents will be tasked with exploring a space and finding a set of objects. We will measure the time to completion and compare against decentralized baselines and non-learning-based methods.
*   **Predator-Prey Scenarios:** We will use a classic predator-prey scenario to analyze the emergent strategies. A team of "predator" agents must learn to coordinate to capture one or more "prey" agents. We will analyze the learned communication protocols and strategies.
*   **Ablation Studies:** We will perform ablations to understand the importance of communication. We will compare the performance of the full model against a version where the adjacency matrix is always zero (i.e., no communication) and a version with a fully connected graph (i.e., all agents can always communicate).

## 5. Conclusion

The Multi-Agent Cognitive Graph provides a flexible and powerful framework for learning complex coordination behaviors in robotic teams. By explicitly modeling the system as a dynamic graph and using GNNs to learn communication protocols, our approach can unlock emergent collaboration that is robust and scalable. This represents a critical step towards deploying teams of intelligent embodied agents that can work together to solve complex, real-world problems.
