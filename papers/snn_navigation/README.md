# Spiking Neural Network Navigation

## Summary
Inspired by the brain's hippocampus and grid cells, spiking neural networks can perform place recognition using event-based sensors with very low energy consumption. A recent study developed SNN modules that detect place by compressing camera input into spatio-temporal spikes and combining them in an ensemble to improve accuracy by 41% while using sequences of images【354383436640087†L37-L74】.

## Proposed Idea
This project proposes to extend neuromorphic SNN navigation by introducing a continuous-learning SNN that updates its weights online as the robot traverses new environments. It will use a memory-efficient synaptic plasticity rule (e.g., spike-timing-dependent plasticity) to adapt place cells while preserving previous knowledge via synaptic consolidation. The navigation system will couple an event-camera front-end with a spiking network to localize and map simultaneously on microcontrollers.

## Code skeleton
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Spik)ingNeuron(nn.M)odule:
    def __init__(self, threshold=1.0, decay=0.9):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.register_buffer('mem', torch.zeros(1))

    def forward(self, input):
        # membrane potential update
        self.mem = self.mem * self.decay + input
        spike = (self.mem >= self.threshold).float()
        # reset after spike
        self.mem = self.mem * (1 - spike)
        return spike

class SNNPla)ceCell(nn.Mo)dule:
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.neuron = SpikingNeuron()

    def forward(self, x):
        u = self.fc1(x)
        spikes = self.neuron(u)
        return spikes

class ContinuousSNNSlam(nn.Module:
    def __init__(self, input_size, hidden_size, num_places):
        super().__init__()
        self.place_cells = nn.ModuleList([
            SNNPlaceCell(input_size, hidden_size) for _ in range(num_places)
        ])

    def forward(self, event_frames):
        # event_frames: sequence of flattened event frames
        place_activations = []
        for frame in event_frames:
            activations = torch.stack([cell(frame) for cell in self.place_cells])
            place_activations.append(activations)
        return torch.stack(place_activations)

    def update_synapses(self, frame, target_place, lr=0.001):
        # dummy online STDP-like update for demonstration
        for i, cell in enumerate(self.place_cells):
            pred = cell(frame)
            error = (pred - (target_place == i).float())
            # gradient descent update on weights
            cell.fc1.weight.data -= lr * error.unsqueeze(1) * frame.unsqueeze(0)

# Example usage:
# event_frames = torch.randn(seq_len, input_size)
# model = ContinuousSNNSlam(input_size=64, hidden_size=32, num_places=10)
# activations = model(event_frames)
```

This skeleton defines a simple spiking neuron model and a place cell network. The `update_synapses` method illustrates how online learning could adjust weights using a simple rule. A real implementation would integrate event-camera preprocessing, more realistic neuron dynamics, and STDP learning.
