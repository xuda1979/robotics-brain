# Online LENS: Neuromorphic Navigation with On-the-Fly Map Building

**Abstract**

Neuromorphic vision, particularly using event-based cameras, offers a path to extremely low-power robotic navigation. The Lightweight Event-based Navigation System (LENS) demonstrated the feasibility of this approach for place recognition, achieving remarkable energy efficiency. However, LENS operates on a pre-recorded map, limiting its use to known environments. This paper proposes Online LENS, a significant extension that incorporates online map building and sensor fusion for autonomous navigation in novel environments. Our system integrates an event camera, an Inertial Measurement Unit (IMU), and occasional standard camera frames to build and update a topological map on the fly. The core of our method is a spiking associative memory that performs robust loop closure detection, allowing the system to recognize previously visited locations and maintain a consistent map. The entire pipeline is designed to run on resource-constrained hardware, such as a microcontroller, by leveraging synaptic plasticity rules for efficient online learning.

## 1. Introduction

Autonomous navigation is a fundamental capability for mobile robots, but it often comes at a high computational and energy cost, especially when using traditional frame-based cameras. Event cameras, which report per-pixel brightness changes asynchronously, provide a low-power, high-dynamic-range alternative that is well-suited for dynamic environments.

The original LENS system showcased the potential of this paradigm, enabling a robot to localize itself along an 8km journey using only 180 kB of memory and consuming 99% less energy than conventional methods. It achieved this by compressing event streams into sparse, spatio-temporal signatures. However, its reliance on a pre-built map of these signatures prevents it from exploring and adapting to new, unknown spaces.

To address this limitation, we propose **Online LENS**. Our system extends the core principles of LENS with two key innovations:
1.  **Online SLAM:** We introduce a lightweight Simultaneous Localization and Mapping (SLAM) module. As the robot moves, it continuously generates new place signatures from event data and adds them as nodes to a topological map.
2.  **Robust Sensor Fusion:** To handle ambiguous situations (e.g., fast rotations or textureless areas), we fuse the event data with an IMU for motion estimation and low-frequency keyframes from a standard camera for absolute orientation and feature-rich place descriptions.

The entire system is orchestrated by a spiking neural network (SNN) that uses a spiking associative memory for efficient place recognition and loop closure. This allows the robot to build a globally consistent map of its environment in real-time, all while maintaining an extremely low power budget.

## 2. Related Work

*   **Event-Based Vision:** Research in this area has exploded, with applications ranging from high-speed tracking to HDR imaging. Several works have explored event-based SLAM, but many still rely on computationally expensive algorithms that are not suitable for microcontrollers.
*   **Neuromorphic Computing:** By mimicking the brain's use of spikes for computation, neuromorphic hardware and SNNs promise massive gains in energy efficiency. LENS is a prime example, and our work follows in this tradition.
*   **Visual Place Recognition:** The task of recognizing a previously visited place is central to SLAM (for loop closure). Methods range from bag-of-words models on image features to sequence-based matching. Our approach uses a spiking associative memory to perform this matching in a highly efficient manner.

## 3. Proposed Method: Online LENS

The Online LENS system consists of an event encoder, a sensor fusion module, and a spiking associative memory for map building and localization.

### 3.1. Event Stream Encoding

As in the original LENS, a stream of events from the camera is collected over a short time window and processed to form a compact signature. This can be a histogram of event locations and polarities or the output of a small SNN. This signature, **s**, represents the robot's current view.

### 3.2. Spiking Associative Memory for SLAM

The core of our SLAM module is the associative memory, which stores the signatures of all known places.
*   **Localization and Loop Closure:** When a new signature **s_t** is generated, it is compared against all signatures **m_i** in the memory **M**. The comparison is done using cosine similarity. If the similarity to the best-matching memory, **m***, exceeds a threshold, the system recognizes a previously visited place (a loop closure).
*   **Map Building:** If no match is found, the signature **s_t** is considered a new place and is added as a new node to the topological map and stored in the associative memory.
*   **Online Learning:** To store a new signature **s_t** at a memory index `idx`, we use a Hebbian-like update rule:
    **m_{idx} = (1 - \eta) \cdot m_{idx} + \eta \cdot s_t**
    where **\eta** is a small learning rate. This allows the memory to be updated efficiently on the fly.

### 3.3. Code Implementation

The code below provides a skeleton for the core components of Online LENS. It demonstrates the event encoding and the associative memory for place recognition and updating.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EventEncoder(nn.Module):
    """Encodes a window of events into a compact latent signature."""
    def __init__(self, input_size: int, latent_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, latent_size)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """
        Args:
            events (torch.Tensor): A flattened histogram of events for a time window.

        Returns:
            torch.Tensor: The latent signature.
        """
        x = F.relu(self.fc1(events))
        latent = torch.tanh(self.fc2(x))  # Tanh to bound the output
        return latent

class SpikingAssociativeMemory(nn.Module):
    """A memory that stores place signatures and performs winner-take-all retrieval."""
    def __init__(self, latent_size: int, memory_size: int):
        super().__init__()
        # Initialize memory with random data
        self.memory = nn.Parameter(torch.randn(memory_size, latent_size), requires_grad=False)
        self.memory_size = memory_size
        self.current_fill = 0 # Track how many slots are filled

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes cosine similarity and finds the best match.

        Args:
            latent (torch.Tensor): The current latent signature.

        Returns:
            A tuple containing the index of the best match and the similarity score.
        """
        if self.current_fill == 0:
            return torch.tensor([-1]), torch.tensor([-1.0])

        # Normalize both the input and the memory for cosine similarity
        latent_norm = F.normalize(latent, p=2, dim=1)
        memory_norm = F.normalize(self.memory[:self.current_fill], p=2, dim=1)

        # Compute similarity matrix
        similarity = latent_norm @ memory_norm.t()

        # Winner-takes-all retrieval
        best_similarity, match_idx = torch.max(similarity, dim=1)
        return match_idx, best_similarity

    def add_or_update_memory(self, latent: torch.Tensor, place_idx: int = -1, lr: float = 0.1):
        """
        Adds a new signature to memory or updates an existing one.

        Args:
            latent (torch.Tensor): The signature to store.
            place_idx (int): The index to update. If -1, adds a new place if space is available.
            lr (float): The learning rate for the update.
        """
        if place_idx == -1: # Add a new place
            if self.current_fill < self.memory_size:
                self.memory.data[self.current_fill] = latent.squeeze()
                self.current_fill += 1
        else: # Update an existing place using a Hebbian-like rule
            self.memory.data[place_idx] = (1 - lr) * self.memory.data[place_idx] + lr * latent.squeeze()


class LENSNavigationSystem(nn.Module):
    """The complete Online LENS system."""
    def __init__(self, input_size: int, latent_size: int, memory_size: int, recognition_threshold: float = 0.8):
        super().__init__()
        self.encoder = EventEncoder(input_size, latent_size)
        self.memory = SpikingAssociativeMemory(latent_size, memory_size)
        self.threshold = recognition_threshold

    def process_events(self, event_window: torch.Tensor) -> int:
        """
        Processes a new window of events, localizes, and updates the map.

        Args:
            event_window (torch.Tensor): The new event data.

        Returns:
            The ID of the recognized or newly created place.
        """
        latent = self.encoder(event_window)
        match_idx, similarity = self.memory(latent)

        if similarity.item() > self.threshold:
            # Recognized a known place (loop closure)
            # Update the existing memory to reinforce it
            self.memory.add_or_update_memory(latent, place_idx=match_idx.item())
            return match_idx.item()
        else:
            # New place detected
            new_place_id = self.memory.current_fill
            self.memory.add_or_update_memory(latent, place_idx=-1)
            return new_place_id


# Example usage
if __name__ == "__main__":
    input_size = 256  # Flattened size of event histogram
    latent_size = 64
    memory_size = 100

    model = LENSNavigationSystem(input_size, latent_size, memory_size)

    # Simulate processing a few windows of events
    for i in range(5):
        event_window = torch.randn(1, input_size)
        place_id = model.process_events(event_window)
        print(f"Processed window {i}, recognized/created place ID: {place_id}")

    # Simulate seeing a previous place again
    event_window = torch.randn(1, input_size) # A new observation
    # Manually make it similar to the first place for demonstration
    latent_to_store = model.encoder(event_window)
    model.memory.add_or_update_memory(latent_to_store, place_idx=0, lr=1.0)

    place_id = model.process_events(event_window)
    print(f"\nProcessed a similar window, recognized/created place ID: {place_id}")
    assert place_id == 0
    print("\nCode skeleton runs successfully.")

```

## 4. Proposed Experiments

*   **Dataset Evaluation:** We will first evaluate Online LENS on publicly available event-based datasets (e.g., MVSEC) to benchmark its localization and mapping accuracy against other event-based SLAM methods.
*   **Real-World Deployment:** The primary experiment will involve deploying the system on a mobile robot equipped with an event camera and an IMU. We will test its ability to autonomously explore and map an unknown indoor environment.
*   **Resource Consumption:** We will measure the power consumption and memory usage of the system running on a microcontroller (e.g., an STM32 or similar) to verify its efficiency.
*   **Robustness Tests:** We will test the system's robustness to challenging conditions, such as rapid motion, poor lighting, and dynamic obstacles.

## 5. Conclusion

Online LENS represents a significant step towards truly autonomous, low-power robotic navigation. By extending the principles of neuromorphic vision with online learning and sensor fusion, our proposed system can adapt to new environments in real-time. This work has the potential to unlock new applications for long-duration autonomy in robotics, from environmental monitoring to search and rescue.
