# Papers

This folder stores research papers and drafts proposing new ideas and algorithms for the next generation of robotics brains.

## Current Research Papers

Below is a summary of the current research proposals in this directory.

*   **[Cross-Modal Diffusion Control](./diffusion_policy_control/)**: Proposes Cross-Modal Diffusion Control (CMDC), a framework for robotic manipulation that integrates vision, tactile, and proprioceptive inputs using a conditional diffusion process to generate robust action trajectories.

*   **[Height-Dimension Neural Networks](./height_dimension_network/)**: Proposes Height-Dimension Neural Networks (HD-Nets), inspired by the brain's cortex, which add intra-layer vertical connections for recurrent processing to improve temporal reasoning and memory in robotics tasks.

*   **[Hierarchical Foundation Model](./hierarchical_foundation_model/)**: Proposes a Hierarchical Omni-Bodied Foundation Model (HOF-Model) that decouples high-level, language-conditioned task planning from low-level, morphology-specific motor control to enable rapid adaptation to new robot bodies.

*   **[Neuromorphic Navigation (LENS)](./lens_navigation/)**: Proposes Online LENS, an extension to the low-power LENS navigation system, that incorporates online map building and sensor fusion (event camera, IMU) using a spiking associative memory for autonomous navigation in novel environments.

*   **[Lifelong World-Model RL](./lifelong_world_model_rl/)**: Proposes Lifelong Dreamer, an architecture that integrates a Dreamer-style world model with a non-parametric knowledge space to enable continual learning, allowing the agent to accumulate knowledge and adapt to new tasks quickly.

*   **[Lp-Convolution Vision](./lp_convolution_vision/)**: Proposes using Lp-Convolution, a brain-inspired variant of standard convolution where layers can dynamically change their kernel shape by learning a `p` parameter, to create more robust and efficient robotic vision systems.

*   **[Multi-Agent Cognitive Graph](./multi_agent_cognitive_graph/)**: Introduces the Multi-Agent Cognitive Graph (MAC-Graph), a framework for learning coordinated behaviors in robotic teams by representing the system as a dynamic graph and using a GNN to facilitate communication and shared understanding.

*   **[Spiking Neural Network Navigation](./snn_navigation/)**: Proposes a neuromorphic SLAM system that leverages Spiking Neural Networks (SNNs) and event-based cameras to achieve continuous, online learning, using a bio-inspired STDP rule to map novel environments with extreme energy efficiency.

*   **[Adaptive Swarm Control](./swarm_control/)**: Proposes an Adaptive Swarm Control framework for cyborg insects that enables dynamic leader election and decentralized, learned coordination to create more robust, resilient, and efficient biobot swarms.

---

## Adding a New Paper

To add a paper, create a new subfolder (e.g. `papers/my_new_idea/`) containing your manuscript (LaTeX, Markdown, or PDF) and supplementary materials. The `README.md` for the new paper should follow the format of the existing papers, including an Abstract, Introduction, and other relevant sections. After adding the paper, please also add a summary to the "Current Research Papers" list in this file.
